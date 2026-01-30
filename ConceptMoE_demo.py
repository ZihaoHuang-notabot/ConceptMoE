import torch
import torch.nn.functional as F
from torch import nn
from einops import repeat, rearrange

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except:
    print("mamba_chunk_scan_combined not found")
    mamba_chunk_scan_combined = None


class ChunkModule(nn.Module):
    def __init__(self, hidden_size=2048, compress_ratio=2):
        super().__init__()
        self.q_proj_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.R = compress_ratio
    
    def forward(self, hidden_states):
        cos_sim = torch.einsum(
                    "b l d, b l d -> b l",
                    F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1),
                    F.normalize(self.k_proj_layer(hidden_states[:, 1:]), dim=-1),
                            )       # shape [1023,]
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)  # shape [1023,]
        # Force boundary probability of the first element to 1.0
        PAD_PROB = 1.0
        boundary_prob = F.pad(boundary_prob, (1, 0), "constant", PAD_PROB) # shape [1024,]

        G = boundary_prob.mean()    # for aux loss

        selected_idx = torch.zeros_like(boundary_prob, dtype=torch.long)
        boundary_mask = boundary_prob >= 0.5
        selected_idx[..., boundary_mask] = 1
        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)

        Freq = boundary_mask.float().mean()    # for aux loss
        aux_loss = self.R/(self.R-1) * ((self.R-1)*Freq*G+(1-Freq)*(1-G))
            


        selected_probs = boundary_prob.gather(
                    dim=-1, index=selected_idx.unsqueeze(-1)
                )  # (shape hidden_states.shape[:-1], 1)
        return boundary_prob, boundary_mask, selected_probs, aux_loss
    
def get_seq_idx(cu_seqlens, device=None):
    seq_idx = torch.zeros(cu_seqlens[-1], dtype=torch.long, device=device)
    seq_idx[cu_seqlens[:-1]] = 1
    seq_idx = (torch.cumsum(seq_idx, dim=0) - 1).unsqueeze(0).int()

    return seq_idx

class DechunkModule(nn.Module):
    def __init__(self, block_size=256, headdim=32, hidden_size=2048):
        super().__init__()
        self.block_size = block_size
        self.headdim = headdim
        assert hidden_size % self.headdim == 0
        self.nheads = hidden_size // self.headdim
        self.hidden_size = hidden_size
        self.dtype = torch.float32

    def forward(self, concept, selected_probs, boundary_mask):
        if mamba_chunk_scan_combined is None:
            concept_prob = selected_probs[boundary_mask].unsqueeze(0)     # shape [1,522,1]
            concept_merge = torch.zeros_like(concept)
            # For ease of understanding, this is written in for-loop form. In practice, it can be accelerated through parallel scan.
            concept_merge[:,0] = concept[:,0]                   # shape [522, 2048]
            for i in range(1, concept.shape[1]):
                concept_merge[:,i] = concept_merge[:,i-1]*(1-concept_prob[:,i]) + concept[:,i] * concept_prob[:,i]
        else:
            cu_seqlens = torch.tensor([0, concept.shape[1]], device=concept.device)
            seq_idx = get_seq_idx(cu_seqlens, device=concept.device)
            p = selected_probs[boundary_mask].unsqueeze(0).squeeze(-1)
            original_dtype = concept.dtype
            # Reuse Mamba2 kernel for EMA Deaggregator.
            dt = torch.log(1 / (1 - p)).to(self.dtype)

            x = (concept / dt[..., None]).to(self.dtype)

            A = -torch.ones(
                (self.nheads,), device=concept.device, dtype=torch.float32
            )
            b = p.to(self.dtype)
            c = torch.ones_like(b)

            out = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                repeat(dt, "b l -> b l h", h=self.nheads),
                A,
                rearrange(b, "b l -> b l 1 1"),
                rearrange(c, "b l -> b l 1 1"),
                chunk_size=self.block_size,
                seq_idx=seq_idx,
            )
            concept_merge = rearrange(out, "b l h p -> b l (h p)")

        concept_merge = concept_merge.squeeze(0)
        plug_back_idx = boundary_mask.squeeze(0).cumsum(dim=0) - 1
        concept_merge = torch.gather(
            concept_merge, dim=0, index=plug_back_idx.unsqueeze(-1).expand(-1, self.hidden_size)
        )
        concept_merge = concept_merge.unsqueeze(0)
        return concept_merge

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)
    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output
        return grad_x
def ste_func(x):
    return STE.apply(x)

class MoELayer(nn.Module):
    def __init__(self, layer_id, use_joint_decoding=False):
        super().__init__()
        self.layer_id = layer_id
        self.use_joint_decoding = use_joint_decoding

        self.attn = lambda x:x
        self.moe  = lambda x:x
        self.attn_joint = lambda x,y:x

    def forward(self, hidden_states, concept_merge=None):
        if self.use_joint_decoding:
            hidden_states = hidden_states + self.attn_joint(hidden_states, concept_merge) 
        else:
            hidden_states = hidden_states + self.attn(hidden_states)
        hidden_states += self.moe(hidden_states)
        return hidden_states

class ConceptMoE(nn.Module):
    def __init__(self, hidden_size=2048, vocab_size=32000):
        super().__init__()

        self.encoder = nn.ModuleList(MoELayer(layer_id=i) for i in range(2))
        self.concept_model = nn.ModuleList(MoELayer(layer_id=i) for i in range(2,25))
        self.decoder = nn.ModuleList(MoELayer(layer_id=i, use_joint_decoding=True) for i in range(25,27))

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.chunk_module = ChunkModule(hidden_size)
        self.dechunk_module = DechunkModule()


    def forward(self, input_ids):
        # encoder
        hidden_state = self.embedding(input_ids)        # shape [1024, 2048]
        for layer in self.encoder:
            hidden_state = layer(hidden_state)

        # chunk
        boundary_prob, boundary_mask, selected_probs, aux_loss = self.chunk_module(hidden_state)

        # main network
        concept = hidden_state[boundary_mask].unsqueeze(0)            # shape [1, 522, 2048]
        for layer in self.concept_model:
            concept = layer(concept)

        # dechunk
        concept_merge = self.dechunk_module(concept, selected_probs, boundary_mask)

        # decoder
        hidden_state = hidden_state + concept_merge * ste_func(selected_probs)
        for layer in self.decoder:
            hidden_state = layer(hidden_state, concept_merge)  # joint decoding

        logits = self.lm_head(hidden_state)
        return logits, aux_loss


if __name__ == "__main__":
    model = ConceptMoE().cuda()
    input_ids = torch.randint(0, 32000, (1, 1024)).cuda()
    logits, aux_loss = model(input_ids)
    print(logits.shape, aux_loss)
