"""Checkpoint-compatible CogViT using the Transformers GLM-OCR building blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.glm_ocr.modeling_glm_ocr import (
    GlmOcrVisionBlock,
    GlmOcrVisionMlp,
    GlmOcrVisionModel,
    GlmOcrVisionPatchEmbed,
    GlmOcrVisionPatchMerger,
)

from .attention import RMSNorm
from transformers.modeling_outputs import BaseModelOutputWithPooling


def vision_swiglu_with_limit(x, limit):
    # ViT parity must not depend on the language model's fused-op switch.
    gate, up = x.float().chunk(2, dim=-1)
    return (F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)).to(x.dtype)


class Glm5NextVisionRMSNorm(RMSNorm):
    def forward(self, x):
        if x.device.type == "npu":
            import torch_npu

            return torch_npu.npu_rms_norm(x, self.weight, self.eps)[0]
        return super().forward(x)


class Glm5NextVisionBlock(GlmOcrVisionBlock):
    def __init__(self, config):
        super().__init__(config)
        self.norm1 = Glm5NextVisionRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm2 = Glm5NextVisionRMSNorm(config.hidden_size, config.rms_norm_eps)
        # Q/K normalization must use the same configured epsilon as serving.
        self.attn.q_norm = Glm5NextVisionRMSNorm(config.hidden_size // config.num_heads, config.rms_norm_eps)
        self.attn.k_norm = Glm5NextVisionRMSNorm(config.hidden_size // config.num_heads, config.rms_norm_eps)
        self.mlp = Glm5NextVisionMLP(config)

    def forward(self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
        attention = self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, position_embeddings=position_embeddings, **kwargs
        )
        if hidden_states.device.type == "npu":
            import torch_npu

            normalized, _, residual = torch_npu.npu_add_rms_norm(
                hidden_states, attention, self.norm2.weight, self.norm2.eps
            )
            return residual + self.mlp(normalized)
        # Match the fused residual-add/RMSNorm: normalize the FP32 sum,
        # while the residual branch itself is stored in the model dtype.
        residual = hidden_states.float() + attention.float()
        normalized = self.norm2(residual).to(hidden_states.dtype)
        return residual.to(hidden_states.dtype) + self.mlp(normalized)


class Glm5NextVisionMLP(GlmOcrVisionMlp):
    def __init__(self, config):
        super().__init__(config, bias=True)
        self.limit = config.swiglu_limit

    def forward(self, x):
        gate_up = torch.cat((self.gate_proj(x), self.up_proj(x)), dim=-1)
        return self.down_proj(vision_swiglu_with_limit(gate_up, self.limit).to(x.dtype))


class Glm5NextVisionMerger(GlmOcrVisionPatchMerger):
    def __init__(self, config):
        super().__init__(config.out_hidden_size, config.projection_intermediate_size, config.hidden_act)
        self.limit = config.swiglu_limit

    def forward(self, x):
        x = self.act1(self.post_projection_norm(self.proj(x)))
        gate_up = torch.cat((self.gate_proj(x), self.up_proj(x)), dim=-1)
        return self.down_proj(vision_swiglu_with_limit(gate_up, self.limit).to(x.dtype))


class Glm5NextPatchEmbed(GlmOcrVisionPatchEmbed):
    def forward(self, hidden_states):
        # Every input is exactly one non-overlapping Conv3d patch. GEMM
        # implements the same contraction without cuDNN's BF16 Conv3d fallback.
        patches = hidden_states.reshape(-1, self.proj.weight[0].numel()).to(self.proj.weight.dtype)
        # SGLang's Conv3dLayer uses this same GEMM and fused bias epilogue.
        return F.linear(patches, self.proj.weight.flatten(1), self.proj.bias)


class Glm5NextVisionModel(GlmOcrVisionModel):
    def __init__(self, config):
        config._attn_implementation = "sdpa"
        super().__init__(config)
        self.patch_embed = Glm5NextPatchEmbed(config)
        self.merger = Glm5NextVisionMerger(config)
        self.blocks = nn.ModuleList([Glm5NextVisionBlock(config) for _ in range(config.depth)])
        self.post_layernorm = Glm5NextVisionRMSNorm(config.hidden_size, config.rms_norm_eps)
        # Use one FP32 frequency table regardless of the construction device.
        # CPU/NPU pow differs by an ULP, which can amplify through this ViT.
        rotary = self.rotary_pos_emb
        rotary.inv_freq = (
            1.0 / (rotary.theta ** (torch.arange(0, rotary.dim, 2, dtype=torch.float32, device="cpu") / rotary.dim))
        ).to(rotary.inv_freq.device)

    def forward(self, hidden_states, grid_thw, **kwargs):
        # Keep GEMM shapes independent of rollout batching and training packing.
        if len(grid_thw) <= 1:
            return super().forward(hidden_states, grid_thw, **kwargs)
        counts = grid_thw.prod(dim=-1).tolist()
        outputs = [
            super(Glm5NextVisionModel, self).forward(pixels, grid_thw[i:i + 1], **kwargs)
            for i, pixels in enumerate(hidden_states.split(counts))
        ]
        return BaseModelOutputWithPooling(
            last_hidden_state=torch.cat([x.last_hidden_state for x in outputs]),
            pooler_output=torch.cat([x.pooler_output for x in outputs]),
        )

    def _apply(self, fn, recurse=True):
        # Frequencies are not checkpoint parameters. Preserve their original
        # FP32 values when Megatron casts the model weights to BF16.
        inv_freq = self.rotary_pos_emb.inv_freq
        super()._apply(fn, recurse=recurse)
        self.rotary_pos_emb.inv_freq = inv_freq.to(device=self.rotary_pos_emb.inv_freq.device)
        return self

# Optional paired ViT feature validation; no effect unless explicitly enabled.
if __import__('os').environ.get('GLM53_VIT_CAPTURE_DIR'):
    from glm53_vit_feature_capture import install as _install_vit_capture
    _install_vit_capture(Glm5NextVisionModel, 'train')
