"""Global-Context-Shared Prompt (GCSP) module.

* ``ctx_g`` (Global Prompt)  — domain-specific, concept-shared.
* ``ctx_c`` (Context Prompt) — produced by a Mixture-of-Experts gated by the
  *complementary* image features (radiology -> pathology and vice versa).
* ``ctx_s`` (Shared Prompt)  — produced by an MLP over the concatenation of
  both modalities' image features.

The class is named ``MPMoE`` for backward compatibility with the original
research code; in the paper it corresponds to the ``GCSP`` mechanism wrapping
the radiology/pathology text encoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from open_clip import get_tokenizer as get_tokenizer_rad
from conch.open_clip_custom import get_tokenizer as get_tokenizer_path
from conch.open_clip_custom import tokenize as tokenize_path

from .moe import MoE


tokenizer_path = get_tokenizer_path()
tokenizer_rad = get_tokenizer_rad('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim, dropout=0.1):
        super().__init__()
        dims = [input_dim] + list(hidden_sizes) + [output_dim]
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2 and dropout > 0:
                self.layers.append(nn.Dropout(dropout))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


class TextEncoderPath(nn.Module):
    """Wraps the CONCH text tower to consume pre-embedded prompts."""

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.transformer.get_cast_dtype()
        self.cls_emb = clip_model.cls_emb
        self.heads = clip_model.heads

        self.register_buffer('attn_mask', self._build_attention_mask(), persistent=False)

    @staticmethod
    def _build_attention_mask(seq_len=128):
        mask = torch.empty(seq_len, seq_len)
        mask.fill_(float('-inf'))
        mask.triu_(1)
        return mask

    def _build_cls_mask(self, text, cast_dtype):
        cls_mask = (text != 0).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=1.0)
        additive = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive.fill_(0)
        additive.masked_fill_(~cls_mask, float('-inf'))
        return torch.repeat_interleave(additive, self.heads, 0)

    @staticmethod
    def _repeat(t, n):
        return t.reshape(1, 1, -1).repeat(n, 1, 1)

    def forward(self, prompts, tokenized_prompts):
        x = prompts.to(self.dtype)
        x = x[:, :-1]
        seq_len = x.shape[1] + 1

        x = torch.cat([x, self._repeat(self.cls_emb, x.shape[0])], dim=1)
        cls_mask = self._build_cls_mask(tokenized_prompts, self.dtype).to(x.device)
        attn_mask = self.attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len].to(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)

        pooled = self.ln_final(x[:, -1])
        pooled = pooled @ self.text_projection
        pooled = F.normalize(pooled, dim=-1)
        return pooled


class TextEncoderRad(nn.Module):
    """Wraps the BiomedCLIP text tower to consume pre-embedded prompts."""

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.pooler
        self.text_projection = clip_model.proj
        self.dtype = clip_model.transformer.dtype

    def forward(self, prompts, mask):
        x = self.transformer(inputs_embeds=prompts, attention_mask=mask.to(prompts.device))
        x = self.ln_final(x, mask).type(self.dtype)
        return self.text_projection(x)


class _PromptLearnerBase(nn.Module):
    """Common scaffolding for the radiology and pathology prompt learners."""

    ctx_dim: int = 0  # set by subclasses

    def __init__(self, n_ctx, texts, n_experts=10):
        super().__init__()
        ctx_vectors = torch.empty(int(n_ctx / 2 + (n_ctx / 2 - 1) * n_experts), self.ctx_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.prompt_prefix = ' '.join(['X'] * n_ctx)
        self.ctx_g = nn.Parameter(ctx_vectors[:int(n_ctx / 2)])
        self.ctx_c = nn.Parameter(ctx_vectors[int(n_ctx / 2):])
        self.n_ctx = n_ctx
        self.texts = texts
        self.n_experts = n_experts

    def _format_classnames(self):
        return [name.replace('_', ' ') for name in self.texts]


class PromptLearnerPath(_PromptLearnerBase):
    """Prompt learner for the pathology text tower (CONCH)."""

    ctx_dim = 768

    def __init__(self, clip_model, n_ctx, texts, n_experts=10):
        super().__init__(n_ctx, texts, n_experts)
        text_tower = clip_model.text
        dtype = clip_model.logit_scale.dtype

        self.W_s = nn.Linear(256, self.ctx_dim)
        self.MoE = MoE(input_size=512, num_experts=n_experts, k=4)

        prompts = [self.prompt_prefix + ' ' + name for name in self._format_classnames()]
        self.tokenized_prompts = tokenize_path(texts=prompts, tokenizer=tokenizer_path)

        with torch.no_grad():
            embedding = text_tower.token_embedding(self.tokenized_prompts.to(text_tower.token_embedding.weight.device)).type(dtype)
        self.register_buffer('token_prefix', embedding[:, :1, :])
        self.register_buffer('token_suffix', embedding[:, 1 + n_ctx:, :])

    def forward(self, complementary_feat, shared_feat):
        ctx_s = self.W_s(shared_feat)
        ctx_c = self.ctx_c.reshape(self.n_experts, -1)
        gates, aux_loss = self.MoE(complementary_feat)
        ctx_c = torch.einsum('be, eh -> bh', gates, ctx_c).reshape(-1, self.ctx_dim)

        n_cls = len(self.texts)
        ctx_g = self.ctx_g.unsqueeze(0).expand(n_cls, -1, -1)
        ctx_s = ctx_s.unsqueeze(0).expand(n_cls, -1, -1)
        ctx_c = ctx_c.unsqueeze(0).expand(n_cls, -1, -1)

        prompts = torch.cat(
            [self.token_prefix, ctx_g, ctx_c, ctx_s, self.token_suffix],
            dim=1,
        )
        return prompts, self.tokenized_prompts, aux_loss


class PromptLearnerRad(_PromptLearnerBase):
    """Prompt learner for the radiology text tower (BiomedCLIP)."""

    ctx_dim = 768

    def __init__(self, clip_model, n_ctx, texts, n_experts=10):
        super().__init__(n_ctx, texts, n_experts)
        text_tower = clip_model.text

        self.W_s = nn.Linear(256, self.ctx_dim)
        self.MoE = MoE(input_size=512, num_experts=n_experts, k=4)

        prompts = [self.prompt_prefix + ' ' + name for name in self._format_classnames()]
        self.tokenized_prompts = torch.cat(
            [tokenizer_rad(p, context_length=128) for p in prompts]
        )

        with torch.no_grad():
            word_emb = text_tower.transformer.embeddings.word_embeddings
            embedding = word_emb(self.tokenized_prompts.to(word_emb.weight.device)).type(torch.float)

        self.mask = (self.tokenized_prompts != 0).long()
        self.register_buffer('token_prefix', embedding[:, :1, :])
        self.register_buffer('token_suffix', embedding[:, 1 + n_ctx:, :])

    def get_mask(self):
        return self.mask

    def forward(self, complementary_feat, shared_feat):
        ctx_s = self.W_s(shared_feat)
        ctx_c = self.ctx_c.reshape(self.n_experts, -1)
        gates, aux_loss = self.MoE(complementary_feat)
        ctx_c = torch.einsum('be, eh -> bh', gates, ctx_c).reshape(-1, self.ctx_dim)

        n_cls = len(self.texts)
        ctx_g = self.ctx_g.unsqueeze(0).expand(n_cls, -1, -1)
        ctx_s = ctx_s.unsqueeze(0).expand(n_cls, -1, -1)
        ctx_c = ctx_c.unsqueeze(0).expand(n_cls, -1, -1)

        prompts = torch.cat(
            [self.token_prefix, ctx_g, ctx_c, ctx_s, self.token_suffix],
            dim=1,
        )
        return prompts, self.tokenized_prompts, aux_loss


class MPMoE(nn.Module):
    """Cross-domain GCSP fuser (MPMoE in the original code)."""

    def __init__(self, path_model, rad_model, text_path, text_rad, args):
        super().__init__()

        self.prompt_learner_path = PromptLearnerPath(path_model, args.n_ctx, text_path)
        self.text_encoder_path = TextEncoderPath(path_model.text)

        self.prompt_learner_rad = PromptLearnerRad(rad_model, args.n_ctx, text_rad)
        self.text_encoder_rad = TextEncoderRad(rad_model.text)

        self.W_shared = MLP(512 * 2, [512, 512], 256)

        self.logit_scale_path = path_model.logit_scale
        self.logit_scale_rad = rad_model.logit_scale

        self.text_path = text_path
        self.text_rad = text_rad

    def forward(self, image_features_path, image_features_rad):
        # Patient-level shared signal feeding the Shared Prompt branch.
        z_shared = self.W_shared(torch.cat([image_features_path, image_features_rad], dim=1))

        # Pathology branch: concepts conditioned on radiology features.
        prompts_path, tok_path, aux_loss_path = self.prompt_learner_path(
            image_features_rad, z_shared,
        )
        text_features_path = self.text_encoder_path(prompts_path, tok_path)
        image_features_path = F.normalize(image_features_path, dim=-1)
        text_features_path = F.normalize(text_features_path, dim=-1)
        logits_path = self.logit_scale_path.exp() * image_features_path @ text_features_path.t()

        # Radiology branch: concepts conditioned on pathology features.
        prompts_rad, _, aux_loss_rad = self.prompt_learner_rad(
            image_features_path, z_shared,
        )
        text_features_rad = self.text_encoder_rad(
            prompts_rad, self.prompt_learner_rad.get_mask(),
        )
        image_features_rad = F.normalize(image_features_rad, dim=-1)
        text_features_rad = F.normalize(text_features_rad, dim=-1)
        logits_rad = self.logit_scale_rad.exp() * image_features_rad @ text_features_rad.t()

        return [logits_path, logits_rad], aux_loss_path + aux_loss_rad
