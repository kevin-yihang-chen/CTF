import torch
import torch.nn as nn
import torch.nn.functional as F
from conch.open_clip_custom import get_tokenizer as get_tokenizer_path
from conch.open_clip_custom import tokenize
from open_clip import get_tokenizer as get_tokenizer_rad
from .moe import MoE

tokenizer_path = get_tokenizer_path()  # load tokenizer
# text_tokens = tokenize(texts=self.concepts, tokenizer=tokenizer)
tokenizer_rad = get_tokenizer_rad('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')


class TextEncoder_Path(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.transformer.get_cast_dtype()
        self.cls_emb = clip_model.cls_emb
        self.heads = clip_model.heads
        self.model = clip_model

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

    def build_attention_mask(self, seq_len=128):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(seq_len, seq_len)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_cls_mask(self, text, cast_dtype: torch.dtype):
        cls_mask = (text != 0).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=1.0)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def _repeat(self, t, N: int):
        return t.reshape(1, 1, -1).repeat(N, 1, 1)

    def forward(self, prompts, tokenized_prompts):

        x = prompts.to(self.dtype)
        x = x[:, :-1]

        # x = prompts + self.positional_embedding.type(self.dtype)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # mask = (tokenized_prompts != 0).long()
        mask = self.attn_mask
        seq_len = x.shape[1]

        seq_len += 1
        x = torch.cat([x, self._repeat(self.cls_emb, x.shape[0])], dim=1)
        cls_mask = self.build_cls_mask(tokenized_prompts, self.dtype)
        cls_mask = cls_mask.to('cuda')
        attn_mask = mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x+self.positional_embedding[:seq_len].to(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x,attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        pooled, tokens = x[:, -1], x[:, :-1]
        pooled = self.ln_final(pooled)

        pooled = pooled @ self.text_projection
        pooled = F.normalize(pooled, dim=-1)


        # x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return pooled, tokens

class TextEncoder_Rad(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.pooler
        self.text_projection = clip_model.proj
        self.dtype = clip_model.transformer.dtype

    def forward(self, prompts, mask):

        x = prompts
        mask = mask.to('cuda')
        x = self.transformer(inputs_embeds=x,attention_mask=mask)
        x = self.ln_final(x,mask).type(self.dtype)
        x = self.text_projection(x)

        return x

class MPMoE(nn.Module):
    def __init__(self, path_model, rad_model, n_ctx, text_path, text_rad):
        super().__init__()
        self.prompt_learner_path = LPromptLearner_Path(path_model,n_ctx,text_path)
        self.prompt_learner_rad = LPromptLearner_Rad(rad_model,n_ctx,text_rad)

        # self.image_encoder_path = path_model.visual
        self.text_encoder_path = TextEncoder_Path(path_model.text)
        self.logit_scale_path = path_model.logit_scale
        self.dtype_path = path_model.text.transformer.get_cast_dtype()
        self.text_path = text_path
        self.W_shared = MLP(512 * 2, [512, 512], 256)
        self.prompt_learner_rad = LPromptLearner_Rad(rad_model,n_ctx,text_rad)
        self.text_encoder_rad = TextEncoder_Rad(rad_model.text)
        self.logit_scale_rad = rad_model.logit_scale
        self.dtype_rad = rad_model.text.transformer.dtype
        self.text_rad = text_rad

        # self.prompts, self.tokenized_prompts = self.prompt_learner(text)

    def forward(self, image_features_path, image_features_rad):

        z_shared = self.W_shared(torch.cat([image_features_path, image_features_rad], dim=1))
        prompts_path, tokenized_prompts_path, aux_loss_path = self.prompt_learner_path(image_features_rad, z_shared, self.text_path)

        text_features_path, _ = self.text_encoder_path(prompts_path,tokenized_prompts_path)

        image_features_path = image_features_path / image_features_path.norm(dim=-1, keepdim=True)
        text_features_path = text_features_path / text_features_path.norm(dim=-1, keepdim=True)

        logit_scale_path = self.logit_scale_path.exp()
        logits_path = logit_scale_path * image_features_path @ text_features_path.t()

        prompts_rad, tokenized_prompts_rad, aux_loss_rad = self.prompt_learner_rad(image_features_path, z_shared, self.text_rad)

        text_features_rad = self.text_encoder_rad(prompts_rad, self.prompt_learner_rad.get_mask())
        image_features_rad = image_features_rad / image_features_rad.norm(dim=-1, keepdim=True)
        text_features_rad = text_features_rad / text_features_rad.norm(dim=-1, keepdim=True)

        logit_scale_rad = self.logit_scale_rad.exp()
        logits_rad = logit_scale_rad * image_features_rad @ text_features_rad.t()

        logits =[logits_path, logits_rad]
        aux_loss = aux_loss_path + aux_loss_rad

        return logits, aux_loss

class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_sizes: list[int],
                 output_dim: int,
                 dropout: float = 0.0):
        """
        input_dim   – dimensionality of each input sample
        hidden_sizes– list of hidden‐layer widths, e.g. [64, 128, 64]
        output_dim  – number of outputs (e.g. #classes or 1 for regression)
        dropout     – dropout rate (0.0 = no dropout)
        """
        super().__init__()
        # build a list of all layer dimensions
        dims = [input_dim] + hidden_sizes + [output_dim]
        # create a ModuleList of Linear + optional Dropout
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            # add dropout after each hidden layer
            if i < len(dims) - 2 and dropout > 0:
                self.layers.append(nn.Dropout(dropout))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten if your input is an image or higher-D tensor
        # x = x.view(x.size(0), -1)
        # apply all but last layer with ReLU
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        # last layer (no activation for regression or logits for classification)
        x = self.layers[-1](x)
        return x

class LPromptLearner_Rad(nn.Module):
    def __init__(self, clip_model, n_ctx, texts, n_experts=10, ablate=False):
        super().__init__()
        self.clip_model = clip_model.text
        dtype = self.clip_model.transformer.dtype
        # dtype = self.clip_model.ln_final.weight.dtype
        ctx_dim = 768
        ctx_vectors = torch.empty(round(n_ctx/2+(n_ctx/2-1)*n_experts), ctx_dim, dtype=dtype)

        nn.init.normal_(ctx_vectors, std=0.02)
        self.prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial text context: "{self.prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        # self.register_buffer("ctx", ctx_vectors)
        self.ctx_g = nn.Parameter(ctx_vectors[:int(n_ctx / 2)])  # (n_ctx, ctx_dim)
        self.ctx_c = nn.Parameter(ctx_vectors[int(n_ctx / 2):])
        self.n_ctx = n_ctx
        self.texts = texts
        self.n_experts = n_experts
        self.W_shared = nn.Linear(256, 768)
        self.MoE = MoE(input_size=512, num_experts=n_experts, k=4)

        classnames = [name.replace("_", " ") for name in texts]
        prompts = [self.prompt_prefix + " " + name for name in classnames]
        self.tokenized_prompts = torch.cat([tokenizer_rad(p,context_length=128) for p in prompts])
        self.tokenized_prompts = self.tokenized_prompts.to('cuda')

        with torch.no_grad():
            embedding = self.clip_model.transformer.embeddings.word_embeddings(self.tokenized_prompts).type(torch.float)

        self.mask = (self.tokenized_prompts != 0).long()
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

    def get_mask(self):
        return self.mask


    def forward(self, path, shared, texts=None, ablate_index=None):

        ctx_s = self.W_shared(shared)
        ctx_c = self.ctx_c.reshape(self.n_experts, -1)
        gates, aux_loss = self.MoE(path)
        ctx_c = torch.einsum("be, eh -> bh", gates, ctx_c)
        ctx_c = ctx_c.reshape(-1, 768)
        ctx_g = self.ctx_g

        n_cls = len(self.texts)

        ctx_g = ctx_g.unsqueeze(0).expand(n_cls, -1, -1)
        ctx_s = ctx_s.unsqueeze(0).expand(n_cls, -1, -1)
        ctx_c = ctx_c.unsqueeze(0).expand(n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        # embedding = embedding.unsqueeze(0)

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx_g,  # (dim0, n_ctx/2, dim)
                ctx_c,  # (dim0, n_ctx/2-1, dim)
                ctx_s,  # (dim0, 1, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        if ablate_index is not None:
            # Ablate the suffix tokens
            suffix = self.token_suffix_ablate
            shape = ctx_c[ablate_index].shape
            zero_vector = torch.zeros(shape, dtype=ctx_c.dtype, device=ctx_c.device)
            prompts[ablate_index] = torch.cat(
                [
                    prefix[ablate_index],
                    ctx_g[ablate_index],
                    zero_vector,
                    ctx_s[ablate_index],
                    suffix[ablate_index],
                ],
                dim=1,
            )

        tokenzied_prompts = self.tokenized_prompts


        return prompts, tokenzied_prompts, aux_loss

class LPromptLearner_Path(nn.Module):
    def __init__(self, clip_model, n_ctx, texts, n_experts=10):
        super().__init__()
        self.clip_model = clip_model.text
        dtype = clip_model.logit_scale.dtype
        # dtype = self.clip_model.ln_final.weight.dtype
        ctx_dim = 768
        ctx_vectors = torch.empty(int(n_ctx/2+(n_ctx/2-1)*n_experts), ctx_dim, dtype=dtype)

        nn.init.normal_(ctx_vectors, std=0.02)
        self.prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial text context: "{self.prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        # self.register_buffer("ctx", ctx_vectors)
        self.ctx_g = nn.Parameter(ctx_vectors[:int(n_ctx/2)]) # (n_ctx, ctx_dim)
        self.ctx_c = nn.Parameter(ctx_vectors[int(n_ctx/2):])
        self.n_ctx = n_ctx
        self.texts = texts
        self.n_experts = n_experts

        self.W_s = nn.Linear(256, 768)
        self.MoE = MoE(input_size=512, num_experts=n_experts, k=4)


        classnames = [name.replace("_", " ") for name in texts]
        prompts = [self.prompt_prefix + " " + name for name in classnames]
        self.tokenized_prompts = tokenize(texts=prompts, tokenizer=tokenizer_path)
        self.tokenized_prompts = self.tokenized_prompts.to('cuda')


        with torch.no_grad():
            embedding = self.clip_model.token_embedding(self.tokenized_prompts).type(dtype)


        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

    def forward(self, rad, shared, texts=None):
        ctx_s = self.W_s(shared)
        ctx_c = self.ctx_c.reshape(self.n_experts,-1)
        gates, aux_loss = self.MoE(rad)
        ctx_c = torch.einsum("be, eh -> bh", gates, ctx_c)
        ctx_c = ctx_c.reshape(-1, 768)
        ctx_g = self.ctx_g

        n_cls = len(self.texts)


        ctx_g = ctx_g.unsqueeze(0).expand(n_cls, -1, -1)
        ctx_s = ctx_s.unsqueeze(0).expand(n_cls, -1, -1)
        ctx_c = ctx_c.unsqueeze(0).expand(n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        # embedding = embedding.unsqueeze(0)

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx_g,  # (dim0, n_ctx/2, dim)
                ctx_c,  # (dim0, n_ctx/2-1, dim)
                ctx_s,  # (dim0, 1, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        tokenzied_prompts = self.tokenized_prompts

        return prompts, tokenzied_prompts, aux_loss