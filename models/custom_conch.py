import torch
import torch.nn as nn
import torch.nn.functional as F
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

tokenizer = get_tokenizer()  # load tokenizer
# text_tokens = tokenize(texts=self.concepts, tokenizer=tokenizer)


class TextEncoder(nn.Module):
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

class CustomCONCH(nn.Module):
    def __init__(self, clip_model, n_ctx, text):
        super().__init__()
        self.prompt_learner = LPromptLearner(clip_model,n_ctx,text)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model.text)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.text.transformer.get_cast_dtype()
        self.text = text
        # self.prompts, self.tokenized_prompts = self.prompt_learner(text)

    def forward(self, image_features):
        # image_features = self.image_encoder(image.type(self.dtype))
        prompts, tokenized_prompts = self.prompt_learner(self.text)

        text_features, _ = self.text_encoder(prompts,tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

class LPromptLearner(nn.Module):
    def __init__(self, clip_model, n_ctx, texts):
        super().__init__()
        self.clip_model = clip_model.text
        dtype = clip_model.logit_scale.dtype
        # dtype = self.clip_model.ln_final.weight.dtype
        ctx_dim = 768
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)

        nn.init.normal_(ctx_vectors, std=0.02)
        self.prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial text context: "{self.prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        # self.register_buffer("ctx", ctx_vectors)
        self.ctx = nn.Parameter(ctx_vectors) # (n_ctx, ctx_dim)
        self.n_ctx = n_ctx
        self.texts = texts


        classnames = [name.replace("_", " ") for name in texts]
        prompts = [self.prompt_prefix + " " + name for name in classnames]
        self.tokenized_prompts = tokenize(texts=prompts, tokenizer=tokenizer)


        with torch.no_grad():
            embedding = self.clip_model.token_embedding(self.tokenized_prompts).type(dtype)


        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

    # def build_cls_mask(self, text, cast_dtype: torch.dtype):
    #     cls_mask = (text != 0).unsqueeze(1)
    #     cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=1.0)
    #     additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
    #     additive_mask.fill_(0)
    #     additive_mask.masked_fill_(~cls_mask, float("-inf"))
    #     additive_mask = torch.repeat_interleave(additive_mask, 12, 0)
    #     return additive_mask

    def forward(self, texts=None):
        ctx = self.ctx
        n_cls = len(self.texts)


        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        # embedding = embedding.unsqueeze(0)

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        tokenzied_prompts = self.tokenized_prompts

        return prompts, tokenzied_prompts