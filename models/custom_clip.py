import torch
import torch.nn as nn

from open_clip import get_tokenizer
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        # self.positional_embedding = clip_model.positional_embedding
        # self.positional_embedding = clip_model.transformer.embeddings.position_embeddings
        # self.ln_final = clip_model.ln_final
        self.ln_final = clip_model.pooler
        # self.text_projection = clip_model.text_projection
        self.text_projection = clip_model.proj
        # self.dtype = clip_model.ln_final.weight.dtype
        self.dtype = clip_model.transformer.dtype
        # self.cls_emb = clip_model.cls_emb
        # self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

    def forward(self, prompts, mask):

        x = prompts
        # x = prompts + self.positional_embedding.type(self.dtype)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # mask = (tokenized_prompts != 0).long()
        mask = mask.to('cuda')
        x = self.transformer(inputs_embeds=x,attention_mask=mask)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x,mask).type(self.dtype)
        x = self.text_projection(x)

        # x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class CustomCLIP(nn.Module):
    def __init__(self, clip_model, n_ctx, text):
        super().__init__()
        self.prompt_learner = LPromptLearner(clip_model,n_ctx,text)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model.text)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.text.transformer.dtype
        self.text = text
        # self.prompts, self.tokenized_prompts = self.prompt_learner(text)

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        prompts, mask = self.prompt_learner(self.text)

        text_features = self.text_encoder(prompts, mask)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

class LPromptLearner(nn.Module):
    def __init__(self, clip_model, n_ctx, texts):
        super().__init__()
        self.clip_model = clip_model.text
        dtype = self.clip_model.transformer.dtype
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
        tokenized_prompts = torch.cat([tokenizer(p,context_length=128) for p in prompts])

        with torch.no_grad():
            embedding = self.clip_model.transformer.embeddings.word_embeddings(tokenized_prompts).type(torch.float)

        self.mask = (tokenized_prompts != 0).long()
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

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

        return prompts, self.mask
