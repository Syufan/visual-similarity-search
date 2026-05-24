import torch
import torch.nn as nn
import open_clip

_model = None
_preprocess = None
_device = None


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.linear = linear
        self.r = r
        self.scale = alpha / r
        d_out, d_in = linear.weight.shape
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, r))

    def forward(self, x):
        return self.linear(x) + (x @ self.A.T @ self.B.T) * self.scale


class CLIPLoRA(nn.Module):
    def __init__(self, clip_model, embed_dim: int = 256, r: int = 8, alpha: int = 16):
        super().__init__()
        self.clip = clip_model
        for p in self.clip.parameters():
            p.requires_grad_(False)

        # inject LoRA into every attention projection in the visual transformer
        for block in self.clip.visual.transformer.resblocks:
            for name in ("in_proj", "out_proj"):
                orig = getattr(block.attn, name, None)
                if isinstance(orig, nn.Linear):
                    setattr(block.attn, name, LoRALinear(orig, r=r, alpha=alpha))

        visual_dim = self.clip.visual.output_dim
        self.head = nn.Sequential(
            nn.Linear(visual_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim),
        )

    def forward(self, images):
        with torch.no_grad():
            feats = self.clip.encode_image(images).float()
        out = self.head(feats)
        return nn.functional.normalize(out, dim=-1)


def get_model(checkpoint_path: str, device: str = "cpu"):
    global _model, _preprocess, _device
    if _model is None:
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model = CLIPLoRA(clip_model).to(device)
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        _model = model
        _preprocess = preprocess
        _device = device
    return _model, _preprocess


@torch.no_grad()
def embed(images: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    model, _ = _model, _preprocess
    with torch.cuda.amp.autocast(enabled=(device != "cpu")):
        return model(images.to(device)).cpu()
