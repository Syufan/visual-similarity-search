import torch
import torch.nn as nn
import torch.nn.functional as F

_model = None
_preprocess = None
_device = None


class _LoRAWeights(nn.Module):
    def __init__(self, r: int, d_in: int, d_out: int):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(r, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.linear = linear
        self.scale = alpha / r
        d_out, d_in = linear.weight.shape
        self.lora = _LoRAWeights(r, d_in, d_out)

    def forward(self, x):
        return self.linear(x) + (x @ self.lora.lora_A.T @ self.lora.lora_B.T) * self.scale


class CLIPLoRA(nn.Module):
    def __init__(self, r: int = 8, alpha: int = 16, embed_dim: int = 256):
        super().__init__()
        from transformers import CLIPModel
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_encoder = clip.vision_model.vision_model  # CLIPVisionTransformer

        for p in self.vision_encoder.parameters():
            p.requires_grad_(False)

        for layer in self.vision_encoder.encoder.layers:
            for name in ("q_proj", "v_proj"):
                orig = getattr(layer.self_attn, name)
                setattr(layer.self_attn, name, LoRALinear(orig, r=r, alpha=alpha))

        hidden_size = self.vision_encoder.config.hidden_size  # 768 for ViT-B/32
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embed_dim),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_encoder(pixel_values=pixel_values)
        feats = outputs.pooler_output
        return F.normalize(self.projector(feats), dim=-1)


def _make_preprocess():
    from transformers import CLIPImageProcessor
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def preprocess(image):
        return processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

    return preprocess


def get_model(checkpoint_path: str, device: str = "cpu"):
    global _model, _preprocess, _device
    if _model is None:
        model = CLIPLoRA().to(device)
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _model = model
        _preprocess = _make_preprocess()
        _device = device
    return _model, _preprocess


@torch.no_grad()
def embed(images: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    with torch.cuda.amp.autocast(enabled=(device != "cpu")):
        return _model(images.to(device)).cpu()
