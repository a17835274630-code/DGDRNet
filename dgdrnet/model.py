"""DGDRNet architecture, multi-scale edges, and edge-guided frequency processing."""
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    width: int = 48
    expansion: int = 2
    use_dsmee: bool = True
    use_dfem: bool = True
    edge_guidance: bool = True
    branches: Tuple[str, ...] = ("small", "medium", "large")
    gate_hidden: int = 96
    fft_norm: str = "ortho"
    spectral_eps: float = 1e-8


def sobel_kernels():
    horizontal = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
    diagonal = torch.tensor([[0., 1., 2.], [-1., 0., 1.], [-2., -1., 0.]])
    other = torch.tensor([[2., 1., 0.], [1., 0., -1.], [0., -1., -2.]])
    return [k / k.abs().sum() for k in (horizontal, diagonal, horizontal.t(), other)]


def laplacian_kernel():
    """Return the normalized discrete 5x5 Laplacian initialization."""
    kernel = torch.tensor([[0., 0., -1., 0., 0.], [0., -1., -2., -1., 0.],
                           [-1., -2., 16., -2., -1.], [0., -1., -2., -1., 0.],
                           [0., 0., -1., 0., 0.]])
    kernel = kernel - kernel.mean()
    return kernel / kernel.abs().sum()


def gaussian_high_pass(sigma=1.5):
    axis = torch.arange(7, dtype=torch.float32) - 3
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    gaussian = torch.exp(-(xx.square() + yy.square()) / (2 * sigma ** 2))
    gaussian = gaussian / gaussian.sum()
    impulse = torch.zeros_like(gaussian)
    impulse[3, 3] = 1
    return impulse - gaussian


class PriorConv(nn.Module):
    def __init__(self, channels, kernel):
        super().__init__()
        weight = kernel[None, None].repeat(channels, 1, 1, 1)
        self.weight = nn.Parameter(weight.clone())
        self.register_buffer("initial_weight", weight.clone())
        self.padding = kernel.shape[-1] // 2
        self.channels = channels

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=self.padding, groups=self.channels)


class DSMEE(nn.Module):
    def __init__(self, channels, branches=("small", "medium", "large")):
        super().__init__()
        self.branches = tuple(branches)
        if not self.branches or len(set(self.branches)) != len(self.branches):
            raise ValueError("At least one unique DSMEE branch is required.")
        if set(self.branches) - {"small", "medium", "large"}:
            raise ValueError("Unknown DSMEE branch.")
        self.base = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                                  nn.Conv2d(channels, channels, 1))
        self.priors = nn.ModuleDict()
        if "small" in self.branches:
            self.priors["small"] = nn.ModuleList([PriorConv(channels, k) for k in sobel_kernels()])
        if "medium" in self.branches:
            self.priors["medium"] = PriorConv(channels, laplacian_kernel())
        if "large" in self.branches:
            self.priors["large"] = PriorConv(channels, gaussian_high_pass())
        self.scale_score = nn.Conv2d(channels * len(self.branches), len(self.branches), 1)
        self.mask_head = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                                       nn.Conv2d(channels, 1, 1))
        self.edge_project = nn.Conv2d(channels, channels, 1)
        nn.init.dirac_(self.edge_project.weight)
        nn.init.zeros_(self.edge_project.bias)
        nn.init.constant_(self.mask_head[-1].bias, -1.)
        self.edge_scale = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        base = self.base(x)
        responses = []
        for name in self.branches:
            branch = self.priors[name]
            if name == "small":
                edge = torch.stack([conv(base).abs() for conv in branch]).amax(dim=0)
            else:
                # Equations (9)-(10) use absolute medium/large responses.
                edge = branch(base).abs()
            responses.append(edge)
        weights = self.scale_score(torch.cat(responses, dim=1)).softmax(dim=1)
        fused = sum(weights[:, i:i+1] * edge for i, edge in enumerate(responses))
        mask = self.mask_head(fused).sigmoid()
        output = x + base * mask + self.edge_scale * self.edge_project(fused)
        return output, {"mask": mask, "weights": weights, "edges": responses}


class DFEM(nn.Module):
    def __init__(self, channels, hidden, guided=True, norm="ortho", eps=1e-8):
        super().__init__()
        if hidden < 1 or eps <= 0:
            raise ValueError("Invalid gate width or numerical epsilon.")
        self.gate = nn.Sequential(nn.Conv2d(channels + 1, hidden, 1), nn.GELU(),
                                  nn.Conv2d(hidden, channels, 1), nn.Sigmoid())
        self.project = nn.Conv2d(channels, channels, 1)
        self.guided, self.norm, self.eps = guided, norm, eps

    def normalize(self, x):
        return x / (x + x.mean(dim=(-2, -1), keepdim=True) + self.eps)

    def forward(self, x, mask):
        if mask is None or mask.shape != (x.shape[0], 1, *x.shape[-2:]):
            raise ValueError("DFEM requires an aligned single-channel DSMEE mask.")
        original_dtype = x.dtype
        spectrum = torch.fft.rfft2(x.float(), norm=self.norm)
        magnitude = spectrum.abs()
        a = self.normalize(magnitude)
        if self.guided:
            centered = mask.float() - mask.float().mean(dim=(-2, -1), keepdim=True)
            s = self.normalize(torch.fft.rfft2(centered, norm=self.norm).abs())
        else:
            s = torch.zeros_like(a[:, :1])
        g = self.gate(torch.cat([a, s], dim=1))
        # The unguided ablation bypasses multiplication by S, not enhancement itself.
        weight = s * g if self.guided else g
        enhanced_magnitude = magnitude + magnitude * weight
        enhanced_spectrum = torch.polar(enhanced_magnitude, torch.angle(spectrum))
        reconstructed = torch.fft.irfft2(enhanced_spectrum, s=x.shape[-2:], norm=self.norm)
        output = x + self.project(reconstructed.to(original_dtype))
        return output, {"A": a, "S": s, "G": g, "W_MS": weight,
                        "magnitude": magnitude, "enhanced_magnitude": enhanced_magnitude}


class SimplifiedChannelAttention(nn.Module):
    """Global average pooling and 1x1 channel projection; no spatial attention."""
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        return x * self.weight(x)


class FFN(nn.Module):
    """Pointwise residual feed-forward network with configurable expansion."""
    def __init__(self, channels, expansion):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, channels * expansion, 1), nn.GELU(),
                                 nn.Conv2d(channels * expansion, channels, 1))

    def forward(self, x):
        return x + self.net(x)


class SpatialBlock(nn.Module):
    def __init__(self, channels, config):
        super().__init__()
        self.dsmee = DSMEE(channels, config.branches) if config.use_dsmee else None
        self.sca = SimplifiedChannelAttention(channels)
        self.ffn = FFN(channels, config.expansion)

    def forward(self, x):
        if self.dsmee is None:
            mask = None
        else:
            x, aux = self.dsmee(x)
            mask = aux["mask"]
        return self.ffn(self.sca(x)), mask


class Decoder(nn.Module):
    def __init__(self, incoming, outgoing, config):
        super().__init__()
        self.up = nn.Sequential(nn.Conv2d(incoming, outgoing * 4, 1), nn.PixelShuffle(2))
        self.fuse = nn.Conv2d(outgoing * 2, outgoing, 1)
        self.block = SpatialBlock(outgoing, config)

    def forward(self, x, skip):
        return self.block(self.fuse(torch.cat([self.up(x), skip], dim=1)))[0]


class DGDRNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        if cfg.width < 1 or cfg.expansion < 1 or (cfg.use_dfem and not cfg.use_dsmee):
            raise ValueError("Invalid model configuration; DFEM requires DSMEE.")
        c = cfg.width
        self.stem = nn.Conv2d(3, c, 3, padding=1)
        self.encoders = nn.ModuleList([SpatialBlock(c * 2 ** i, cfg) for i in range(3)])
        self.downs = nn.ModuleList([nn.Conv2d(c * 2 ** i, c * 2 ** (i+1), 3, stride=2, padding=1)
                                    for i in range(3)])
        self.bottleneck_dsmee = DSMEE(c * 8, cfg.branches) if cfg.use_dsmee else None
        self.dfem = DFEM(c * 8, cfg.gate_hidden, cfg.edge_guidance, cfg.fft_norm,
                         cfg.spectral_eps) if cfg.use_dfem else None
        self.bottleneck_ffn = FFN(c * 8, cfg.expansion)
        self.decoders = nn.ModuleList([Decoder(c * 8, c * 4, cfg), Decoder(c * 4, c * 2, cfg),
                                       Decoder(c * 2, c, cfg)])
        self.tail = nn.Conv2d(c, 3, 3, padding=1)

    def forward(self, image, return_aux=False):
        if image.ndim != 4 or image.shape[1] != 3 or any(s % 8 for s in image.shape[-2:]):
            raise ValueError("Expected RGB BCHW input with height and width divisible by eight.")
        x = self.stem(image)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x, _ = encoder(x)
            skips.append(x)
            x = down(x)
        mask, frequency = None, {}
        if self.bottleneck_dsmee is not None:
            x, aux = self.bottleneck_dsmee(x)
            mask = aux["mask"]
        if self.dfem is not None:
            x, frequency = self.dfem(x, mask)
        x = self.bottleneck_ffn(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)
        restored = image + self.tail(x)
        return (restored, {"mask": mask, "frequency": frequency}) if return_aux else restored

    def kernel_consistency_loss(self):
        """Unweighted sum of squared Frobenius norms, not a mean or square root."""
        terms = [(m.weight - m.initial_weight).square().sum()
                 for m in self.modules() if isinstance(m, PriorConv)]
        return torch.stack(terms).sum() if terms else next(self.parameters()).new_zeros(())
