"""Training losses corresponding to manuscript equations (23)-(30)."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import laplacian_kernel


class RestorationLoss(nn.Module):
    def __init__(self, weighted_pixel=True, eps=1e-3, beta=0.95,
                 edge_weight=0.05, spectral_weight=0.10, kernel_weight=1e-4,
                 fft_norm="ortho"):
        super().__init__()
        if eps <= 0 or not 0 <= beta < 1:
            raise ValueError("Invalid Charbonnier epsilon or EMA coefficient.")
        if min(edge_weight, spectral_weight, kernel_weight) < 0:
            raise ValueError("Loss weights must be nonnegative.")
        self.weighted_pixel = weighted_pixel
        self.eps, self.beta, self.fft_norm = eps, beta, fft_norm
        self.edge_weight, self.spectral_weight, self.kernel_weight = edge_weight, spectral_weight, kernel_weight
        self.register_buffer("laplacian", laplacian_kernel()[None, None].repeat(3, 1, 1, 1))
        self.register_buffer("reference", torch.tensor(float('nan')))
        self.register_buffer("ema", torch.tensor(float('nan')))

    def charbonnier(self, diff):
        return torch.sqrt(diff.square() + self.eps ** 2)

    @torch.no_grad()
    def set_reference(self, value):
        value = torch.as_tensor(value, device=self.reference.device, dtype=torch.float32)
        if value.numel() != 1 or not torch.isfinite(value).item() or value.item() <= 0:
            raise ValueError("Reference pixel loss must be a finite positive scalar.")
        self.reference.copy_(value)
        self.ema.copy_(value)

    def spectral_loss(self, pred, target):
        # Full complex FFT difference: phase differences must not be discarded.
        a = torch.fft.fft2(pred.float(), norm=self.fft_norm)
        b = torch.fft.fft2(target.float(), norm=self.fft_norm)
        return (a - b).abs().mean()

    def forward(self, pred, target, mask, model, update_state=True):
        if pred.shape != target.shape or mask.shape != (pred.shape[0], 1, *pred.shape[-2:]):
            raise ValueError("Image and binary mask dimensions do not agree.")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("The bounding-box mask must be binary.")
        error = self.charbonnier(pred.float() - target.float())
        unweighted = error.mean()
        gamma = unweighted.new_zeros(())
        if self.weighted_pixel:
            if not torch.isfinite(self.reference).item():
                raise RuntimeError("Initialize the reference loss before weighted training.")
            if self.training and update_state:
                with torch.no_grad():
                    self.ema.mul_(self.beta).add_(unweighted.detach() * (1 - self.beta))
            gamma = (1 - self.ema / self.reference).clamp(0, 1).detach()
        area = mask.sum(dim=(-2, -1), keepdim=True)
        background = mask.shape[-2] * mask.shape[-1] - area
        alpha = torch.where(area > 0, gamma * torch.log1p(background / area.clamp_min(1)),
                            torch.zeros_like(area))
        pixel = ((1 + alpha * mask) * error).mean()
        edge_diff = F.conv2d(pred.float() - target.float(), self.laplacian, padding=2, groups=3)
        edge = self.charbonnier(edge_diff).mean()
        spectral = self.spectral_loss(pred, target)
        kernel = model.kernel_consistency_loss()
        total = pixel + self.edge_weight * edge + self.spectral_weight * spectral + self.kernel_weight * kernel
        return total, {"total": total.detach(), "pixel": pixel.detach(), "unweighted_pixel": unweighted.detach(),
                       "edge": edge.detach(), "spectral": spectral.detach(), "kernel": kernel.detach(),
                       "gamma": gamma, "alpha_mean": alpha.mean().detach()}
