"""
Script for computing all our  performance metrics.
"""

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

def get_metrics(predictions: torch.Tensor, targets: torch.Tensor, device: str):
    """Return performance metrics to quantify our models abilities to denoise CT Images"""
    # Compute metrics of all element in batch
    mses = _mse(predictions, targets)
    psnrs = _psnr(mses)
    # Compute the mean metric of the current batch
    mse = mses.mean().item()
    psnr = psnrs.mean().item()
    ssim = _ssim(predictions, targets, device)

    return mse, psnr, ssim


def _mse(predictions: torch.Tensor, targets: torch.Tensor):
    """Compute the Mean Square Error (MSE) of the prediction.
    Since images are from shape [B, H, W] we perform the mean on H and W to have one mean per B.
    """
    return torch.mean(torch.square(predictions - targets), dim=(1, 2))


def _psnr(mse, max_pixel=255):
    """Compute Peak Signal-to-Noise Ratio (PSNR) of the prediction"""
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))


def _ssim(predictions: torch.Tensor, targets: torch.Tensor, device: str):
    """Compute Structural Similarity Index Measure (SSIM) of the predictions"""
    ssim_calculator = StructuralSimilarityIndexMeasure(data_range=255).to('cuda')
    ssim = ssim_calculator(predictions, targets).item()
    return ssim
