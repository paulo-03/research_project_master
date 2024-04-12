"""
Script allowing the user to use few types of noise, from basic to complex.
"""
import torch


def gaussian(image: torch.tensor, var=10) -> torch.tensor:
    # Get image size
    height, width = image.shape[-2], image.shape[-1]
    # Compute the standard deviation
    std = torch.sqrt(torch.tensor(var))
    # Compute the noise and add to the initial image
    noise = torch.randn(height, width) * std
    noisy_image = image + noise

    return torch.clamp(noisy_image, 0, 255)  # clamp it to be sure all pixel values are into [0, 255]


def pixel_intensity_adaptive(image: torch.tensor, var_dict: dict):
    # Get image size
    height, width = image.shape[-2], image.shape[-1]
    # Create a tensor to store the noise
    noise_tensor = torch.zeros_like(image)

    # Iterate over each pixel intensity value
    for intensity, var in var_dict.items():
        # Compute the standard deviation
        std = torch.sqrt(torch.tensor(var))
        # Generate noise with variance depending on the pixel intensity
        noise = torch.randn(height, width) * std
        # Mask for pixels with the current intensity value
        mask = image == intensity
        # Add noise to the corresponding pixels
        noise_tensor[mask] = noise[mask]

    # Add the noise tensor to the image tensor
    noisy_image = image + noise_tensor

    return torch.clamp(noisy_image, 0, 255)  # clamp it to be sure all pixel values are into [0, 255]
