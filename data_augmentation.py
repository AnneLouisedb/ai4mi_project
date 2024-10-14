import torch
import random
import torchvision.transforms.functional as TF

class RandomCrop:
    """
    Randomly crop the input image and ground truth to the specified size.
    If the input image is smaller than the target size, it will be padded.
    """
    def __init__(self, output_size: tuple[int, int]):
        self.output_size = output_size

    def __call__(self, img: torch.Tensor, seg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # img and seg must be of shape [C, H, W] and [1, H, W] respectively
        assert img.shape[1:] == seg.shape[1:], "Image and ground truth size mismatch"
        
        _, h, w = img.shape
        new_h, new_w = self.output_size

        # Pad if needed
        if h < new_h or w < new_w:
            pad_h = max(new_h - h, 0)
            pad_w = max(new_w - w, 0)
            padding = [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2]  # left, top, right, bottom
            img = TF.pad(img, padding, fill=0)
            seg = TF.pad(seg, padding, fill=0)

        # Update dimensions after padding
        _, h, w = img.shape

        # Randomly crop the image
        top = random.randint(0, h - new_h) if h > new_h else 0
        left = random.randint(0, w - new_w) if w > new_w else 0

        img = TF.crop(img, top, left, new_h, new_w)
        seg = TF.crop(seg, top, left, new_h, new_w)

        return img, seg
