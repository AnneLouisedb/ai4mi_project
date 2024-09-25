import cv2
import pywt
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

# Apply Gaussian filter
def apply_gaussian_filter(nd: np.ndarray, sigma: float = 1) -> np.ndarray:
    return gaussian_filter(nd, sigma=sigma)

# Apply Median filter 
def apply_median_filter(nd: np.ndarray, size: int = 3) -> np.ndarray:
    return median_filter(nd, size=size)

# Apply Non-local Means Denoising 
def apply_non_local_means_denoising(nd: np.ndarray, h: int = 10) -> np.ndarray:
    nd_uint8 = (nd * 255).astype(np.uint8)  # Convert to uint8 for OpenCV
    return cv2.fastNlMeansDenoising(nd_uint8, None, h, 7, 21).astype(np.float32) / 255  # Scale back to [0, 1]

# Apply Bilateral Filtering
def apply_bilateral_filtering(nd: np.ndarray, d: int = 9, sigmaColor: int = 75, sigmaSpace: int = 75) -> np.ndarray:
    nd_uint8 = (nd * 255).astype(np.uint8)  # Convert to uint8 for OpenCV
    return cv2.bilateralFilter(nd_uint8, d, sigmaColor, sigmaSpace).astype(np.float32) / 255  # Scale back to [0, 1]

# Apply Wavelet Transform Denoising 
def apply_wavelet_transform_denoising(nd: np.ndarray, wavelet: str = 'db1') -> np.ndarray:
    coeffs = pywt.wavedec2(nd, wavelet, level=2)
    threshold = np.median(np.abs(coeffs[-1])) / 0.6745  # Universal threshold estimation
    coeffs = list(map(lambda x: pywt.threshold(x, threshold, mode='soft'), coeffs))
    return pywt.waverec2(coeffs, wavelet)