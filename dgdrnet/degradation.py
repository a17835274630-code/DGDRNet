"""Offline motion blur, additive noise, and one JPEG compression step."""
from dataclasses import dataclass, asdict
import hashlib
import cv2
import numpy as np


@dataclass(frozen=True)
class Parameters:
    blur_length: int
    blur_angle: float
    noise_sigma: int
    jpeg_quality: int


def per_image_rng(seed, name):
    digest = hashlib.sha256(f'{seed}:{name}'.encode('utf-8')).digest()
    return np.random.default_rng(int.from_bytes(digest[:8],'little'))


def sample_parameters(rng):
    return Parameters(int(rng.integers(7,66)), float(rng.uniform(0,180)),
                      int(rng.integers(1,55)), int(rng.integers(50,96)))


def motion_blur_kernel(length, angle):
    if length < 1 or int(length) != length or not np.isfinite(angle):
        raise ValueError('Invalid motion blur parameters.')
    length = int(length)
    support = length if length % 2 else length+1
    kernel = np.zeros((support,support),np.float32)
    center = (support-1)/2
    half = (length-1)/2
    theta = np.deg2rad(angle % 180)
    dx,dy = np.cos(theta)*half,np.sin(theta)*half
    # Subpixel rasterization keeps even requested lengths; no L -> L+1 remapping.
    scale = 256
    first = (round((center-dx)*scale),round((center-dy)*scale))
    last = (round((center+dx)*scale),round((center+dy)*scale))
    cv2.line(kernel,first,last,1.0,thickness=1,lineType=cv2.LINE_8,shift=8)
    kernel /= kernel.sum()
    return kernel


def degrade_bgr(image, params, rng):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected a uint8 BGR image.')
    if params.noise_sigma < 0 or not 1 <= params.jpeg_quality <= 100:
        raise ValueError('Invalid noise or JPEG parameters.')
    kernel = motion_blur_kernel(params.blur_length, params.blur_angle)
    blurred = cv2.filter2D(image,-1,kernel,borderType=cv2.BORDER_REFLECT101)
    noise = rng.normal(0,params.noise_sigma,blurred.shape).astype(np.float32)
    noisy = np.clip(blurred.astype(np.float32)+noise,0,255).astype(np.uint8)
    ok,encoded = cv2.imencode('.jpg',noisy,[cv2.IMWRITE_JPEG_QUALITY,params.jpeg_quality])
    if not ok:
        raise RuntimeError('JPEG encoding failed.')
    decoded = cv2.imdecode(encoded,cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError('JPEG decoding failed.')
    return decoded, encoded
