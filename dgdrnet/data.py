"""Strict paired RGB loading and union-of-boxes loss masks."""
from pathlib import Path
import math
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def image_index(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f'Image directory does not exist: {folder}')
    result = {}
    for path in sorted(folder.rglob('*')):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            key = path.relative_to(folder).with_suffix('').as_posix()
            if key in result:
                raise ValueError(f'Ambiguous image stem: {key}')
            result[key] = path
    if not result:
        raise ValueError(f'No images found: {folder}')
    return result


def load_rgb(path, size):
    with Image.open(path) as source:
        image = source.convert('RGB')
        if size:
            image = image.resize((size, size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def bounding_box_mask(label_path, height, width):
    mask = torch.zeros(1, height, width)
    for line in Path(label_path).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f'Expected YOLO box annotations: {label_path}')
        cls = int(fields[0])
        x, y, w, h = map(float, fields[1:])
        if cls < 0 or not all(math.isfinite(v) for v in (x,y,w,h)) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f'Invalid normalized box: {label_path}')
        left, right = max(0, math.floor((x-w/2)*width)), min(width, math.ceil((x+w/2)*width))
        top, bottom = max(0, math.floor((y-h/2)*height)), min(height, math.ceil((y+h/2)*height))
        mask[:, top:bottom, left:right] = 1
    return mask


class PairedDataset(Dataset):
    def __init__(self, blur_dir, sharp_dir, labels_dir=None, size=512):
        if size < 8 or size % 8:
            raise ValueError('Image size must be a positive multiple of eight.')
        self.blur = image_index(blur_dir)
        self.sharp = image_index(sharp_dir)
        if self.blur.keys() != self.sharp.keys():
            missing_sharp = sorted(self.blur.keys()-self.sharp.keys())[:5]
            missing_blur = sorted(self.sharp.keys()-self.blur.keys())[:5]
            raise ValueError(f'Unpaired files. Missing sharp: {missing_sharp}; missing blur: {missing_blur}')
        self.keys = sorted(self.sharp)
        self.labels_dir = Path(labels_dir) if labels_dir else None
        self.size = size
        if self.labels_dir:
            for key in self.keys:
                if not (self.labels_dir / (key+'.txt')).is_file():
                    raise FileNotFoundError(f'Missing label: {key}.txt')

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        blur = load_rgb(self.blur[key], self.size)
        sharp = load_rgb(self.sharp[key], self.size)
        mask = bounding_box_mask(self.labels_dir/(key+'.txt'), self.size, self.size) if self.labels_dir else torch.zeros(1,self.size,self.size)
        return {'blur':blur, 'sharp':sharp, 'mask':mask, 'name':key}
