# DGDRNet

PyTorch implementation of a three-level residual U-Net with multi-scale edge enhancement (DSMEE), edge-spectrum-guided frequency enhancement (DFEM), and a bounding-box-weighted restoration objective.

## Overview

DGDRNet restores degraded RGB images through a shallow feature projection, three encoder stages, a spatial-frequency bottleneck, three decoder stages, and an output residual connection. Encoder and decoder stages combine DSMEE, simplified channel attention, and a residual feed-forward network. The bottleneck combines DSMEE, DFEM, and a feed-forward network. Decoder upsampling uses pointwise convolutions and pixel shuffle.

DSMEE combines four directional 3 x 3 Sobel responses, a 5 x 5 Laplacian response, and a 7 x 7 Gaussian high-pass response. Learnable pixel-wise branch weights fuse the responses, and a spatial edge mask modulates the residual features. DFEM uses the centered edge mask's magnitude spectrum to guide feature-magnitude enhancement while retaining the feature phase.

Training combines a bounding-box-weighted Charbonnier pixel loss, a Laplacian edge loss, a full complex spectral consistency loss, and a kernel consistency loss. The public interfaces support independent training, checkpoint selection, restoration, evaluation, and offline degradation generation.

This source package does not include datasets, pretrained restoration weights, or a YOLO detector. Download the datasets separately and train a model before running inference. Restoration and detection scores depend on the training run, data split, and evaluation configuration.

## Contents

```text
dgdrnet/
  model.py              Network, DSMEE, DFEM, kernel initialization
  losses.py             Pixel, edge, spectral, and kernel objectives
  data.py               Fixed image pairs and binary union-of-boxes masks
  degradation.py        Motion blur, Gaussian noise, JPEG compression
  metrics.py            RGB PSNR and SSIM
  runtime.py            Checkpoint handling and random seeds
configs/
  wtb_cdd.json          Final restoration objective with defect annotations
  gopro.json            Same objective, with explicit zero annotation masks
train.py                Training, resume, validation-only model selection
evaluate.py             Paired-image evaluation and optional restored outputs
infer.py                Restoration without sharp references or labels
generate_degradation.py Offline degradation with per-image parameter records
generate_ablation_configs.py
profile_model.py        Parameters and convolution-only MACs
tests/test_core.py      Formula and data-handling tests
requirements.txt
```

## Installation

Use Python 3.10 or newer. Create an isolated environment, install the appropriate CPU/CUDA build of PyTorch for your machine, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The scripts use standard PyTorch, NumPy, Pillow, and OpenCV. Data and output locations are supplied through command-line arguments. GPU availability does not guarantee bitwise identity across hardware, drivers, or package versions.

## WTB-CDD layout

Place the dataset in `data/WTB-CDD`, or pass another path with `--data-root`:

```text
data/WTB-CDD/
  Train/
    Blur/images/
    Sharp/images/
    Sharp/labels/
  VAL/
    Blur/images/
    Sharp/images/
  Test/
    Blur/images/
    Sharp/images/
```

Pair images by matching relative filename stems within a subset. Duplicate stems within an image directory and unmatched files are errors. Additional `Blur/labels` and validation/test annotation folders may exist for downstream detection but are not required by the restoration evaluator.

YOLO label rows have the format `class_id x_center y_center width height`, with normalized coordinates. The training mask is the union of all boxes, not a segmentation target. It is used only in the objective, never as an inference input. Class IDs for the supplied five-class WTB-CDD are 0: OIL LEAKAGE, 1: dirt, 2: Paint, 3: LE-Erosion, 4: PU-tape.

## Train

Run commands from this repository's root:

```bash
python train.py --config configs/wtb_cdd.json --data-root data/WTB-CDD --output runs/wtb_cdd
```

Defaults: 512 x 512 complete-image bicubic resizing, RGB [0,1], Adam, batch 2, 500 epochs, learning rate 0.0002, weight decay 0.0001, cosine minimum 0.000001, seed 3407. No online degradation is performed by the training loader. Use the fixed paired dataset described in the manuscript.

Before the first optimizer update, the initialized model is evaluated on the first 20 fixed-order training batches to calibrate the weighted-pixel reference loss. A smaller dataset uses all available reference batches. The EMA and reference are checkpointed. Validation images select `best_psnr.pt`; test images are not read during training. Logs include the actual loss components, gamma, alpha, and validation metrics.

```bash
python train.py --config configs/wtb_cdd.json --data-root data/WTB-CDD --output runs/wtb_cdd --resume runs/wtb_cdd/latest.pt
```

Resume requires the same full configuration, including total epochs, so the cosine schedule remains unchanged. `--stop-after-epoch` permits an early stop while retaining that schedule. Use a new output directory for each independent experiment, and fix the data split before training.

## Evaluate or restore

```bash
python evaluate.py --checkpoint runs/wtb_cdd/best_psnr.pt --blur-dir data/WTB-CDD/Test/Blur/images --sharp-dir data/WTB-CDD/Test/Sharp/images --output runs/wtb_cdd/test --save-images
python infer.py --checkpoint runs/wtb_cdd/best_psnr.pt --input data/WTB-CDD/Test/Blur/images --output runs/wtb_cdd/restored
```

Both commands use 512 x 512 by default. Restored files are saved as lossless PNG images. Evaluation produces per-image CSV metrics and their arithmetic means. SSIM is explicitly defined in `metrics.py`; different window/border conventions can yield different values. Image inputs are not overwritten.

Only load trusted checkpoints: PyTorch checkpoint loading can deserialize executable pickle content. This implementation validates its checkpoint format and loads model weights strictly.

## GoPro and HIDE

Obtain these public benchmarks from the original authors, not from this release:

- GoPro: https://seungjunnah.github.io/Datasets/gopro.html
- HIDE: https://github.com/joanshen0508/HA_deblur

For the protocol described in the paper, reserve 500 pairs from GoPro's official training split for validation and use the remaining 1,603 pairs for training. Keep the official 1,111 test pairs unchanged. Save and reuse the selected validation filenames across methods. Arrange the subsets in the same layout, or specify image directories explicitly:

```bash
python train.py --config configs/gopro.json --data-root data/GoPro --output runs/gopro
python evaluate.py --checkpoint runs/gopro/best_psnr.pt --blur-dir data/GoPro/Test/Blur/images --sharp-dir data/GoPro/Test/Sharp/images --output runs/gopro/test
python evaluate.py --checkpoint runs/gopro/best_psnr.pt --blur-dir data/HIDE/Test/Blur/images --sharp-dir data/HIDE/Test/Sharp/images --output runs/gopro/hide
```

`configs/gopro.json` explicitly disables defect annotations, making B zero and the weighted pixel loss equal to the unweighted pixel loss. HIDE is evaluated with the selected GoPro checkpoint without training or fine-tuning. Do not apply the synthetic degradation generator to the benchmark blur files.

## Generate synthetic degradation

The order is linear motion blur -> additive Gaussian noise -> one JPEG compression. Apply this at the source resolution before model-input resizing. Random defaults follow the manuscript:

- Blur length: integer 7-65 pixels.
- Angle: uniform [0,180) degrees.
- Noise sigma: integer 1-54 on the 0-255 scale.
- JPEG quality: integer 50-95.

```bash
python generate_degradation.py --sharp-dir data/source/Sharp/images --labels-dir data/source/Sharp/labels --output-dir data/generated/Blur
python generate_degradation.py --sharp-dir data/source/Sharp/images --output-dir data/fixed/Blur --fixed 30 35 10 90 --format jpg
```

The first command writes decoded JPEG pixels into lossless PNG files; this retains the JPEG distortion without a second lossy encoding. JPG mode writes the first JPEG bytes directly. Fixed parameters are given as L, angle, sigma, Q. Each run records relative filenames, seed, sampled parameters, and kernel canvas size in `parameters.csv`.

An even requested L is not incremented: an odd canvas centers a line with the requested nominal span. Linear blur uses OpenCV subpixel line rasterization and reflect-101 image borders. Additive noise is sampled independently for each pixel and channel, clipped to [0,255], and converted to uint8 before JPEG compression. Each image receives a stable random seed derived from the global seed and relative filename. Keep the generated pairs and parameter records fixed for validation and testing. The output directory must be empty and separate from the input.

## Implementation settings

- Default base width: 48; FFN expansion: 2; DFEM gate hidden width: 96. These settings are exposed in the JSON configurations.
- Simplified channel attention uses global average pooling and a 1 x 1 channel projection. The FFN uses pointwise expansion, GELU, pointwise projection, and a residual connection.
- Gaussian high-pass initialization uses a normalized 7 x 7 Gaussian with sigma 1.5 subtracted from a unit impulse. The learnable DSMEE edge coefficient starts at 0.3.
- DFEM and the spectral loss use orthonormal FFT normalization. DFEM normalization epsilon is 1e-8; the Charbonnier smoothing epsilon is 1e-3.
- Bounding boxes are rasterized at the model input resolution using clipped floor/ceil bounds and combined by union. The resulting mask remains binary.
- SSIM uses an 11 x 11 Gaussian window with sigma 1.5, padding five, C1=0.01^2, and C2=0.03^2. RGB metrics are computed for each image and then averaged.

## Ablations

```bash
python generate_ablation_configs.py
python train.py --config configs/ablations/structure_full.json --data-root data/WTB-CDD --output runs/structure_full
python train.py --config configs/ablations/frequency_unguided.json --data-root data/WTB-CDD --output runs/frequency_unguided
```

The generator creates 15 configs for overall modules, seven branch combinations, the unguided frequency variant, and four loss combinations. Structural ablations use unweighted pixel + edge + spectral loss and no kernel regularizer. Loss ablations fix the complete architecture. Train each config independently. The unguided gate has a zero edge input but bypasses multiplication by S so that enhancement remains active.

## Parameter profiling

```bash
python profile_model.py --size 512
```

This reports trainable parameters and convolution-only multiply-accumulate operations (MACs), not total FLOPs. FFTs, elementwise operations, nonlinearities, pooling, and bias additions are excluded. Use the same counting convention when comparing models.

## Data and permissions

Third-party datasets are obtained from their original providers and remain subject to their respective licenses and attribution requirements. The source package contains no dataset images or annotation files. No software license file is included in this package.
