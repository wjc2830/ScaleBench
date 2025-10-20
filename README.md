## ScaleBench (Domain Generalization for Crowd Localization)

This repository contains training and evaluation code for domain generalization in crowd localization using PyTorch. It supports multiple algorithms and backbones (e.g., HRNet, ViT-UNet, ResNet18-UNet).

### 1) Environment setup (conda)

Requires Python 3.8+ and CUDA-enabled PyTorch.

```bash
# 1. Create and activate a conda environment
conda create -n scalebench python=3.10 -y
conda activate scalebench

# 2. Install PyTorch that matches your CUDA (example below is CUDA 12.1)
# Visit https://pytorch.org for the correct command for your system.
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# 3. Install project dependencies
pip install -r requirements.txt
```

Notes:
- If you prefer CPU-only, install the CPU wheels from the PyTorch site, but training will be slow.
- `tensorflow` and `tensorboard` are listed for logging/visualization compatibility; they are optional for pure training.

### 2) Data preparation

Set up a temporary data root (passed via `--tmp_root`) that contains domains and split files expected by the dataset configs under `datasets/setting/`.

Expected layout (example for 4 domains):

```text
/path/to/data_root/
  ├─ domain_1/
  │   ├─ images/                  # JPEG images, names without extension must match IDs in txt files
  │   ├─ masks/                   # PNG masks, same stem as images
  │   ├─ new_train.txt            # train IDs (one per line)
  │   ├─ Gaussian_val.txt         # val IDs
  │   ├─ Gaussian_val_gt_loc.txt  # val GT locations (index and boxes/points)
  │   ├─ Gaussian_whole.txt       # test IDs
  │   └─ Gaussian_whole_gt_loc.txt# test GT locations
  ├─ domain_2/
  │   ├─ images/
  │   ├─ masks/
  │   ├─ new_train.txt
  │   ├─ Gaussian_val.txt
  │   ├─ Gaussian_val_gt_loc.txt
  │   ├─ Gaussian_whole.txt
  │   └─ Gaussian_whole_gt_loc.txt
  ├─ domain_3/
  │   └─ ... (same structure as domain_1)
  └─ domain_4/
      └─ ... (same structure as domain_1)
```

Important:
- The dataset loader expects the folders `images/` and `masks/` for each domain. For some public datasets, mask folder names can be different (e.g., `mask_50_60` or `mask_30_60`), but in this project’s domain configs (`data1`–`data4`) the default is `masks/`.
- The split file names are defined in `datasets/setting/data{1,2,3,4}.py` and can be adapted if your filenames differ.

### 3) How dataset selection works

The `--dataset` argument encodes source and target domains using the format `ABCtoD`, where each letter is a domain index in {1,2,3,4,5}. For example:
- `123to4` trains on `data1`, `data2`, `data3` and evaluates on `data4`.
- `12to3` trains on `data1`, `data2` and evaluates on `data3`.

Internally, this maps to domain config files `datasets/setting/data{N}.py` and domain folders `domain_{N}` under `--tmp_root`.

### 4) Training

Common arguments (see `train.py` for all):
- `--tmp_root`: absolute path to your data root (required)
- `--dataset`: domain selection like `123to4` (required)
- `--DGAlg`: algorithm. Options include `Mixup`, `DANN`, `CORAL`, `EFDM`, `IRM`, `MMD`, `RSC`, `VRex`, `SAGM`, `SagNet`, `SAM`, `GAM`, `HGP`, `SD`, `DomainDrop`, `InfoBot`, `CausalIRL`, `SemanticHook` (see `algorithms/`)
- `--model_name`: backbone. Options: `HR_Net`, `ViTUNet`, `Res18UNet`
- `--gpuid`: CUDA device id(s), e.g. `0` or `0,1`
- `--batch_size`, `--lr`, `--num_iter`, `--val_freq`, `--val_start`
- `--ifdebug`: quick debug mode (True/False)

HRNet pretrained weights:
- Provide HRNet weights via `--Pre_HR_Net /abs/path/to/HRNet.pth`, or place `HRNet.pth` in the working directory.

Example: Train HRNet with Mixup on domains 1–3, test on domain 4

```bash
cd code
python train.py \
  --tmp_root /absolute/path/to/data_root \
  --dataset 123to4 \
  --DGAlg Mixup \
  --model_name HR_Net \
  --Pre_HR_Net /absolute/path/to/HRNet.pth \
  --gpuid 0 \
  --batch_size 16 \
  --lr 1e-5 \
  --num_iter 30000 \
  --val_freq 1000 \
  --val_start 10000
```

Quick sanity check run (short, uses validation early):

```bash
python train.py --tmp_root /absolute/path/to/data_root --dataset 12to3 --ifdebug True --gpuid 0
```

Outputs and logs:
- Experiments are saved under `code/exp/` with an auto-generated run name containing mode, dataset, model, and algorithm.

### 5) Evaluation (test-only)

Use a saved model checkpoint for evaluation by setting `--only_test` to the model path. All other dataset and data-root flags still apply.

```bash
python train.py \
  --tmp_root /absolute/path/to/data_root \
  --dataset 123to4 \
  --only_test /absolute/path/to/saved_model.pth \
  --gpuid 0
```

### 6) Tips and troubleshooting

- Ensure your `--tmp_root` is absolute and the domain folders (`domain_1`, `domain_2`, …) exist with the required txt files.
- If you see CUDA visibility issues, verify `--gpuid` and that the environment variable `CUDA_VISIBLE_DEVICES` resolves to the intended device(s).
- For large images, the validator uses tiling based on `TRAIN_SIZE` from the domain config.

### 7) Repository structure (key paths)

```text
code/
  train.py                 # Entry point
  trainer.py               # Training/validation loop
  algorithms/              # DG methods (forward/backward hooks per algorithm)
  datasets/                # Dataset loader and per-domain config
  model/                   # Backbones: HRNet, UNet, ViT-UNet
  misc/                    # Metrics, transforms, utilities
  requirements.txt
```

### 8) Citation

If you find this project useful, please cite the associated paper or repository (add your citation here).


