# Server setup (from a bare machine to train + eval)

Steps to bring VFlash up on a fresh Linux box you SSH into directly (no SLURM).
For the university SLURM cluster use the `scripts/*.sh` + sbatch commands in `README.md`
instead — the `scripts/local/*.sh` used here assume you activate the env yourself.

## 0. Check the GPU
```bash
nvidia-smi          # shows GPUs + the max CUDA version the driver supports (top-right)
```
If this prints nothing, the box has no usable NVIDIA driver and nothing below will run.
Note the CUDA version — it decides which PyTorch wheel to install in step 4.

## 1. Install Miniconda (if `conda` is missing)
```bash
cd ~
curl -L -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/etc/profile.d/conda.sh
conda init bash
source ~/.bashrc      # now the prompt shows (base)
```
Recent conda refuses the default channels until you accept their ToS:
```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

## 2. Create + activate the env
```bash
conda create -n vflash python=3.10 -y
conda activate vflash        # prompt switches to (vflash)
```
The env lives at `$HOME/miniconda3/envs/vflash` and persists across logins.
On reconnect just run `conda activate vflash` (conda auto-loads from `~/.bashrc`).

## 3. Install git (if `which git` is empty) and clone
```bash
conda install -c conda-forge git -y
git clone https://github.com/ItzikWaizman/VFlash.git
cd VFlash
```

## 4. Install PyTorch (match the CUDA from step 0) + deps
```bash
# driver CUDA >= 12.1 -> cu121 (most common); CUDA 11.8 -> use cu118
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
Verify torch sees the GPUs:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```
Expect `True` and the right device count.

## 5. (Optional) point the HF cache at a big disk
Setup downloads ~30-40 GB of video + ~15 GB of weights. If `$HOME` is small, set this
once per shell (or add it to `~/.bashrc`):
```bash
export HF_HOME=/path/to/big_disk/hf_cache
```

## 6. Run the pipeline
All local scripts auto-detect the GPU count; `pregen` runs one process per GPU.
```bash
bash scripts/local/setup.sh     experiments/baseline.json   # download models + data, build manifest
bash scripts/local/pregen.sh    experiments/baseline.json   # target greedy responses -> gen_cache
bash scripts/local/train.sh     experiments/baseline.json   # torchrun DDP over all GPUs
bash scripts/local/infer.sh     experiments/baseline.json   # speedup vs AR
bash scripts/local/benchmark.sh experiments/baseline.json   # VideoDetailCaption (M + speedup)
```
Outputs (rolling checkpoint, `metrics.jsonl`, live plots) land in `outputs/<run>/`.

## Quick reconnect cheatsheet
```bash
conda activate vflash
cd ~/VFlash
git pull
export HF_HOME=/path/to/big_disk/hf_cache   # only if you used a custom cache
```
