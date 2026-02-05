# Hyperbolic Latent Flow Matching

![Model architecture](HFM.drawio.png)

This repository combines:
- HAE (Hyperbolic Attribute Editing) for training and inference/embedding extraction.
- Riemannian Flow Matching (RFM) for learning a flow on those embeddings.



## 1) Environments

### hyper_env (HAE)
Create the conda environment from the root-level YAML:

```bash
conda env create -f hyper_env.yaml
conda activate hyper_env
```

### manifm (RFM)
Create the conda environment for RFM:

```bash
conda env create -f riemannian-fm/environment.yml
conda activate manifm
```

## 2) Dataset

Datasets are listed here:

```text
https://github.com/bcmi/Awesome-Few-Shot-Image-Generation
```

Download the datasets you need (flowers, animal faces, etc.), place them on disk, and then update the dataset locations in:

- `HAE/configs/paths_config.py`

That file controls where HAE looks for `flowers_train`, `flowers_test`, `animal_faces`, etc. Make sure the paths match your local dataset locations.

Pretrained HAE/PSP weights are provided here:

```text
https://drive.google.com/drive/folders/18zMfAEjd4JLsjQM78ky2GmHsolV7OJ_x?usp=share_link
```

Place those weights under:

- `HAE/pretrained_models/`

The HAE weights are given only for curvature=1.

## 3) Train HAE

The SLURM job script is:

- `HAE/run_train.sh`

It activates `hyper_env` and runs `python scripts/train.py` with the full set of flags. Key flags to adjust:

- `--dataset_type` (e.g. `animalfaces_encode_eva`, `flowers_encode_eva`)
- `--psp_checkpoint_path` (points into `HAE/pretrained_models/`)
- `--exp_dir` (output directory for checkpoints/logs)
- `--hyperbolic_curvature` (insert a negative value)
- `--hyperbolic_lambda` and `--reverse_lambda` (loss weights)

Example (from `HAE/run_train.sh`):

```bash
python scripts/train.py \
  --dataset_type=animalfaces_encode_eva \
  --psp_checkpoint_path=pretrained_models/psp_animalfaces.pt \
  --exp_dir=/scratch-shared/fvaleau/output_animals_c5.0 \
  --feature_size=512 \
  --workers=8 \
  --batch_size=8 \
  --test_batch_size=8 \
  --test_workers=8 \
  --val_interval=80000 \
  --save_interval=10000 \
  --encoder_type=GradualStyleEncoder \
  --start_from_latent_avg \
  --lpips_lambda=1 \
  --l2_lambda=1 \
  --image_interval=1000 \
  --hyperbolic_lambda=0.3 \
  --reverse_lambda=1 \
  --hyperbolic_curvature=-5 \
  --use_wandb
```

## 4) Inference (Extract Hyperbolic + Euclidean Embeddings)

The inference job script is:

- `HAE/inference.sh`

It runs `scripts/inference_umap.py`. That script **saves embeddings** to the `--exp_dir` you specify:

- Hyperbolic embeddings: `<exp_dir>/embed`
- Euclidean embeddings: `<exp_dir>/euc_embed`
- Labels: `<exp_dir>/label`
- Image paths: `<exp_dir>/image_path`

Example (from `HAE/inference.sh`):

```bash
python scripts/inference_umap.py \
  --exp_dir proj/animals_emb_5 \
  --checkpoint_path /scratch-shared/fvaleau/output_animals_c5.0/checkpoints/best_model.pt \
  --data_path /home/fvaleau/HAE/proj/animal_faces \
  --test_batch_size 1
```

## 5) Train RFM on the Embeddings

The SLURM job script is:

- `riemannian-fm/train.sh`

RFM reads paths from the Hydra config:

- `riemannian-fm/configs/train.yaml`

Set these fields to point at your HAE outputs:

- `images_datadir` -> `<HAE exp_dir>/embed`
- `images_labels` -> `<HAE exp_dir>/label`

Then run (in `riemannian-fm/train.sh`) the hyperbolic flow matching training:

```bash
python train.py experiment=images seed=0
```

Model/optimization settings live in:

- `riemannian-fm/configs/experiment/images.yaml`

If you need different data paths, **edit `riemannian-fm/configs/train.yaml`**. This is where to model the paths for RFM training.

## 6) Sample Latents from RFM

Sampling is done via `sample.py`. In `riemannian-fm/train.sh`, use a line like:

```bash
python sample.py \
  --checkpoint /path/to/your/checkpoint.ckpt \
  --name animals_c0.5 \
  --n_samples 49980
```

This writes the samples to:

- `/scratch-shared/fvaleau/FID/fm_samples_<name>`

That path is hard-coded in `riemannian-fm/sample.py`, to change it, please modify the script.

## 7) Decode Samples + Compute FID

HAE decoding + FID is driven by:

- `HAE/visualize.sh`

To decode, use a line like:

```bash
python Visualization/decode_hyperemb.py \
  --dataset animals \
  --emb_name fm_samples_animals_c0.5 \
  --model_path /home/fvaleau/HAE/pretrained_models/hae_animalfaces.pt
```

`decode_hyperemb.py` expects the samples at:

- `/scratch-shared/fvaleau/FID/<emb_name>`

It will save generated images to:

- `/scratch-shared/fvaleau/FID/<dataset>_generated/<emb_name>`

…and compute FID against:

- `/scratch-shared/fvaleau/FID/<dataset>_train`

Please modify these paths in `decode_hyperemb.py` if needed.

If your samples are Euclidean, switch the decode branch in:

- `HAE/Visualization/decode_hyperemb.py`

There are commented sections that indicate how to decode Poincaré samples vs. Euclidean W+ samples.

## Euclidean Experiment (Baseline)

1) **Extract Euclidean embeddings** with `HAE/inference.sh` (already produces `<exp_dir>/euc_embed`).

2) **Point RFM to euclidean embeddings** by editing:

- `riemannian-fm/configs/train.yaml`
  - `euclidean_datadir` -> `<HAE exp_dir>/euc_embed`

3) **Train Euclidean flow matching** by uncommenting in `riemannian-fm/train.sh`:

```bash
python train.py experiment=euclidean seed=0
```

4) **Sample Euclidean latents** via `sample.py` (same as above, but using the euclidean checkpoint).

5) **Decode + FID** using `HAE/visualize.sh` with a matching `--emb_name` and the HAE model path. The default decode branch in `decode_hyperemb.py` already handles Euclidean W+ samples.

## Citations

This code was adapted from the following repositories

``` 
https://github.com/lingxiao-li/HAE.git
https://github.com/facebookresearch/riemannian-fm
```


