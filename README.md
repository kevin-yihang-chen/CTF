# CTF: Bridging Radiology and Pathology Foundation Models via Concept-Based Multimodal Co-Adaptation

## 1. Repository layout

```
.
├── train.py                          # 10-fold CV training entry point.
├── make_splits.py                    # Generates the splits/ files from a label CSV.
├── concept_generate/
│   ├── generate.py                   # Stage 1: LLM-prompted candidate pool.
│   └── selection.py                  # Stage 2: MI ranking + diversity selection.
├── data_csv_template/                
├── models/
│   ├── par.py                        # CTF model (frozen FMs + GCSP + classifier).
│   ├── MP_MoE.py                     # GCSP prompt mechanism.
│   ├── moe.py                        # Sparse Mixture-of-Experts (gating).
│   └── abmil.py                      # Attention-based MIL pooling for pathology.
├── datasets/
│   ├── dataset_generic.py            
│   └── dataset_h5.py                 
└── utils/
    ├── core_utils.py                 
    ├── utils.py                      
    ├── loss_func.py                  
    └── file_utils.py                 
```

Two files are needed before running CTF:

* ``splits/<TASK>/splits_{0..9}.csv`` — 10-fold CV splits, where each CSV has the columns ``train``, ``val``, ``test`` listing the slide IDs in the corresponding fold.  Generate them with ``make_splits.py`` (Sec. 3.4) once you have the label CSV.
* ``concepts/<TASK>_path.xlsx`` and ``concepts/<TASK>_rad.xlsx`` — concept lists per task / modality (single ``concepts`` column).  We provide ``concepts/LGGGBM_Grading_{path,rad}.xlsx`` as an example for the TCGA-GBMLGG grading task; generate the others via ``concept_generate/`` (Sec. 4).  


> **Note on naming.** The model class is named ``PaR`` (in `models/par.py`) and the prompt
> module is named ``MPMoE`` (in `models/MP_MoE.py`).  These map directly to
> *CTF* and *GCSP* in the paper.

---

## 2. Installation

```bash
git clone https://github.com/HKU-MedAI/CTF.git
cd CTF

# Create a fresh environment (Python 3.10 recommended).
conda create -n ctf python=3.10 -y
conda activate ctf

# Install PyTorch matching your CUDA toolkit (https://pytorch.org/get-started/locally/).
pip install torch torchvision

# Remaining dependencies.
pip install -r requirements.txt
```

You also need to install the CONCH package from
[mahmoodlab/CONCH](https://github.com/mahmoodlab/CONCH) and download its
`pytorch_model.bin`.  CTF reads its location from the `CONCH_CKPT` environment
variable:

```bash
export CONCH_CKPT=/path/to/conch/pytorch_model.bin
```

---

## 3. Data preparation

### 3.1 Pathology features

We use the
[CLAM pipeline](https://github.com/mahmoodlab/CLAM), and extract
512-dim CONCH features per patch.  Save one HDF5 per slide at
`<data_root_dir>/h5_files/<slide_id>.h5`.


### 3.2 Label CSV

Place a CSV under `data_csv/` matching the columns shown in the templates
under `data_csv_template/`.  Adjust the corresponding entry in `TASKS` at
the top of `train.py` if you put it elsewhere.

### 3.3 10-fold CV splits

Generate the per-task splits directory once per cohort:

```bash
python make_splits.py \
    --task LGGGBM_Grading \
    --label_csv data_csv/TCGA_LGGGBM_info.csv \
    --out_dir splits \
    --k 10 --val_frac 0.1 --test_frac 0.1 --seed 1
```

This writes `splits/LGGGBM_Grading/splits_{0..9}.csv` (long format),
`splits_{0..9}_bool.csv` (wide boolean), `splits_{0..9}_descriptor.csv`
(per-fold class balance) and `splits_{0..9}_bool_label.csv` — the last is
what `concept_generate/selection.py` consumes for MI ranking.

---

## 4. Concept selection

`models/par.py` looks up two Excel files per training task under `$CTF_CONCEPTS_DIR`
(default `./concepts/`):

| File                       | Description                                            |
| -------------------------- | ------------------------------------------------------ |
| `<TASK>_path.xlsx`         | Pathology concepts for the task; one ``concepts`` column. |
| `<TASK>_rad.xlsx`          | Radiology concepts for the task; one ``concepts`` column. |

`<TASK>` is the value of `--task` passed to `train.py`.  This release ships
the single worked example `LGGGBM_Grading` (TCGA-GBMLGG grading) — both its
concept files (`concepts/LGGGBM_Grading_{path,rad}.xlsx`) and a matching
entry in the `TASKS` dict at the top of `train.py` and `make_splits.py`.
Add a new task by appending an entry to those `TASKS` dicts and dropping
the corresponding `<TASK>_path.xlsx` / `<TASK>_rad.xlsx` files into
`concepts/`.  Use the two-stage pipeline below to build the concept files
for the new task.

```bash
# 1. Query an LLM for a candidate pool of 250 concepts per progression stage.
export OPENAI_API_KEY=...
python concept_generate/generate.py \
    --cancer "Lower-Grade Glioma" --modal rad --n 250 \
    --out_dir concept_generate/

# 2. Rank by mutual information with labels and prune by diversity.
#    Output filename must match the convention <TASK>_<modal>.xlsx.
python concept_generate/selection.py \
    --cancer "Lower-Grade Glioma" --modal rad \
    --candidate_json "concept_generate/Lower-Grade Glioma_rad.json" \
    --label_csv data_csv/TCGA_LGGGBM_info.csv \
    --data_split_dir splits/LGGGBM_Grading \
    --data_dir /path/to/lggbm/radiology \
    --out_xlsx concepts/LGGGBM_Grading_rad.xlsx
```

---

## 5. Training

```bash
export CONCH_CKPT=/path/to/conch/pytorch_model.bin

python train.py \
    --task LGGGBM_Grading \
    --data_root_dir /path/to/lggbm/conch_features \
    --rad_dir /path/to/lggbm/radiology \
    --exp_code lggbm_grading \
    --test_name ctf_default \
    --max_epochs 100 --lr 1e-3 --reg 1e-6 \
    --early_stopping --patience 6 --weighted_sample \
    --n_concept 256 --n_ctx 12 \
    --cls_hidden_dim 128 --cls_layers 2 \
    --k 10
```

A run produces, under `results/<exp_code>/s<seed>_<test_name>/`:

* `splits_<i>.csv`            — dump of the per-fold split for reproducibility.
* `s_<i>_checkpoint.pt`       — best-validation checkpoint (only with `--save_model`).
* `summary.csv`               — per-fold validation/test metrics + mean/std.
* `split_<i>_results.pkl`     — per-patient predictions for fold `i`.


## 6. Citation

```bibtex
@inproceedings{
  chen2026bridging,
  title={Bridging Radiology and Pathology Foundation Models via Concept-Based Multimodal Co-Adaptation},
  author={Yihang Chen and Yanyan Huang and Fuying Wang and Maximus Yeung and Yuming Jiang and Shujun Wang and Lequan Yu},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```

## 7. Acknowledgements

Our codebase builds on:

* [CLAM](https://github.com/mahmoodlab/CLAM) for WSI tissue tiling and patch attention pooling.
* [CONCH](https://github.com/mahmoodlab/CONCH) and [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) for the frozen pathology / radiology foundation models.
* The Sparse Mixture-of-Experts implementation in `models/moe.py` is adapted from
  [davidmrau/mixture-of-experts](https://github.com/davidmrau/mixture-of-experts).
