# DSRP: Neurosymbolic Regulatory Programs

Code converted 1:1 from the Colab notebook `nesy_gex_atacipynb.ipynb`. Each notebook code cell is now a standalone script in `scripts/`. Only `#` comments were removed; no logic, constants, strings or paths were changed.

DSRP learns a layer of binary "concepts" and a layer of differentiable AND / XOR / NAND symbolic modules on top of them, then reads predictions out of the module activations.

## Repository layout

```
scripts/
  01_dsrp_v19_perturbation.py      # notebook cell 2: DSRP v19, K562 perturbation-seq
  02_dsrp_v17_bidirectional.py     # notebook cell 3: DSRP v17, NeurIPS 2021 BMMC multiome GEX<->ATAC
  03_inspect_input_files.py        # notebook cell 4: inspects the STRING alias/links and CORUM files
requirements.txt                   # from notebook cell 1 (`!pip install ...`) plus libraries imported by the scripts
```

Notebook cell 1 (`!pip install -q scanpy scvi-tools scikit-learn scipy anndata leidenalg igraph`) is a shell command, not Python, so it became `requirements.txt`. The first seven entries are exactly that pip line. `torch`, `numpy`, `pandas`, `matplotlib`, `seaborn` and `requests` are added because the scripts import them (Colab preinstalls them).

## Scripts

### 01 — DSRP v19 (perturbation response)
- **Data:** `K562_essential_normalized_singlecell_01.h5ad`. Perturbations with at least 30 cells are kept. The top 5000 genes by variance are used, and expression is expressed as a delta from the mean of `non-targeting` control cells.
- **Annotation files:** CORUM human complexes, STRING v12.0 aliases, and STRING v12.0 physical links (score threshold 700).
- **Model:** a gated encoder maps expression delta to K=128 binary concepts. A symbolic layer with G=64 modules (12 AND, 6 XOR, 6 NAND clauses each, 6 active) maps those to pseudo-bulk reconstruction.
- **CORUM anchoring:** `K // 2` = up to 64 concepts are supervised by CORUM complexes. The rest are free.
- **Split:** by perturbation, 70% train / 10% val / 20% test (seed 42).
- **Training:** 100 epochs, 3 stages (stage ends at epoch 30 and 70), early stopping, seed list `[0]`.
- **Evaluation:** Pearson delta (all genes and top-20 DE), rank score, direction match, CORUM retrieval MAP@50, STRING score correlation, XOR epistasis validation, plus a summary figure.
- **Outputs:** written to `dsrp_v19_outputs/`.

### 02 — DSRP v17 (bidirectional GEX <-> ATAC)
- **Data:** `GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad`. The top 8,000 HVGs (mitochondrial genes excluded) and top 20,000 peaks are used. `ATAC_TARGET_MODE = "processed_x"`.
- **Model:** two encoders (GEX, ATAC) share a K=96 binary concept space, followed by G=64 AND/XOR/NAND modules (8 AND, 4 XOR, 4 NAND clauses, 5 active). Linear heads predict ATAC and GEX. The auxiliary cell-type classifier receives only the module outputs.
- **Split:** site1–3 for train/val (12% val), site4 held out as test, with a site-adaptive bias estimated on 2,000 site4 cells.
- **Training:** 250 epochs, seeds `[0, 1, 2]`, stage boundaries at epochs 20 and 200.
- **Regulons:** the regulon activity matrix is computed post hoc (CollecTRI/DoRothEA via OmniPath, falling back to sklearn GRN and curated regulons). The script pip-installs pySCENIC, decoupler and related packages at runtime.
- **Analyses:** assays for cross-modal agreement, cell-type specificity (JSD), regulon alignment, XOR mutual exclusivity (with zero-shot transfer to site4), GO enrichment, and concept stability across seeds.
- **Outputs:** written to `dsrp_v17_outputs/`.

### 03 — Input file inspection
Prints columns, dtypes, samples and line counts for the STRING alias file, STRING physical links file and CORUM file.

## Running

The scripts keep the notebook's original Google Drive paths (`/content/drive/MyDrive/...`). Edit these constants at the top of each script to run elsewhere:

- `01`: `PATH`, `CORUM_PATH`, `ALIAS_PATH`, `STRING_PATH`
- `02`: `PATH`
- `03`: `ALIAS_PATH`, `STRING_PATH`, `CORUM_PATH`

```bash
pip install -r requirements.txt
python scripts/03_inspect_input_files.py
python scripts/01_dsrp_v19_perturbation.py
python scripts/02_dsrp_v17_bidirectional.py
```

A CUDA GPU is used if available. Script 02 loads the full multiome dataset into memory and needs substantial RAM.

## Recorded notebook outputs (for reference)

From the saved outputs in the notebook, v19 test set (seed 0): Pearson delta 0.2514, Pearson delta top-20 DE 0.4008, rank score 0.2007, direction match 0.7416, CORUM MAP@50 0.3671, STRING rho 0.2725 (n=794).

The saved v17 output was produced by an earlier configuration (see below): GEX->ATAC site4 RMSE 0.32656 and cell-mean Pearson 0.26866, against a train-mean baseline of 0.33998.

## Known discrepancies in the source notebook (preserved, not fixed)

- The saved v17 output does not match the v17 code in the notebook. The output header prints `G=48` and stage boundaries of epochs 5 and 15 (and says "Stage3 removed"), while the code sets `G = 64`, `STAGE1_END = 20`, `STAGE2_END = 200`. It also lists only seed 0.
- The saved v17 output ends in `KeyError: 'arm_A_tfs'` in the visualization step. The code in the cell uses `arm_A_labels` and has no `arm_A_tfs`, so the run was made with an older version of the code.
- In v19, the printed "Train cells / Val / Test" counts use `idx_train.sum()` on an index array, which sums indices rather than counting cells (hence the implausible 33,674,751,272). This is cosmetic; the actual arrays are correct.
- The saved v19 output stops after the "Naming concepts" step.
