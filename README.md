# PAN-GAM Code

This repository contains the Python implementation and supporting scripts for **Pan-GAM**: a species-stratified pan-genomic group association framework for identifying horizontal gene transfer (HGT)-driven antimicrobial resistance determinants while reducing cross-resistance artifacts in multi-drug resistant bacterial cohorts.

Pan-GAM integrates species-specific pangenome analysis, co-transfer unit (CTU) construction, drug-specific hypergeometric association testing, Benjamini-Hochberg false discovery rate correction, and machine-learning-based external validation.

## Repository contents

```text
PAN_GAM_Code/
├── pan_gam.py
├── epsilon_scan.py
├── pangam_synthetic_demo.py.py
├── generate_sample_data.py
├── generate_main_figures.py
├── generate_robustness_figure.py
└── generate_supplementary_example_figures.py
```

## File descriptions

### `pan_gam.py`

Main Pan-GAM analysis pipeline.

The current version of `pan_gam.py` integrates all analyses reported in the manuscript, including the species-stratified Pan-GAM workflow, discovery-only CTU construction, leakage-free external validation, epsilon-threshold sensitivity analysis, functional CTU auditing, curated-driver recovery, runtime profiling, bootstrap confidence intervals, and optional baseline-result comparison.

This script performs the core workflow, including:

- loading species-specific Roary gene presence/absence matrices;
- cleaning and aligning isolate metadata;
- separating discovery and external validation cohorts;
- constructing CTUs using Jaccard-distance-based hierarchical clustering;
- performing drug-specific hypergeometric association testing;
- applying Benjamini-Hochberg false discovery rate correction;
- selecting resistance-associated CTUs;
- training gradient boosting classifiers using selected CTUs;
- evaluating external validation performance;
- exporting CTU maps, association statistics, performance tables, trained models, and species-level summaries.

### `epsilon_scan.py`

Jaccard-threshold sensitivity analysis script.

This script runs the main Pan-GAM pipeline across multiple Jaccard-distance thresholds:

```text
0.01, 0.03, 0.05, 0.10, 0.15, 0.20
```

It summarizes the number of input genes, the number of resulting CTUs, the feature-reduction ratio, and external validation performance across threshold values.

### `pangam_synthetic_demo.py.py`

Synthetic demonstration script.

This script provides a class-based demonstration of the Pan-GAM framework using simulated data. It includes modular implementations of phenotypic partitioning, HGT-driven CTU construction, hypergeometric association testing, and significant-hit filtering.

This file is intended for demonstration and educational purposes. It is not required for reproducing the main Pan-GAM analysis pipeline.


### `generate_sample_data.py`

Synthetic sample-data generation script.

This script creates a small, reproducible test dataset that can be used to verify that the updated `pan_gam.py` pipeline executes correctly before real clinical data are available. It generates:

- species-specific synthetic Roary-style gene presence/absence matrices for `Kp` and `Sa`;
- a metadata file containing discovery and external cohorts;
- two binary resistance phenotypes (`DrugA` and `DrugB`);
- synthetic driver, regulatory, mobility, passenger, lineage-like, and background genes;
- optional gene-annotation, curated-driver, and baseline-result files.

The generated dataset is intended only for software testing and demonstration. It must not be interpreted as real clinical evidence or used to reproduce the manuscript's reported biological results.

### `generate_main_figures.py`

Main figure-generation script.

This script generates manuscript-style schematic figures related to:

- pan-genomic heterogeneity and resistance-profile grouping;
- CTU-based feature compression;
- HGT-driven resistance discovery;
- cross-resistance artifact reduction;
- machine-learning performance visualization.

Some plotted values are illustrative and should be replaced with cohort-derived result tables when reproducing final publication figures.

### `generate_robustness_figure.py`

Robustness figure-generation script.

This script generates visualizations for robustness analyses, including:

- sample-size downsampling;
- effect of missing phenotypic labels;
- comparison between Pan-GAM and baseline association models under reduced data availability.

### `generate_supplementary_example_figures.py`

Supplementary/example figure-generation script.

This script contains additional plotting examples, including performance comparison plots, geographic generalizability heatmaps, and resistance-mechanism visualization examples.

Some figures in this script use example or simulated values and should be interpreted as supplementary visualization templates rather than raw-result-generation scripts.

## Software requirements

The code is written in Python and was developed for Python 3.9 or later.

Required Python packages:

```text
numpy
pandas
scipy
statsmodels
scikit-learn
joblib
matplotlib
seaborn
```

## Installation

Create and activate a clean Python environment:

```bash
conda create -n pangam python=3.9
conda activate pangam
```

Install the required packages:

```bash
pip install numpy pandas scipy statsmodels scikit-learn joblib matplotlib seaborn
```

Alternatively, create a `requirements.txt` file containing:

```text
numpy
pandas
scipy
statsmodels
scikit-learn
joblib
matplotlib
seaborn
```

and install all dependencies with:

```bash
pip install -r requirements.txt
```

## Input data format

The main pipeline requires three input files:

```text
data/metadata.csv
data/roary_kp.rtab
data/roary_sa.rtab
```

### 1. Metadata file

The metadata file should be a comma-separated file containing isolate identifiers, species labels, cohort labels, and binary antimicrobial resistance phenotypes.

Required columns:

```text
isolate_id,species,cohort
```

The metadata file should also contain one binary phenotype column for each antibiotic included in the analysis.

Example metadata format:

```text
isolate_id,species,cohort,methicillin,carbapenem,ciprofloxacin,gentamicin,erythromycin,clindamycin,tetracycline
Iso001,Kp,discovery,0,1,1,0,1,0,1
Iso002,Kp,external,0,0,1,0,0,0,1
Iso003,Sa,discovery,1,0,0,1,1,1,0
Iso004,Sa,external,1,0,0,0,1,0,1
```

Column descriptions:

- `isolate_id`: unique isolate or sample identifier;
- `species`: species label, for example `Kp` for *Klebsiella pneumoniae* and `Sa` for *Staphylococcus aureus*;
- `cohort`: cohort label, either `discovery` or `external`;
- antibiotic columns: binary phenotype values, where `1` indicates resistant and `0` indicates susceptible.

### 2. Roary gene presence/absence matrices

The Roary `.rtab` files should be generated separately for each species:

```text
data/roary_kp.rtab
data/roary_sa.rtab
```

Each `.rtab` file should contain gene clusters as rows and isolate identifiers as columns. Values should indicate gene presence or absence.

The isolate identifiers in the Roary matrices must match the `isolate_id` values in the metadata file.

## Running the main Pan-GAM pipeline

Example command:

```bash
python pan_gam.py \
  --metadata data/metadata.csv \
  --kp-rtab data/roary_kp.rtab \
  --sa-rtab data/roary_sa.rtab \
  --drugs methicillin,carbapenem,ciprofloxacin,gentamicin,erythromycin,clindamycin,tetracycline \
  --outdir results \
  --epsilon 0.05
```

Argument descriptions:

```text
--metadata    Path to the isolate metadata file.
--kp-rtab     Path to the K. pneumoniae Roary gene presence/absence matrix.
--sa-rtab     Path to the S. aureus Roary gene presence/absence matrix.
--drugs       Comma-separated list of antibiotic phenotype columns.
--outdir      Output directory for Pan-GAM results.
--epsilon     Jaccard-distance threshold for CTU construction.
```

## Main output files

The pipeline writes output files to the directory specified by `--outdir`.

Example output files:

```text
results/Kp_ctu_map.tsv
results/Sa_ctu_map.tsv
results/Kp_pangam_association.tsv
results/Sa_pangam_association.tsv
results/Kp_external_performance.tsv
results/Sa_external_performance.tsv
results/all_species_pangam_association.tsv
results/all_species_external_performance.tsv
results/species_summary.tsv
results/Kp_models.joblib
results/Sa_models.joblib
```

Output descriptions:

| Output file | Description |
|---|---|
| `Kp_ctu_map.tsv` | CTU membership map for *K. pneumoniae* gene clusters |
| `Sa_ctu_map.tsv` | CTU membership map for *S. aureus* gene clusters |
| `Kp_pangam_association.tsv` | Drug-specific CTU association results for *K. pneumoniae* |
| `Sa_pangam_association.tsv` | Drug-specific CTU association results for *S. aureus* |
| `Kp_external_performance.tsv` | External validation performance for *K. pneumoniae* models |
| `Sa_external_performance.tsv` | External validation performance for *S. aureus* models |
| `all_species_pangam_association.tsv` | Combined species-level association results |
| `all_species_external_performance.tsv` | Combined external validation performance results |
| `species_summary.tsv` | Summary of gene counts, CTU counts, and feature-reduction ratios |
| `Kp_models.joblib` | Trained drug-specific models for *K. pneumoniae* |
| `Sa_models.joblib` | Trained drug-specific models for *S. aureus* |

## Running threshold-sensitivity analysis

To evaluate the effect of the Jaccard-distance threshold on CTU construction and model performance, run:

```bash
python epsilon_scan.py
```

This script runs `pan_gam.py` across multiple epsilon values and saves:

```text
epsilon_sensitivity_summary.tsv
```

The output table includes:

- epsilon value;
- total number of genes;
- total number of CTUs;
- feature-reduction ratio;
- accuracy;
- sensitivity;
- specificity;
- positive predictive value;
- F1-score;
- area under the ROC curve.

## Running the synthetic demonstration

To run the synthetic Pan-GAM demonstration:

```bash
python pangam_synthetic_demo.py.py
```

This script generates simulated gene presence/absence data and simulated resistance phenotypes, then runs the Pan-GAM workflow to demonstrate CTU construction and association testing.

## Generating the synthetic sample dataset

To create the synthetic test files, run:

```bash
python generate_sample_data.py
```

This command creates a `sample_data/` directory containing:

```text
sample_data/metadata.csv
sample_data/Kp_gene_presence_absence.Rtab
sample_data/Sa_gene_presence_absence.Rtab
sample_data/gene_annotations.csv
sample_data/curated_drivers.csv
sample_data/baseline_results.csv
```

The generated files can then be supplied to the updated `pan_gam.py` pipeline. For example:

```bash
python pan_gam.py \
  --metadata sample_data/metadata.csv \
  --kp-rtab sample_data/Kp_gene_presence_absence.Rtab \
  --sa-rtab sample_data/Sa_gene_presence_absence.Rtab \
  --drugs DrugA,DrugB \
  --outdir test_results \
  --epsilon 0.05 \
  --gene-annotations sample_data/gene_annotations.csv \
  --curated-drivers sample_data/curated_drivers.csv \
  --baseline-results sample_data/baseline_results.csv \
  --n-bootstrap 50 \
  --no-tune-gbdt
```

A successful synthetic run confirms that the software workflow executes and produces output files. It does not validate the manuscript's real-data results, which require the original clinical dataset.

## Generating figures

To generate main manuscript-style figures:

```bash
python generate_main_figures.py
```

To generate robustness figures:

```bash
python generate_robustness_figure.py
```

To generate supplementary/example figures:

```bash
python generate_supplementary_example_figures.py
```

Some figure scripts use simulated or illustrative values. These scripts are included to support visualization and should be adapted to real output tables when generating final publication figures.

## Reproducibility workflow

A typical Pan-GAM analysis can be reproduced using the following steps:

1. Generate species-specific Roary gene presence/absence matrices.
2. Prepare a metadata file containing isolate IDs, species labels, cohort labels, and binary antibiotic resistance phenotypes.
3. Run `pan_gam.py` using the metadata file and species-specific Roary matrices.
4. Inspect the CTU maps and association result tables.
5. Evaluate the external validation performance outputs.
6. Optionally run `epsilon_scan.py` to evaluate Jaccard-threshold sensitivity.
7. Optionally run the figure-generation scripts to visualize results.

## Data availability

The full clinical isolate metadata and antimicrobial resistance phenotype data are not included in this repository because such data may be subject to institutional, ethical, and patient-privacy restrictions.

Users can apply the Pan-GAM workflow to their own datasets by providing compatible species-specific Roary gene presence/absence matrices and binary phenotype metadata files following the input format described above.

## Code availability

This repository provides the Pan-GAM source code, including:

- the main analysis pipeline, including all analyses reported in the manuscript;
- Jaccard-threshold sensitivity analysis;
- synthetic demonstration code;
- the `generate_sample_data.py` testing-data generator;
- main and supplementary figure-generation scripts.

## Citation

If you use this code, please cite the associated manuscript:

```text
Pan-GAM: Disentangling Horizontal Gene Transfer and Complex Cross-Resistance in Multi-Drug Resistant Bacteria via Pan-Genomic Group Association.
```
