# sci-Plex3

## Source And Version

- Source: scPerturb Zenodo `10.5281/zenodo.7041849`
- Local raw file: `data/single-cell/SrivatsanTrapnell2020_sciplex3.h5ad`
- Local note: md5 was previously audited against the harmonized scPerturb sci-Plex3 file.

## Raw Files

- `SrivatsanTrapnell2020_sciplex3.h5ad`

## Sample Definition

- One row corresponds to one single cell profile after drug perturbation.
- Current prepared export keeps 12 high-coverage perturbations after filtering and caps each retained perturbation at 1200 cells.
- Current prepared export contains 14400 perturbed cells and 12000 control cells.

## Intervention Definition

- Intervention id: `obs["perturbation"]`
- Intervention type: `obs["perturbation_type"]`
- Dose: `obs["dose_value"]`
- Time: `obs["time"]`
- Additional annotation fields used in the sample table: `chembl-ID`, `target`, `pathway`, `plate`, `well`, `replicate`

## Context Fields

- Categorical context: `cell_line`
- Numeric context: `dose_value`, `time`, `ncounts`, `ngenes`
- Matching fields for baseline construction: `cell_line`, `time`

## Baseline Availability

- Baseline mode: `matched_control_prototype`
- Controls are matched on `cell_line` plus `time`
- Pairing regime: `semi_paired`

## Preprocessing Steps

- Read sparse H5AD matrix and required `obs` fields with fail-closed metadata validation on retained rows.
- Exclude the explicit bad placeholder label `NA`.
- The excluded `NA` bucket contributes 36522 rows with missing metadata and is not silently retained.
- Select perturbations with at least 5000 cells, then keep the top 12 by coverage.
- Cap retained perturbation rows at 1200 per intervention and controls at 12000.
- Select the top 128 variable features on the retained control plus selected perturbation subset.
- Apply `log1p` to expression values.
- Export:
  - `data/prepared/sciplex3/prepared_bundle.npz`
  - `data/prepared/sciplex3/samples.csv`
  - `data/prepared/sciplex3/features.npy`
  - `data/prepared/sciplex3/feature_names.json`
  - `data/prepared/sciplex3/metadata.json`

## Feature Space

- `sciplex3_top128_log1p`
- `features.npy` stores `delta_response`
- `prepared_bundle.npz` stores `x_baseline`, `y_response`, and `delta_response`

## Supported Protocols

- `iid_sanity`
- `cold_intervention`
- `cold_context`
- `cold_dose_time`

## Current Limitations And Risks

- Current export is a coverage-filtered subset, not the full sci-Plex3 compound panel.
- Control matching is on `cell_line` and `time`; it is not dose-matched.
- The `NA` placeholder rows are unusable and must stay excluded in any downstream protocol generation.
