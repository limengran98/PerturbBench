# Adamson

## Source And Version

- Source: scPerturb Zenodo `10.5281/zenodo.7041849`
- Local raw file: `data/single-cell/AdamsonWeissman2016_GSM2406681_10X010.h5ad`
- Current local prepared config uses the 10X010 subset only.

## Raw Files

- `AdamsonWeissman2016_GSM2406681_10X010.h5ad`

## Sample Definition

- One row corresponds to one single-cell CRISPR perturbation profile.
- Current prepared export keeps 16 high-coverage perturbations after filtering and capping.
- Current prepared export contains 14009 perturbed cells and 2500 control cells.

## Intervention Definition

- Intervention id: `obs["perturbation"]`
- Intervention type: `obs["perturbation_type"]`
- Controls are identified by the explicit label `NA`
- Multi-perturbation status is derived from `obs["nperts"]`

## Context Fields

- Numeric context kept in the prepared sample table: `ncounts`, `ngenes`
- Baseline matching field: `cell_line`

## Baseline Availability

- Baseline mode: `matched_control_prototype`
- Controls are matched on `cell_line`
- Pairing regime: `semi_paired`

## Preprocessing Steps

- Read sparse H5AD matrix and required `obs` fields with fail-closed validation.
- Keep the explicit `NA` control bucket and do not silently relabel it.
- Select perturbations with at least 650 cells, then keep the top 16 by coverage.
- Cap retained perturbation rows at 1000 per intervention and controls at 2500.
- Select the top 128 variable features on the retained subset and apply `log1p`.
- Export:
  - `data/prepared/adamson/prepared_bundle.npz`
  - `data/prepared/adamson/samples.csv`
  - `data/prepared/adamson/features.npy`
  - `data/prepared/adamson/feature_names.json`
  - `data/prepared/adamson/metadata.json`

## Feature Space

- `adamson_top128_log1p`
- `features.npy` stores `delta_response`
- `prepared_bundle.npz` stores `x_baseline`, `y_response`, and `delta_response`

## Supported Protocols

- `iid_sanity`
- `cold_intervention`
- `cold_combination`

## Current Limitations And Risks

- Current prepared subset contains only one context group (`K562`), so `cold_context` is unsupported.
- The export is a coverage-filtered subset rather than the full perturbation catalog.
- `cold_combination` is derived from local `nperts` metadata and intervention identity; this should stay explicit in the manuscript protocol description.
