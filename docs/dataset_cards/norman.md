# Norman

## Source And Version

- Source: scPerturb Zenodo `10.5281/zenodo.7041849`
- Local raw file: `data/single-cell/NormanWeissman2019_filtered.h5ad`
- Local note: md5 was previously audited against the harmonized scPerturb Norman file.

## Raw Files

- `NormanWeissman2019_filtered.h5ad`

## Sample Definition

- One row corresponds to one single cell profile.
- Current prepared export keeps 16 perturbations after coverage filtering and caps each retained perturbation at 1200 cells.
- Current prepared export contains 16673 perturbed cells and 11855 control cells.

## Intervention Definition

- Intervention id: `obs["perturbation"]`
- Intervention type: `obs["perturbation_type"]`
- Guide-level metadata: `obs["guide_id"]`

## Context Fields

- Numeric context kept in the prepared sample table: `ncounts`, `ngenes`
- Matching field for baseline construction: `cell_line`

## Baseline Availability

- Baseline mode: `matched_control_prototype`
- Controls are matched on `cell_line`
- Pairing regime: `semi_paired`

## Preprocessing Steps

- Read sparse H5AD matrix and required `obs` fields with fail-closed metadata validation.
- Keep explicit control label `control`.
- Select perturbations with at least 700 cells, then keep the top 16 by coverage.
- Cap retained perturbation rows at 1200 per intervention and controls at 12000.
- Select the top 128 variable features on the retained control plus selected perturbation subset.
- Apply `log1p` to expression values.
- Export:
  - `data/prepared/norman/prepared_bundle.npz`
  - `data/prepared/norman/samples.csv`
  - `data/prepared/norman/features.npy`
  - `data/prepared/norman/feature_names.json`
  - `data/prepared/norman/metadata.json`

## Feature Space

- `norman_top128_log1p`
- `features.npy` stores `delta_response`
- `prepared_bundle.npz` stores `x_baseline`, `y_response`, and `delta_response`

## Supported Protocols

- `iid_sanity`
- `cold_intervention`
- `cold_combination`

## Current Limitations And Risks

- Single-cell-line dataset in current prepared form (`K562`), so context generalization is limited.
- Current export is a coverage-filtered subset, not the full perturbation catalog.
- No dose or time fields are available in the local Norman harmonized file.
