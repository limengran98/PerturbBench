# Papalexi Arrayed RNA

## Source And Version

- Source: scPerturb Zenodo `10.5281/zenodo.7041849`
- Local raw file: `data/single-cell/PapalexiSatija2021_eccite_arrayed_RNA.h5ad`

## Raw Files

- `PapalexiSatija2021_eccite_arrayed_RNA.h5ad`

## Sample Definition

- One row corresponds to one single-cell RNA profile in the arrayed ECCITE-seq subset.
- Current prepared export keeps 8 perturbations after filtering and capping.
- Current prepared export contains 3909 perturbed cells and 2009 control cells.

## Intervention Definition

- Intervention id: `obs["perturbation"]`
- Intervention type: `obs["perturbation_type"]`
- Guide-level metadata: `obs["guide_id"]`

## Context Fields

- Numeric context kept in the prepared sample table: `ncounts`, `ngenes`
- Baseline matching field: `cell_line`

## Baseline Availability

- Baseline mode: `matched_control_prototype`
- Controls are matched on `cell_line`
- Pairing regime: `semi_paired`

## Preprocessing Steps

- Read sparse H5AD matrix and required `obs` fields with fail-closed validation.
- Keep explicit control label `control`.
- Select perturbations with at least 250 cells, then cap each retained perturbation at 500 cells.
- Select the top 128 variable features on the retained subset and apply `log1p`.
- Export:
  - `data/prepared/papalexi_arrayed_rna/prepared_bundle.npz`
  - `data/prepared/papalexi_arrayed_rna/samples.csv`
  - `data/prepared/papalexi_arrayed_rna/features.npy`
  - `data/prepared/papalexi_arrayed_rna/feature_names.json`
  - `data/prepared/papalexi_arrayed_rna/metadata.json`

## Feature Space

- `papalexi_arrayed_rna_top128_log1p`
- `features.npy` stores `delta_response`
- `prepared_bundle.npz` stores `x_baseline`, `y_response`, and `delta_response`

## Supported Protocols

- `iid_sanity`
- `cold_intervention`

## Current Limitations And Risks

- Current prepared subset contains only one context group (`THP-1`), so `cold_context` is unsupported.
- No dose/time metadata are available in the local harmonized file.
- After filtering, the retained subset does not contain multi-perturbation rows, so `cold_combination` is unsupported.
