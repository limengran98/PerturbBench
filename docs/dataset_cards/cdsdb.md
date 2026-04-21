# CDS-DB

## Source And Version

- Local raw source:
  - `data/drug perturbation/CDSDB/raw_export/all_dataset.tar`
- Current local ingest parses the nested CDS-DB export directly instead of relying on a manually reconstructed table.

## Raw Files

- `all_dataset.tar`
- The top-level tar contains 181 nested `CDS_dataset_*/*.tar.gz` archives.
- Each nested archive contains:
  - `metadata.txt`
  - `profile.txt`
  - `signature.txt`

## Sample Definition

- One row in the prepared benchmark corresponds to one patient-level paired baseline/post-treatment record.
- Current prepared export parses all 181 nested CDS datasets and constructs 1974 paired samples.
- Pair construction is fail-closed: a patient is retained only when a dataset contains exactly one baseline and one post-treatment sample for that patient.

## Intervention Definition

- Intervention id: post-treatment therapeutic regimen
- Intervention type: `drug_category_compact` with fallback to `drug_category`
- Numeric dose: parsed conservatively from `Administration dose` only when a single unambiguous numeric value is present
- Numeric time: parsed conservatively from `Sampling time` into days when the unit is explicit

## Context Fields

- Prepared model context includes:
  - `source_dataset`
  - `platform`
  - `data_type`
  - `cancer_subtype`
  - `sampling_location_category`
- Leakage guards include:
  - `patient_id`
  - `source_dataset`
  - `platform`

## Baseline Availability

- Baseline mode: `exact_baseline`
- Pairing regime: `paired`
- `x_baseline` is the baseline patient profile and `y_response` is the paired post-treatment profile

## Preprocessing Steps

- Read `all_dataset.tar` directly and stream each nested `CDS_dataset_*` archive.
- Parse `metadata.txt`, `profile.txt`, and `signature.txt` for every nested dataset.
- Construct patient-level pairs only when the local dataset has an exact one-to-one baseline/post structure.
- Normalize feature identifiers by Gene ID and build a common feature space using genes present in all 181 parsed datasets.
- Apply `log1p` to RNA-seq profiles and keep Microarray values on their provided scale.
- Preserve every nested `signature.txt` under `data/prepared/cdsdb/signatures/`.
- Export:
  - `data/prepared/cdsdb/prepared_bundle.npz`
  - `data/prepared/cdsdb/samples.csv`
  - `data/prepared/cdsdb/features.npy`
  - `data/prepared/cdsdb/feature_names.json`
  - `data/prepared/cdsdb/metadata.json`
  - `data/prepared/cdsdb/field_coverage.json`
  - `data/prepared/cdsdb/source_manifest.json`

## Feature Space

- `cdsdb_geneid_common181`
- 73 common Gene ID features shared across all 181 parsed datasets after normalization
- `features.npy` stores `delta_response`
- `prepared_bundle.npz` stores `x_baseline`, `y_response`, and `delta_response`

## Supported Protocols

- `iid_sanity`
- `cold_intervention`
- `cold_context`
- `cross_source_transfer`

## Current Limitations And Risks

- The current common benchmark space is intentionally conservative: only 73 Gene IDs are shared across all 181 nested datasets.
- `response_group` coverage is incomplete: 870/1974 pairs are non-missing.
- Numeric dose coverage is incomplete and conservative: 766/1974 pairs parse to a single numeric dose.
- Numeric time coverage is partial: 1696/1974 pairs parse to an explicit day-scale value.
- `cross_source_transfer` is supported and should be preferred over naive pooled claims when evaluating study transfer.
