# Protocols

## Principles

- `iid_sanity` is a sanity-only split and must not be used as the headline protocol.
- Headline claims must come from explicit holdout protocols with saved split files under `splits/<dataset>/<protocol>/`.
- If a dataset lacks the fields required to define a protocol, that protocol is marked unsupported and must fail closed.

## Protocol Claims

### `iid_sanity`

- Supports: basic implementation sanity, loss surface sanity, artifact plumbing checks.
- Does not support: claims about generalization to unseen interventions, unseen contexts, unseen combinations, or unseen dose-time regimes.

### `cold_intervention`

- Supports: generalization to unseen perturbation identities under observed context families.
- Does not support: claims about unseen context transfer or unseen dose-time transfer.

### `cold_context`

- Supports: transfer to unseen cell lines or other explicit context groups available in the prepared dataset.
- Does not support: claims about unseen interventions unless intervention holdout is also enforced separately.

### `cold_combination`

- Supports: generalization to unseen perturbation combinations when the dataset contains explicit multi-perturbation samples.
- Does not support: claims about unseen component perturbations if those components are absent from train.
- Current repo behavior: for Norman, singleton perturbations remain in train and only multi-perturbation combinations are held out across partitions.

### `cold_dose_time`

- Supports: generalization to unseen dose-time regimes, optionally conditioned on context when the dataset has explicit context plus dose/time fields.
- Does not support: claims about unseen interventions unless intervention holdout is also enforced separately.

### `cross_source_transfer`

- Supports: transfer to unseen source studies or source-grouped acquisition regimes such as source dataset or platform.
- Does not support: claims about unseen interventions unless intervention holdout is also enforced separately.
- Intended use: multi-study resources where platform or study provenance must remain explicit rather than implicitly mixed.

## Current Coverage

| Dataset | `iid_sanity` | `cold_intervention` | `cold_context` | `cold_combination` | `cold_dose_time` | `cross_source_transfer` |
|---|---|---|---|---|---|---|
| Norman | supported | supported | unsupported | supported | unsupported | unsupported |
| Adamson | supported | supported | unsupported | supported | unsupported | unsupported |
| sci-Plex3 | supported | supported | supported | unsupported | supported | unsupported |
| Papalexi RNA | supported | supported | unsupported | unsupported | unsupported | unsupported |
| Papalexi Protein | supported | supported | unsupported | unsupported | unsupported | unsupported |
| L1000 public | supported | supported | supported | unsupported | supported | unsupported |
| L1000 local dev/internal | supported | supported | supported | unsupported | unsupported | unsupported |
| CDS-DB | supported | supported | supported | unsupported | not a default headline protocol in current export | supported |

## Dataset Notes

### Norman

- `cold_context`: unsupported because the prepared subset only contains one context group (`K562`).
- `cold_combination`: supported because the prepared subset includes three explicit multi-perturbation combinations.
- `cold_dose_time`: unsupported because dose/time are absent.

### sci-Plex3

- `cold_context`: supported with `cell_line` holdout.
- `cold_dose_time`: supported with `context_dose_time_id = cell_line || dose || time`.
- `cold_combination`: unsupported because no combination metadata is available in the prepared dataset.

### Adamson

- `cold_context`: unsupported because the prepared subset only contains one context group (`K562`).
- `cold_combination`: supported because the prepared subset contains explicit multi-perturbation rows (`nperts > 1`) and the split derives `combination_id` from intervention identity plus multiplicity.
- `cold_dose_time`: unsupported because dose/time are absent.

### Papalexi RNA

- `cold_context`: unsupported because the prepared subset only contains one context group (`THP-1`).
- `cold_combination`: unsupported because the prepared subset contains only singleton perturbations after filtering.
- `cold_dose_time`: unsupported because dose/time are absent.

### Papalexi Protein

- `cold_context`: unsupported because the prepared subset only contains one context group (`THP-1`).
- `cold_combination`: unsupported because the prepared subset contains only singleton perturbations after filtering.
- `cold_dose_time`: unsupported because dose/time are absent.

### L1000 public

- `cold_context`: supported with `cid` holdout on the public GEO reconstruction.
- `cold_combination`: unsupported because the public compound-only benchmark has no combination structure.
- `cold_dose_time`: supported because `dose_um` and `time_h` are reconstructed from `sig_info`.
- `cross_source_transfer`: still unsupported in the current helper because the public benchmark exposes 2 source accessions, while the generic protocol currently requires at least 3 source groups.

### L1000 local dev/internal

- `cold_context`: supported with `cid` holdout.
- `cold_combination`: unsupported because the local derived bundle has no combination structure.
- `cold_dose_time`: unsupported because the local derived bundle does not expose dose/time.

### CDS-DB

- `cold_context`: supported with patient-level holdout derived from `patient_id`.
- `cross_source_transfer`: supported with `source_dataset` holdout and explicit zero-overlap guard on source study identifiers.
- `cold_combination`: unsupported because the current paired patient-level export does not model perturbation combinations separately.
- `cold_dose_time`: numeric time and dose are partially recoverable, but coverage is incomplete and the current repo does not treat dose-time transfer as a default CDS-DB headline protocol.

## Leakage Guard Coverage

- All generated split summaries audit overlap for `intervention_id`.
- `combination_id` is audited when derivable; otherwise it is marked unsupported in the summary.
- `context_id` is audited when derivable from explicit context metadata such as `cell_line` or `cid`.
- `source_dataset` is audited when present and is the required zero-overlap field for `cross_source_transfer`.
- Batch or patient style fields are audited when present, currently including fields such as `plate`, `well`, and `replicate` for sci-Plex3.
