"""Fetch *publicly released* independent expert-corrected MRI+mask pairs.

Dataset selection is deterministic, based only on public file metadata and
acquisition modality, BEFORE any labels or predictions are inspected.
No MRI file, identifiable metadata or voxel array is committed or uploaded.

- NFBS sample via pinned publicly distributed pybrook source repo.
- External SynthStrip dataset public mirror, pinned by its Hub revision.
Any dataset source failure is a visible CI failure, not silent substitution.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


FROZEN_HF_DATASET_COMMIT = "d5a92c65c549b2a099bdb5a6e61ebeed6cb6eca3"
FROZEN_PATIENT_IMAGE_SHA256 = (
    "de0a5e82647d5c7ca10c1dfd907d47d464351b4c0fe495c1a98fef87912f998e",
    "94a8d90daa1dcaa86b67fa4da13c49028516ee2d48e6870b87510ad0879faeb7",
    "40abe55b8f4f199d3312e7bc2306fb76596760a59a6ba8a45229fafc41efde42",
)
NFBS_SOURCE = "MehdiZouitine/pybrook"
NFBS_REV = "14cd765ea07cf0ea50c9a7ffa56859401e9333a9"
NFBS_FILES = {
    "image.nii.gz": (
        "example/sub-A00028185_ses-NFB3_T1w.nii.gz",
        "f8afc7d0353461a5b94f36d2cd56e3ca265a4fc0",
    ),
    "mask.nii.gz": (
        "example/sub-A00028185_ses-NFB3_T1w_brainmask.nii.gz",
        "65a0e8958f41be0f1802a0435e8f9f3dd85b2840",
    ),
}


def git_blob(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha1(
        b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
    ).hexdigest()


def fetch_nfbs(out: Path) -> dict:
    dest = out / "nfbs_heldout_001"
    dest.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, (remote, expected_blob) in NFBS_FILES.items():
        url = (
            "https://raw.githubusercontent.com/" + NFBS_SOURCE + "/"
            + NFBS_REV + "/" + remote
        )
        data = urlopen(url, timeout=100).read()
        path = dest / name
        path.write_bytes(data)
        got = git_blob(path)
        if got != expected_blob:
            raise ValueError(f"NFBS external independent MRI/label Git blob: {name}")
        hashes[name] = hashlib.sha256(data).hexdigest()
    return {
        "case_id": dest.name,
        "directory": str(dest),
        "dataset": "NFBS skullstripped, publicly mirrored in pybrook/example",
        "independent_subject": "A00028185",
        "image_sha256": hashes["image.nii.gz"],
        "mask_sha256": hashes["mask.nii.gz"],
        "external_source_commit": NFBS_REV,
        "label_provenance": "NFBS publicly released manually corrected mask",
    }


def fetch_synthstrip(out: Path, count: int, start_index: int = 0) -> tuple[list[dict], dict]:
    from huggingface_hub import HfApi, hf_hub_download, list_repo_tree
    dataset = "MedOtter/SynthStrip"
    api = HfApi()
    revision = FROZEN_HF_DATASET_COMMIT
    if api.dataset_info(dataset, revision=revision).sha != revision:
        raise ValueError("Frozen external expert data revision is unavailable")
    nodes = list(
        list_repo_tree(
            dataset, repo_type="dataset", revision=revision, recursive=True
        )
    )
    sizes = {n.path: getattr(n, "size", None) for n in nodes}
    available = sorted(
        p.rsplit("/", 1)[0]
        for p in sizes if p.endswith("/image.nii.gz")
        and p[:-len("image.nii.gz")] + "mask.nii.gz" in sizes
    )
    from huggingface_hub import hf_hub_download

    metadata_candidates = sorted(
        p for p in sizes if p.endswith("scan_metadata.csv")
    )
    if not metadata_candidates:
        raise ValueError("External dataset has no required scan metadata CSV")
    meta_file = hf_hub_download(
        repo_id=dataset, repo_type="dataset", revision=revision,
        filename=metadata_candidates[0]
    )
    metadata = list(csv.DictReader(Path(meta_file).open()))
    by_scan = {row.get("scan_id", ""): row for row in metadata}
    candidates = []
    for folder in available:
        scan = folder.rsplit("/", 1)[-1]
        info = by_scan.get(scan)
        if info is None:
            continue
        modality = str(info.get("modality_full") or info.get("modality") or "")
        if not any(s in modality.lower() for s in ("t1", "mprage", "spgr")):
            continue
        image_size = sizes.get(folder + "/image.nii.gz")
        if image_size is None or image_size > 65_000_000 or image_size < 30000:
            continue
        subject = str(info.get("subject") or scan)
        candidates.append((folder, subject, modality))
    # Deterministic subject-disjoint cohort independent of mask/content
    selection = []
    taken = set()
    for folder, subject, modality in candidates:
        if subject in taken:
            continue
        taken.add(subject)
        selection.append((folder, subject, modality))
        if len(selection) == count + start_index:
            break
    if start_index < 0:
        raise ValueError("Negative frozen prospective cohort offset")
    selection = selection[start_index:] if start_index else selection
    if len(selection) != count:
        raise ValueError(
            f"Only {len(selection)} independent T1 eligible SynthStrip "
            f"pairs exist, needed {count}. Do not replace by unverified labels"
        )
    patients = []
    for i, (folder, subject, modality) in enumerate(selection):
        dest = out / f"synthstrip_heldout_{i+1:03d}"
        dest.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for name in ("image.nii.gz", "mask.nii.gz"):
            source = hf_hub_download(
                repo_id=dataset, repo_type="dataset",
                revision=revision, filename=folder + "/" + name
            )
            target = dest / name
            # Copy bytes into one ephemeral run dir: never publish raw MRI.
            payload = Path(source).read_bytes()
            target.write_bytes(payload)
            hashes[name] = hashlib.sha256(payload).hexdigest()
        patients.append({
            "case_id": dest.name,
            "directory": str(dest),
            "dataset": dataset,
            "independent_subject": subject,
            "modality": modality,
            "image_sha256": hashes["image.nii.gz"],
            "mask_sha256": hashes["mask.nii.gz"],
            "external_dataset_revision": revision,
            "source_folder": folder,
            "label_provenance": "dataset release manually-refined mask claim",
        })
    return patients, {
        "dataset": dataset,
        "revision": revision,
        "total_image_mask_pairs_discovered": len(available),
        "eligible_t1_subjects_metadata": len(candidates),
        "predeclared_subject_selection": (
            "lexicographically first distinct-subject T1 acquisitions "
            "with public image+mask and compressed image <=65 MB, "
            "without looking at mask content, image output or model predictions"
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--synthstrip-subjects", type=int, default=2)
    ap.add_argument("--selection-offset", type=int, default=0)
    ap.add_argument("--skip-nfbs", action="store_true")
    a = ap.parse_args()
    if a.synthstrip_subjects < 1:
        raise ValueError("Need at least one independent extra subject")
    a.output_root.mkdir(exist_ok=True, parents=True)
    nfbs = None if a.skip_nfbs else fetch_nfbs(a.output_root)
    others, provider = fetch_synthstrip(
        a.output_root, a.synthstrip_subjects,
        start_index=a.selection_offset
    )
    records = ([] if nfbs is None else [nfbs]) + others
    if a.synthstrip_subjects == 2 and a.selection_offset == 0 and not a.skip_nfbs and tuple(
        r["image_sha256"] for r in records
    ) != FROZEN_PATIENT_IMAGE_SHA256:
        raise ValueError("Frozen independently preselected subject MRI provenance changed")
    if len({p["independent_subject"] for p in records}) != len(records):
        raise ValueError("cross-source identifier collision")
    manifest = {
        "data_selection_label_blind": True,
        "frozen_public_cohort_offset": a.selection_offset,
        "nfbs_included": not a.skip_nfbs,
        "no_mask_tuned_crop": True,
        "no_ground_truth_used_to_select_subjects": True,
        "subjects": records,
        "hf_source": provider,
        "expert_mask_provenance": "repository dataset documentation; no independent raters",
        "raw_data_not_redistributed": True,
    }
    (a.output_root / "external_subject_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps({
        "subject_count": len(records),
        "datasets": [r["dataset"] for r in records],
        "subject_sha256": [r["image_sha256"] for r in records],
        "dataset_revision": provider["revision"],
        "selection_rule": provider["predeclared_subject_selection"],
    }))


if __name__ == "__main__":
    main()
