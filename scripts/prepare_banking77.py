"""Prepare BANKING77 while retaining the official test examples unchanged."""

import csv
import hashlib
import io
import json
import urllib.request
from collections import Counter

from sklearn.model_selection import train_test_split

from data_utils import ROOT, normalize

BASE_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
HASHES = {
    "train": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}


def main():
    raw = ROOT / "data" / "raw" / "banking77"
    raw.mkdir(parents=True, exist_ok=True)
    source = {}
    for name, checksum in HASHES.items():
        path = raw / f"{name}.csv"
        payload = path.read_bytes() if path.exists() else urllib.request.urlopen(BASE_URL + path.name, timeout=60).read()
        if hashlib.sha256(payload).hexdigest() != checksum:
            raise ValueError(f"Unexpected source checksum for {name}")
        if not path.exists():
            path.write_bytes(payload)
        source[name] = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
        if any(not r["text"].strip() or not r["category"].strip() for r in source[name]):
            raise ValueError(f"Missing text or label in {name}")
    assert len(source["train"]) == 10003 and len(source["test"]) == 3080
    labels = sorted({r["category"] for r in source["train"]})
    assert len(labels) == 77 and set(labels) == {r["category"] for r in source["test"]}
    test_keys = {normalize(r["text"]) for r in source["test"]}
    unique = {}
    duplicates = 0
    overlaps = 0
    for row in source["train"]:
        key = normalize(row["text"])
        if key in test_keys:
            overlaps += 1
            continue
        if key in unique:
            if unique[key]["category"] != row["category"]:
                raise ValueError("Conflicting labels on duplicate training text")
            duplicates += 1
            continue
        unique[key] = row
    usable = list(unique.values())
    train, validation = train_test_split(
        usable, test_size=0.2, random_state=42,
        stratify=[r["category"] for r in usable],
    )
    splits = {"train": train, "validation": validation, "test": source["test"]}
    seen = set()
    for name, rows in splits.items():
        keys = {normalize(r["text"]) for r in rows}
        assert not keys & seen, f"Cross-split exact overlap in {name}"
        seen.update(keys)
        assert {r["category"] for r in rows} == set(labels)
    output = ROOT / "data" / "processed" / "banking77"
    output.mkdir(parents=True, exist_ok=True)
    label2id = {label: i for i, label in enumerate(labels)}
    for name, rows in splits.items():
        with (output / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["text", "intent", "label"])
            writer.writeheader()
            writer.writerows({"text": r["text"], "intent": r["category"], "label": label2id[r["category"]]} for r in rows)
    # IDs are our explicitly saved sorted-name mapping, not assumed HF IDs.
    (output / "label_mapping.json").write_text(json.dumps(label2id, indent=2) + "\n", encoding="utf-8")
    report = {
        "dataset": "BANKING77", "publisher": "PolyAI", "license": "CC BY 4.0",
        "source_base_url": BASE_URL, "source_sha256": HASHES, "seed": 42,
        "source_rows": {n: len(r) for n, r in source.items()},
        "training_duplicate_copies_removed": duplicates,
        "training_rows_matching_official_test_removed": overlaps,
        "test_internal_exact_duplicate_copies": len(source["test"]) - len(test_keys),
        "split_counts": {n: len(r) for n, r in splits.items()},
        "intent_counts": {n: dict(sorted(Counter(r["category"] for r in rows).items())) for n, rows in splits.items()},
        "checks": {"no_missing_text_or_labels": True, "all_77_intents_in_each_split": True,
                   "no_normalized_exact_text_overlap": True, "official_test_text_labels_order_preserved": True},
        "test_set_evaluated": False,
        "limitations": ["Exact duplicate audit only; semantic near-duplicates not checked",
                        "Known label errors may remain; no relabeling performed"],
    }
    # Verify persisted test text/labels/order against the official source.
    with (output / "test.csv").open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert [(r["text"], r["intent"]) for r in saved] == [(r["text"], r["category"]) for r in source["test"]]
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "intent_counts"}, indent=2))


if __name__ == "__main__":
    main()
