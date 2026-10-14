"""Package only training/validation inputs and their matching baseline metrics."""

import hashlib
import json
import zipfile

from data_utils import ROOT


def main():
    data = ROOT / "data" / "processed" / "banking77"
    metrics_path = ROOT / "reports" / "banking77_baseline" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    files = {name: data / name for name in ("train.csv", "validation.csv", "label_mapping.json")}
    files["baseline_metrics.json"] = metrics_path
    for split in ("train", "validation"):
        assert hashlib.sha256(files[f"{split}.csv"].read_bytes()).hexdigest() == metrics["input_sha256"][split]
    assert hashlib.sha256(files["label_mapping.json"].read_bytes()).hexdigest() == metrics["label_mapping_sha256"]
    destination = ROOT / "data" / "colab" / "banking77_colab_inputs.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, arcname=name)
    with zipfile.ZipFile(destination) as archive:
        assert set(archive.namelist()) == set(files)
        assert archive.testzip() is None
    print(f"Created {destination} ({destination.stat().st_size:,} bytes). Test set excluded.")


if __name__ == "__main__":
    main()
