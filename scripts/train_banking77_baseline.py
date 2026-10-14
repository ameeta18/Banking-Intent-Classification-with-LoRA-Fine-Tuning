"""Benchmark BANKING77 on validation only; never load the official test CSV."""

import csv
import hashlib
import json
import platform
import time
import warnings
from collections import Counter

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from data_utils import ROOT, normalize


def main():
    data = ROOT / "data" / "processed" / "banking77"
    mapping_path = data / "label_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    labels = sorted(mapping, key=mapping.get)
    assert len(labels) == 77 and sorted(mapping.values()) == list(range(77))
    paths = {name: data / f"{name}.csv" for name in ("train", "validation")}
    splits = {}
    for name, path in paths.items():
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or any(not r["text"].strip() or r["intent"] not in mapping for r in rows):
            raise ValueError(f"Invalid input in {name}")
        assert all(int(r["label"]) == mapping[r["intent"]] for r in rows)
        assert {r["intent"] for r in rows} == set(labels)
        splits[name] = rows
    assert not ({normalize(r["text"]) for r in splits["train"]}
                & {normalize(r["text"]) for r in splits["validation"]})
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("classifier", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)),
    ])
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit([r["text"] for r in splits["train"]], [r["intent"] for r in splits["train"]])
    training_seconds = time.perf_counter() - start
    rows = splits["validation"]
    texts, actual = [r["text"] for r in rows], [r["intent"] for r in rows]
    predicted = model.predict(texts)
    scores = model.predict_proba(texts)
    matrix = confusion_matrix(actual, predicted, labels=labels)
    errors = Counter((a, p) for a, p in zip(actual, predicted) if a != p)
    report = {
        "dataset": "BANKING77", "target": "intent", "label_count": 77,
        "model": "word unigram/bigram TF-IDF + Logistic Regression",
        "settings": {"sublinear_tf": True, "C": 1.0, "solver": "lbfgs", "max_iter": 1000, "seed": 42},
        "evaluation_split": "validation", "hyperparameter_search": False,
        "train_rows": len(splits["train"]), "validation_rows": len(rows),
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(actual, predicted, labels=labels, average="macro"),
        "error_count": sum(errors.values()),
        "classification_report": classification_report(actual, predicted, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": {"labels": labels, "rows": "actual", "columns": "predicted", "values": matrix.tolist()},
        "top_confusions": [{"actual": a, "predicted": p, "count": n} for (a, p), n in errors.most_common(10)],
        "training_seconds": training_seconds,
        "vocabulary_size": len(model.named_steps["tfidf"].vocabulary_),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__},
        "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        "label_mapping_sha256": hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
        "test_set_loaded": False,
        "limitations": ["Single split and seed", "No hyperparameter tuning", "Uncalibrated confidence scores",
                        "Exact text overlap checked; semantic near-duplicates may remain"],
    }
    models = ROOT / "models" / "banking77"
    output = ROOT / "reports" / "banking77_baseline"
    models.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    artifact = models / "baseline.joblib"
    joblib.dump(model, artifact)
    restored = joblib.load(artifact)
    assert restored.predict(texts).tolist() == predicted.tolist()
    np.testing.assert_allclose(restored.predict_proba(texts), scores)
    report["checks"] = {"no_exact_train_validation_overlap": True, "all_77_intents_present": True,
                        "label_mapping_consistent": True, "converged": True, "saved_model_reload": True}
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual / predicted", *labels])
        writer.writerows([label, *values] for label, values in zip(labels, matrix.tolist()))
    with (output / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "actual_intent", "predicted_intent", "confidence", "correct"])
        writer.writerows((r["text"], a, p, float(max(s)), a == p) for r, a, p, s in zip(rows, actual, predicted, scores))
    print(f"Validation accuracy: {report['accuracy']:.2%}; macro-F1: {report['macro_f1']:.4f}")
    print(f"Errors: {report['error_count']}/{len(rows)}; training: {training_seconds:.1f} seconds")
    print(json.dumps(report["top_confusions"], indent=2))
    print("All checks passed. Official test set was not loaded.")


if __name__ == "__main__":
    main()
