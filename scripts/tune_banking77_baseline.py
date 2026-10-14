"""Small, predefined validation-only search; preserves the original baseline."""

import csv
import hashlib
import json
import time
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from data_utils import ROOT, normalize


def main():
    data = ROOT / 'data' / 'processed' / 'banking77'
    original = json.loads((ROOT / 'reports/banking77_baseline/metrics.json').read_text())
    mapping_path = data / 'label_mapping.json'
    assert hashlib.sha256(mapping_path.read_bytes()).hexdigest() == original['label_mapping_sha256']
    mapping = json.loads(mapping_path.read_text())
    labels = sorted(mapping, key=mapping.get)
    splits = {}
    for name in ['train', 'validation']:
        path = data / f'{name}.csv'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == original['input_sha256'][name]
        with path.open(encoding='utf-8', newline='') as handle:
            splits[name] = list(csv.DictReader(handle))
        assert {r['intent'] for r in splits[name]} == set(labels)
        assert all(r['text'].strip() and int(r['label']) == mapping[r['intent']] for r in splits[name])
    assert not ({normalize(r['text']) for r in splits['train']} & {normalize(r['text']) for r in splits['validation']})
    train_text = [r['text'] for r in splits['train']]
    train_labels = [r['intent'] for r in splits['train']]
    val_text = [r['text'] for r in splits['validation']]
    actual = [r['intent'] for r in splits['validation']]
    trials, best_score, best_model, best_settings = [], -1, None, None
    # Declare the complete search up front; no test-based or adaptive selection.
    grid = [(ngram, c) for ngram in [(1, 1), (1, 2)] for c in [1.0, 3.0, 10.0]]
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        for ngram, c in grid:
            model = Pipeline([
                ('tfidf', TfidfVectorizer(ngram_range=ngram, sublinear_tf=True)),
                ('classifier', LogisticRegression(C=c, solver='lbfgs', max_iter=1000, random_state=42)),
            ])
            start = time.perf_counter()
            model.fit(train_text, train_labels)
            elapsed = time.perf_counter() - start
            predicted = model.predict(val_text)
            score = float(f1_score(actual, predicted, labels=labels, average='macro'))
            settings = dict(ngram_range=list(ngram), C=c)
            trial = dict(**settings, accuracy=float(accuracy_score(actual, predicted)), macro_f1=score, training_seconds=elapsed)
            trials.append(trial)
            print(json.dumps(trial), flush=True)
            if score > best_score:
                best_score, best_model, best_settings = score, model, settings
    predicted = best_model.predict(val_text)
    probabilities = best_model.predict_proba(val_text)
    output = ROOT / 'reports/banking77_tuned_baseline'
    output.mkdir(parents=True, exist_ok=True)
    artifact = ROOT / 'models/banking77/tuned_baseline.joblib'
    joblib.dump(best_model, artifact)
    restored = joblib.load(artifact)
    assert restored.predict(val_text).tolist() == predicted.tolist()
    np.testing.assert_allclose(restored.predict_proba(val_text), probabilities)
    report = dict(dataset='BANKING77', model='TF-IDF + Logistic Regression',
                  selection_metric='validation macro-F1', tie_break='first setting in predefined grid',
                  trials=trials, selected_settings=best_settings,
                  fixed_settings=dict(sublinear_tf=True, solver='lbfgs', max_iter=1000, seed=42),
                  accuracy=float(accuracy_score(actual, predicted)), macro_f1=best_score,
                  error_count=int(sum(a != p for a, p in zip(actual, predicted))),
                  train_rows=len(train_text), validation_rows=len(actual),
                  input_sha256=original['input_sha256'], label_mapping_sha256=original['label_mapping_sha256'],
                  sklearn_version=sklearn.__version__, test_set_loaded=False,
                  checks=dict(saved_model_reload=True, no_exact_split_overlap=True, inputs_match_original_baseline=True),
                  classification_report=classification_report(actual, predicted, labels=labels, output_dict=True, zero_division=0))
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    with (output / 'confusion_matrix.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['actual / predicted', *labels])
        writer.writerows([label, *row] for label, row in zip(labels, confusion_matrix(actual, predicted, labels=labels).tolist()))
    with (output / 'validation_predictions.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['text', 'actual_intent', 'predicted_intent', 'confidence', 'correct'])
        writer.writerows((t, a, p, float(max(s)), a == p) for t, a, p, s in zip(val_text, actual, predicted, probabilities))
    print('SELECTED', best_settings, 'accuracy', report['accuracy'], 'macro-F1', best_score, flush=True)
    print('All checks passed. Official test data was not loaded.', flush=True)


if __name__ == '__main__':
    main()
