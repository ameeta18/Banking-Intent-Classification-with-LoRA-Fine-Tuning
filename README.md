# AI Banking-Support Intent Router

Banking-support intent classification across 77 intents using DistilBERT with
LoRA fine-tuning. The repository includes reproducible data preparation,
TF-IDF baseline training and tuning, and a Colab notebook comparing a frozen
DistilBERT encoder with LoRA adaptation.

**Tech stack:** Python, PyTorch, Hugging Face Transformers, PEFT, scikit-learn,
and Google Colab.

## Validation results

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 84.79% | 0.8405 |
| Tuned TF-IDF + Logistic Regression | 87.94% | 0.8802 |
| Frozen DistilBERT with trained classification head | 64.18% | 0.6210 |
| **DistilBERT + LoRA** | **89.39%** | **0.8950** |

LoRA improves accuracy by **1.45 percentage points** over tuned TF-IDF,
with 29 fewer errors on 1,999 validation queries. It trains approximately
797,000 parameters. Scores are from a single validation split; the official
test set was not evaluated.

## BANKING77 data preparation

```powershell
python -m pip install -r requirements-baseline.txt
python scripts/prepare_banking77.py
```

Source: [PolyAI's official BANKING77 repository](https://github.com/PolyAI-LDN/task-specific-datasets).
License: CC BY 4.0; credit PolyAI and Casanueva et al. (2020),
*Efficient Intent Detection with Dual Sentence Encoders*.

Official CSV contents are pinned by SHA-256. Preparation removes normalized
exact duplicate training requests and any training copies matching the official
test text, then makes an 80/20 stratified train/validation split with seed 42.
The official test text, labels and order remain unchanged. Reading test text
for overlap checks is a data-integrity audit, not a model evaluation.

Files live in `data/processed/banking77/`: `train.csv`, `validation.csv`,
`test.csv`, `label_mapping.json`, and `report.json`. The model input is `text`;
the target is `intent` or its saved numeric `label`. Use the explicit mapping
for every model; do not assume it matches another library's label IDs.
This preparation checks exact duplicates, not semantic near-duplicates.
Known annotation errors may remain. No manual labeling is required.

Run the BANKING77 baseline locally on CPU:

```powershell
python scripts/train_banking77_baseline.py
```

It fits word unigram/bigram TF-IDF and Logistic Regression on training data
only, evaluates validation data, and never loads the official test CSV.
Results are saved in `reports/banking77_baseline/metrics.json`, alongside a
confusion matrix and individual validation predictions. The fitted pipeline
is saved as `models/banking77/baseline.joblib`. The script checks labels,
exact train/validation overlap, convergence, and reloaded predictions and
probabilities. This is a fixed-settings initial baseline, not a tuned result.

Initial BANKING77 validation result: **84.79% accuracy, macro-F1 0.8405**,
with 304 mistakes among 1,999 validation queries after training on 7,993.
Training took about 31 seconds in the local environment (hardware-dependent).
Common confusions include failed versus pending top-ups and identity
verification failures versus questions about why verification is needed.
This fixed-settings run provides the reference for the tuned baseline and
transformer comparisons below.

## Colab: frozen DistilBERT and LoRA

### Saved experiment results and tuned baseline

Downloaded Colab artifacts are stored locally in `models/frozen_artifacts.zip`,
`models/lora_artifacts.zip`, and `models/comparison.csv`. Their ZIP integrity,
data hashes, label mappings, and prediction-file metrics were checked locally;
the LoRA checkpoint includes adapters and both trained head layers. See
`reports/colab_artifact_audit.json`. This audit does not execute the transformers
locally. Colab reported successful best-checkpoint reload checks.

Run `python scripts/tune_banking77_baseline.py` for the predefined six-setting
validation search: word unigrams or unigrams/bigrams, with C = 1, 3, or 10.
Selection uses macro-F1; ties retain the first setting. The winner uses unigrams
and C = 10: accuracy 87.94%, macro-F1 0.8802. It is saved separately as
`models/banking77/tuned_baseline.joblib`; detailed results are in
`reports/banking77_tuned_baseline/metrics.json`. The initial baseline is preserved.

LoRA validation accuracy is 89.39%, macro-F1 0.8950; frozen DistilBERT scores
64.18% and 0.6210. LoRA leads the tuned TF-IDF by 1.45 percentage points of
accuracy, or 29 fewer errors among 1,999 validation examples. These results
come from a single split, with different tuning budgets, and do not establish
statistical significance or test performance. The official test set is excluded
from model selection and the reported evaluation.

Open `notebooks/banking77_lora.ipynb` through Google Colab's **File > Upload
notebook**. Select **Runtime > Change runtime type > T4 GPU** (or another
available GPU). Run the cells in order. The notebook installs its own pinned
transformer dependencies; do not install the local baseline requirements in Colab.
The setup uses Transformers 4.57.6, Tokenizers 0.22.2, PEFT 0.18.1,
and Accelerate 1.12.0, with binary-only installation to avoid compiling an
older tokenizer on newer Python runtimes. Restart the Colab session after
installation, then continue with the imports cell. Actual dependency versions
are recorded in the experiment metrics. Both experiments have run successfully
in Colab; downloaded predictions and metrics were subsequently checked locally.

When prompted, upload `data/colab/banking77_colab_inputs.zip`. To rebuild it:

```powershell
python scripts/package_colab_data.py
```

The bundle contains training/validation CSVs, the label mapping, and baseline
metrics. Hash checks ensure the inputs match the baseline. Test data is excluded.

Both experiments use five epochs, batch size 16, maximum length 128, seed 42,
and AdamW learning rate 0.0003. Both train the two classification-head layers
from the same seeded initialization. LoRA additionally trains rank-8 adapters
on attention query/value layers. Best checkpoints are selected by validation
macro-F1 and reloaded to verify metrics. The shared training configuration
provides a controlled comparison; it is not an exhaustive hyperparameter search.

Download `frozen_artifacts.zip` and `lora_artifacts.zip` after their respective
training cells, then `comparison.csv` at the end. Colab storage is temporary;
interrupted training restarts rather than resuming optimizer state. Each ZIP
includes metrics, predictions, confusion matrix, label mapping and tokenizer.
The LoRA ZIP saves the adapters and both trained head layers, and needs the
base model revision recorded in its metrics when reloaded.

Notebook schema, Python cell syntax, and the upload/hash-validation cell have
been checked locally against the real bundle. GPU training and saved-checkpoint
checks completed in Colab. The scores above are validation results.
CPU versus GPU timing is not a
hardware-matched speed comparison.

## Repository contents

The repository contains scripts, a notebook, requirements, and aggregate
reports (metrics, confusion matrices, artifact audit). Raw/processed datasets,
trained models, downloaded ZIPs, individual prediction files and the Colab upload
bundle are generated local artifacts excluded by `.gitignore`. Recreate the
data and baseline with the commands above; run Colab to recreate the transformer
artifacts. Model ZIPs mentioned above are not included in a fresh clone.
