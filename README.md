# WasteViT SSL pipeline

Reproducible proof-of-concept for the THWS / WeSort.AI project. It compares a
frozen pretrained encoder with the same encoder after SimCLR self-supervised
domain adaptation. Every run, parameter, metric, checkpoint, dataset audit and
test report is recorded in MLflow.

## Recommended experiment

Use **ZeroWaste-s** (6,212 unlabeled conveyor-belt frames) for SSL and
**ZeroWaste-f** for the labeled linear-probe/evaluation stage. This matches the
project documents much more closely than clean, single-object Kaggle datasets.

Expected layout (other layouts also work when paths are changed in YAML):

```text
data/
  zerowaste-s/data/*.jpg
  zerowaste-f/train/data/*.jpg
  zerowaste-f/train/labels.json
  zerowaste-f/val/data/*.jpg
  zerowaste-f/val/labels.json
  zerowaste-f/test/data/*.jpg
  zerowaste-f/test/labels.json
```

The first implementation evaluates image-level classification. For the
segmentation annotations in ZeroWaste, `prepare_zerowaste.py` converts COCO
annotations to one multi-class image label using the dominant annotated object.
This is an intentionally simple probe of representation quality, not the final
industrial segmentation metric.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python scripts/prepare_zerowaste.py \
  --coco data/zerowaste-f/train/labels.json \
  --images data/zerowaste-f/train/data \
  --output data/manifests/train.csv
python scripts/prepare_zerowaste.py --coco data/zerowaste-f/val/labels.json --images data/zerowaste-f/val/data --output data/manifests/val.csv
python scripts/prepare_zerowaste.py --coco data/zerowaste-f/test/labels.json --images data/zerowaste-f/test/data --output data/manifests/test.csv
python -m wastevit.train --config configs/zerowaste_simclr.yaml
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://127.0.0.1:5000`. Two nested runs appear under the parent run:
`baseline` and `ssl_adapted`. The key comparison is `test_macro_f1` and
`test_accuracy`; SSL is useful only if the adapted encoder beats the baseline.

For a fast end-to-end check with generated images:

```bash
python scripts/make_toy_data.py --output data/toy
python -m wastevit.train --config configs/toy_smoke.yaml
pytest --junitxml=reports/pytest.xml
```

## Data rules

- Never infer that a folder is unlabeled merely because no CSV is present;
  class-named folders are labels.
- SSL may read labeled images, but the dataset object discards the label.
- Validation/test images never enter SSL training.
- Prefer plant/time-separated evaluation. ZeroWaste-v1 to v2 is a useful domain
  shift test; real WeSort evaluation should be split by plant and time.
- Do not publish or commit WeSort confidential data, images, or checkpoints.

The repository intentionally excludes private project documents, datasets,
generated checkpoints and local MLflow databases.
