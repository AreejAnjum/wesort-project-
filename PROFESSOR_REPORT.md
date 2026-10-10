# WasteViT self-supervised learning pipeline - project update

## What was completed

I reviewed the supplied waste datasets and built a reproducible self-supervised
learning pipeline. It compares a pretrained baseline with the same encoder after
waste-domain self-supervised adaptation. MLflow records the configuration,
parameters, training curves, metrics, reports and checkpoints.

## Dataset decision

The listed Kaggle waste datasets are arranged in class folders, so they are
labeled rather than dedicated unlabeled datasets. Their labels can be hidden
during SSL, but their clean images are a weak match for an industrial belt.

The recommended real dataset is:

- **ZeroWaste-s:** 6,212 unlabeled industrial conveyor frames intended for
  semi-supervised and self-supervised learning.
- **ZeroWaste-f:** labeled train, validation and test frames for evaluation.

It is closer to the future WeSort.AI setting because it includes clutter,
overlap, deformation and belt backgrounds.

## Technique used: SimCLR

For each unlabeled image, the pipeline creates two augmented views using random
crops, flips, color changes, grayscale and blur. An encoder maps both views to
representations. A projection head is trained with NT-Xent contrastive loss so
views of the same image become similar and different images remain distinct.

The real configuration uses a pretrained ResNet-18. After SSL, the encoder is
frozen and a linear classifier is trained on labeled data. This tests whether
SSL produced more useful features.

## Controlled comparison

1. **Baseline:** pretrained frozen encoder plus linear classifier.
2. **SSL adapted:** identical starting encoder, SimCLR on unlabeled images,
   followed by the same linear-classifier evaluation.
3. **Decision metric:** change in test accuracy and macro-F1.

Both probes use the same random initialization. Validation macro-F1 selects the
best probe, the test set is not used for selection, and the pipeline rejects
exact image-path leakage between training and evaluation splits.

## Results currently recorded in MLflow

The completed run used generated images only to verify that every stage works.

| Result | Baseline | SSL adapted | Change |
|---|---:|---:|---:|
| Test accuracy | 0.500 | 0.500 | 0.000 |
| Test macro-F1 | 0.333 | 0.333 | 0.000 |

MLflow run ID: `b482549b78d34dfb9d2145e80e31829c`.

These are **software smoke-test results, not research results**. They verify
data loading, SSL training, evaluation, checkpoint saving and MLflow tracking.
They do not show whether SSL improves industrial waste recognition.

## What MLflow records

- Dataset name, dataset status and experiment purpose
- Encoder, pretrained status, seed, device and image counts
- Training hyperparameters and SSL loss per epoch
- Probe loss and validation macro-F1 per epoch
- Test accuracy, macro-F1 and per-class classification report
- Baseline-to-SSL difference
- Configuration, checkpoints, results JSON and experiment summary

Start the local interface with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Real-world use at WeSort.AI

Unlabeled frames collected continuously from sorting machines can support
periodic SSL adaptation without paying to label every frame. A smaller,
carefully labeled belt dataset can train and evaluate the downstream task. The
encoder can then feed a classifier, detector or segmentation model. New plants
or seasons can supply more unlabeled frames for adaptation, while fixed plant-
and time-separated tests monitor performance degradation.

Production evaluation should include mIoU, per-class IoU, mAP, latency and
robustness across plants, lighting, dirt, overlap and time periods.

## Limitation and counterargument

SimCLR benefits from large batches and good augmentations. With limited
hardware, MAE or a DINO-style teacher-student method may be more stable. SSL on
clean Kaggle images may adapt the model to the wrong domain. Therefore the real
conclusion must come from ZeroWaste or WeSort.AI belt images, not the toy run.

## Next experiment

1. Obtain ZeroWaste-s and ZeroWaste-f.
2. Generate manifests from the ZeroWaste COCO annotations.
3. Run `configs/zerowaste_simclr.yaml`.
4. Compare baseline and SSL-adapted macro-F1 over several seeds.
5. Add segmentation evaluation before claiming industrial effectiveness.
