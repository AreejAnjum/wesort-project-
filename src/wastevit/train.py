from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader

from .data import ManifestDataset, UnlabeledPairDataset
from .model import SimCLR, build_encoder, nt_xent_loss


def build_run_summary(cfg, results):
    metadata = cfg.get("metadata", {})
    baseline = results["baseline"]
    adapted = results["ssl_adapted"]
    return f"""# Experiment summary

- Dataset: {metadata.get('dataset_name', 'not specified')}
- Dataset status: {metadata.get('dataset_status', 'not specified')}
- Purpose: {metadata.get('run_purpose', 'not specified')}
- SSL technique: SimCLR (two augmented views, projection head, NT-Xent loss)
- Backbone: {cfg['model']['backbone']} (pretrained={cfg['model']['pretrained']})
- Evaluation: frozen encoder with a supervised linear probe

| Variant | Test accuracy | Test macro-F1 |
|---|---:|---:|
| Baseline encoder | {baseline['accuracy']:.4f} | {baseline['macro_f1']:.4f} |
| SSL-adapted encoder | {adapted['accuracy']:.4f} | {adapted['macro_f1']:.4f} |
| SSL delta | {adapted['accuracy'] - baseline['accuracy']:+.4f} | {adapted['macro_f1'] - baseline['macro_f1']:+.4f} |

These metrics are valid only for the dataset status stated above. A synthetic
smoke test verifies the software path but is not research evidence.
"""


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(value: str):
    if value != "auto": return torch.device(value)
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed); random.seed(worker_seed)


def loader(dataset, cfg, shuffle=False, drop_last=False):
    generator = torch.Generator().manual_seed(cfg["seed"])
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=shuffle,
                      num_workers=cfg["num_workers"], drop_last=drop_last,
                      worker_init_fn=seed_worker, generator=generator)


def train_ssl(encoder, feature_dim, dataset, cfg, device):
    model = SimCLR(encoder, feature_dim, cfg["projection_dim"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    train_loader = loader(dataset, cfg, shuffle=True, drop_last=True)
    for epoch in range(cfg["ssl_epochs"]):
        model.train(); total = 0.0
        for view1, view2 in train_loader:
            view1, view2 = view1.to(device), view2.to(device)
            optimizer.zero_grad()
            loss = nt_xent_loss(model(view1), model(view2), cfg["temperature"])
            loss.backward(); optimizer.step(); total += loss.item() * view1.size(0)
        mlflow.log_metric("ssl_loss", total / len(dataset), step=epoch)
    return model.encoder


def evaluate_probe(encoder, feature_dim, train, val, test, cfg, device):
    encoder = encoder.to(device).eval()
    for parameter in encoder.parameters(): parameter.requires_grad = False
    head = nn.Linear(feature_dim, len(train.class_to_idx)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["probe_learning_rate"])
    loss_fn = nn.CrossEntropyLoss()
    best_state, best_val = None, -1.0
    train_loader = loader(train, cfg, shuffle=True)
    for epoch in range(cfg["probe_epochs"]):
        head.train(); total = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad(): features = encoder(images)
            optimizer.zero_grad(); loss = loss_fn(head(features), labels)
            loss.backward(); optimizer.step(); total += loss.item() * images.size(0)
        mlflow.log_metric("probe_train_loss", total / len(train), step=epoch)
        val_metrics, _ = predict(encoder, head, val, cfg, device)
        mlflow.log_metric("val_macro_f1", val_metrics["macro_f1"], step=epoch)
        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            best_state = deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    metrics, report = predict(encoder, head, test, cfg, device)
    return head, metrics, report


def predict(encoder, head, dataset, cfg, device):
    truth, pred = [], []
    encoder.eval(); head.eval()
    with torch.no_grad():
        for images, labels in loader(dataset, cfg):
            output = head(encoder(images.to(device))).argmax(1).cpu().tolist()
            truth.extend(labels.tolist()); pred.extend(output)
    metrics = {"accuracy": accuracy_score(truth, pred), "macro_f1": f1_score(truth, pred, average="macro", zero_division=0)}
    names = [name for name, _ in sorted(dataset.class_to_idx.items(), key=lambda x: x[1])]
    report = classification_report(truth, pred, labels=list(range(len(names))), target_names=names, zero_division=0, output_dict=True)
    return metrics, report


def run(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text())
    set_seed(cfg["seed"]); device = resolve_device(cfg["device"])
    mlflow.set_tracking_uri(cfg["tracking_uri"]); mlflow.set_experiment(cfg["experiment_name"])
    image_size = cfg["data"]["image_size"]
    unlabeled = UnlabeledPairDataset(cfg["data"]["unlabeled_dir"], image_size)
    train = ManifestDataset(cfg["data"]["train_manifest"], image_size)
    val = ManifestDataset(cfg["data"]["val_manifest"], image_size, train.class_to_idx)
    test = ManifestDataset(cfg["data"]["test_manifest"], image_size, train.class_to_idx)
    split_paths = {"ssl": {p.resolve() for p in unlabeled.paths}, "train": {p.resolve() for p in train.paths},
                   "val": {p.resolve() for p in val.paths}, "test": {p.resolve() for p in test.paths}}
    for left, right in (("ssl", "val"), ("ssl", "test"), ("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_paths[left] & split_paths[right]
        if overlap: raise ValueError(f"Data leakage: {left}/{right} share {len(overlap)} image path(s)")
    model_cfg = cfg["model"]; train_cfg = {**cfg["training"], "projection_dim": model_cfg["projection_dim"], "seed": cfg["seed"]}
    if len(unlabeled) < train_cfg["batch_size"] or train_cfg["batch_size"] < 2:
        raise ValueError("SSL needs at least one full batch and batch_size >= 2")
    with mlflow.start_run(run_name=cfg["run_name"]) as parent:
        metadata = cfg.get("metadata", {})
        mlflow.set_tags({"project": "WasteViT", "method": "SimCLR",
                         "evaluation": "frozen-linear-probe", **metadata})
        mlflow.log_params({"seed": cfg["seed"], "device": str(device), "backbone": model_cfg["backbone"],
                           "pretrained": model_cfg["pretrained"], "unlabeled_images": len(unlabeled),
                           "train_images": len(train), "val_images": len(val), "test_images": len(test), **cfg["training"]})
        mlflow.log_artifact(config_path, "config")
        encoder, dim = build_encoder(model_cfg["backbone"], model_cfg["pretrained"])
        initial = deepcopy(encoder.state_dict())
        results = {}
        for stage in ("baseline", "ssl_adapted"):
            with mlflow.start_run(run_name=stage, nested=True):
                mlflow.set_tags({"project": "WasteViT", "method": "SimCLR",
                                 "variant": stage, **metadata})
                set_seed(cfg["seed"])
                encoder.load_state_dict(initial)
                if stage == "ssl_adapted": encoder = train_ssl(encoder, dim, unlabeled, train_cfg, device)
                # Identical probe initialization makes the baseline comparison fair.
                set_seed(cfg["seed"] + 1)
                head, metrics, report = evaluate_probe(encoder, dim, train, val, test, train_cfg, device)
                mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})
                mlflow.log_dict(report, "classification_report.json")
                checkpoint = Path("artifacts") / f"{stage}.pt"; checkpoint.parent.mkdir(exist_ok=True)
                torch.save({"encoder": encoder.state_dict(), "head": head.state_dict(), "classes": train.class_to_idx}, checkpoint)
                mlflow.log_artifact(str(checkpoint), "checkpoints"); results[stage] = metrics
        mlflow.log_metrics({"ssl_accuracy_delta": results["ssl_adapted"]["accuracy"] - results["baseline"]["accuracy"],
                            "ssl_macro_f1_delta": results["ssl_adapted"]["macro_f1"] - results["baseline"]["macro_f1"]})
        Path("artifacts/results.json").write_text(json.dumps(results, indent=2))
        mlflow.log_artifact("artifacts/results.json")
        summary = build_run_summary(cfg, results)
        Path("artifacts/experiment_summary.md").write_text(summary)
        mlflow.log_artifact("artifacts/experiment_summary.md")
        return parent.info.run_id, results


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    args = parser.parse_args(); run_id, results = run(args.config)
    print(json.dumps({"run_id": run_id, "results": results}, indent=2))


if __name__ == "__main__": main()
