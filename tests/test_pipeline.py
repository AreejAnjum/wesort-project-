from pathlib import Path
import csv
import torch
from PIL import Image

from wastevit.data import ManifestDataset, UnlabeledPairDataset
from wastevit.model import nt_xent_loss
from wastevit.train import build_run_summary


def test_unlabeled_dataset_discards_folder_names(tmp_path: Path):
    folder = tmp_path / "looks_like_a_class"; folder.mkdir()
    Image.new("RGB", (20, 20), "red").save(folder / "sample.png")
    dataset = UnlabeledPairDataset(tmp_path, 16)
    first, second = dataset[0]
    assert first.shape == second.shape == (3, 16, 16)


def test_manifest_class_mapping_is_stable(tmp_path: Path):
    paths=[]
    for i in range(2):
        path=tmp_path/f"{i}.png"; Image.new("RGB",(10,10)).save(path); paths.append(path)
    manifest=tmp_path/"data.csv"
    with manifest.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["path","label"]); w.writeheader(); w.writerows([{"path":paths[0],"label":"z"},{"path":paths[1],"label":"a"}])
    dataset=ManifestDataset(manifest,8)
    assert dataset.class_to_idx == {"a": 0, "z": 1}


def test_nt_xent_prefers_matching_pairs():
    z=torch.eye(4)
    good=nt_xent_loss(z,z.clone(),0.2)
    bad=nt_xent_loss(z,z.roll(1,0),0.2)
    assert good < bad


def test_nt_xent_rejects_singleton_batch():
    try:
        nt_xent_loss(torch.ones(1, 4), torch.ones(1, 4), 0.2)
    except ValueError as error:
        assert "at least two" in str(error)
    else:
        raise AssertionError("singleton contrastive batch should fail")


def test_run_summary_labels_synthetic_results():
    cfg = {"metadata": {"dataset_name": "toy", "dataset_status": "synthetic-smoke-test",
                        "run_purpose": "verification"},
           "model": {"backbone": "tiny", "pretrained": False}}
    results = {"baseline": {"accuracy": 0.5, "macro_f1": 0.3},
               "ssl_adapted": {"accuracy": 0.6, "macro_f1": 0.4}}
    summary = build_run_summary(cfg, results)
    assert "synthetic-smoke-test" in summary
    assert "not research evidence" in summary
    assert "+0.1000" in summary
