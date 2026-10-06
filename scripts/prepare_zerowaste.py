#!/usr/bin/env python3
import argparse, csv, json
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(description="Convert COCO annotations to dominant-class image labels")
parser.add_argument("--coco", required=True); parser.add_argument("--images", required=True); parser.add_argument("--output", required=True)
args = parser.parse_args()
data = json.loads(Path(args.coco).read_text())
categories = {c["id"]: c["name"] for c in data["categories"]}
scores = defaultdict(lambda: defaultdict(float))
for ann in data["annotations"]:
    scores[ann["image_id"]][ann["category_id"]] += float(ann.get("area", 1.0))
output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["path", "label"]); writer.writeheader()
    for image in data["images"]:
        if scores[image["id"]]:
            category = max(scores[image["id"]], key=scores[image["id"]].get)
            writer.writerow({"path": str(Path(args.images) / image["file_name"]), "label": categories[category]})

