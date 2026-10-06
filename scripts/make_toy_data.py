#!/usr/bin/env python3
import argparse, csv, random
from pathlib import Path
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser(); parser.add_argument("--output", default="data/toy"); args = parser.parse_args()
root = Path(args.output); random.seed(7)
for split, count in [("train", 24), ("val", 8), ("test", 8)]:
    rows=[]; folder=root/split; folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        label = "blue" if i % 2 == 0 else "orange"; base=(30,80,210) if label=="blue" else (225,100,25)
        image=Image.new("RGB",(40,40),base); draw=ImageDraw.Draw(image); draw.rectangle((8,8,31,31),outline=(255,255,255),width=2)
        path=folder/f"{i:03d}.png"; image.save(path); rows.append({"path":str(path),"label":label})
    with (root/f"{split}.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=["path","label"]); w.writeheader(); w.writerows(rows)
unlabeled=root/"unlabeled"; unlabeled.mkdir(parents=True,exist_ok=True)
for i in range(32):
    color=(random.randrange(256),random.randrange(256),random.randrange(256)); Image.new("RGB",(40,40),color).save(unlabeled/f"{i:03d}.png")

