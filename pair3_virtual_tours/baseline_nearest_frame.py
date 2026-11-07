# Scores the nearest training frame against each test view. Runs inside the container.
import argparse, json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

parser = argparse.ArgumentParser()
parser.add_argument("scene", type=Path)
parser.add_argument("frames", choices=["full", "reduced"])
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
meta = json.loads((args.scene / f"{args.frames}.json").read_text())
by_name = {f["file_path"]: f for f in meta["frames"]}
train, test = [by_name[n] for n in meta["train_filenames"]], [by_name[n] for n in meta["test_filenames"]]


def pose(frame):
    m = np.array(frame["transform_matrix"])
    return m[:3, 3], -m[:3, 2]  # Camera position, and viewing direction (it looks along the negative third column).


def image(frame):
    path = args.scene / "images_2" / Path(frame["file_path"]).name  # Half size, like the models.
    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(device)


positions = np.array([pose(f)[0] for f in train])
scale = float(np.median(np.linalg.norm(np.diff(positions, axis=0), axis=1))) or 1.0  # Typical step between neighbouring frames.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)  # The same metric settings as Nerfstudio.
lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
scores = {"psnr": [], "ssim": [], "lpips": []}
with torch.no_grad():
    for frame in test:
        position, direction = pose(frame)
        best, best_score = None, float("inf")
        for other in train:  # Closest position plus angle, with one typical step equal to 10 degrees.
            p, d = pose(other)
            score = np.linalg.norm(position - p) / scale + np.degrees(np.arccos(np.clip(direction @ d, -1.0, 1.0))) / 10.0
            if score < best_score:
                best, best_score = other, score
        truth, shown = image(frame), image(best)
        scores["psnr"].append(float(psnr(truth, shown)))
        scores["ssim"].append(float(structural_similarity_index_measure(truth, shown)))
        scores["lpips"].append(float(lpips(truth, shown)))
result = {key: float(np.mean(values)) for key, values in scores.items()}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2))
print(result)
