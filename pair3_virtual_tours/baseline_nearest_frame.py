"""
baseline_nearest_frame.py - The simple baseline for Pair 3.

A hotel could skip 3D entirely and show the closest photo it already has. For each test
view, this file finds the nearest training frame and scores it against the real view,
using the same image metrics Nerfstudio uses. If a 3D model can't beat this, the extra
cost of 3D isn't justified.

Run it inside the container:   python baseline_nearest_frame.py processed/place1 full
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# Each model's images are half size. The baseline must use the same size.
DOWNSCALE = 2


def camera_position_and_direction(frame: dict) -> tuple[np.ndarray, np.ndarray]:
    """Read one camera's location and viewing direction from its 4x4 matrix."""
    matrix = np.array(frame["transform_matrix"])
    # In this format, the camera looks along the negative third column.
    return matrix[:3, 3], -matrix[:3, 2]


def nearest_train_frame(test_frame: dict, train_frames: list[dict], position_scale: float) -> dict:
    """
    Pick the training frame with the closest position and viewing direction.

    Position and angle have different units, so each is scaled before they are added.
    One typical step between frames counts the same as a 10 degree turn.
    """
    position, direction = camera_position_and_direction(test_frame)
    best_frame, best_score = None, float("inf")
    for frame in train_frames:
        other_position, other_direction = camera_position_and_direction(frame)
        distance = np.linalg.norm(position - other_position) / position_scale
        angle = np.degrees(np.arccos(np.clip(direction @ other_direction, -1.0, 1.0))) / 10.0
        if distance + angle < best_score:
            best_frame, best_score = frame, distance + angle
    return best_frame


def load_image(scene: Path, frame: dict, device: torch.device) -> torch.Tensor:
    """Load a half-size image as a [1, 3, height, width] tensor with values from 0 to 1."""
    path = scene / f"images_{DOWNSCALE}" / Path(frame["file_path"]).name
    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1)[None].to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nearest-frame baseline")
    parser.add_argument("scene", type=Path, help="processed scene folder")
    parser.add_argument("frames", choices=["full", "reduced"], help="which training set to use")
    parser.add_argument("--output", type=Path, required=True, help="result JSON file")
    args = parser.parse_args()

    meta = json.loads((args.scene / f"{args.frames}.json").read_text())
    by_name = {frame["file_path"]: frame for frame in meta["frames"]}
    train = [by_name[name] for name in meta["train_filenames"]]
    test = [by_name[name] for name in meta["test_filenames"]]

    # The typical gap between neighbouring training cameras sets the position scale.
    positions = np.array([camera_position_and_direction(frame)[0] for frame in train])
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    position_scale = float(np.median(steps)) or 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # These match the metric settings in Nerfstudio's model code.
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    scores = {"psnr": [], "ssim": [], "lpips": []}
    with torch.no_grad():
        for frame in test:
            truth = load_image(args.scene, frame, device)
            shown = load_image(args.scene, nearest_train_frame(frame, train, position_scale), device)
            scores["psnr"].append(float(psnr(truth, shown)))
            scores["ssim"].append(float(structural_similarity_index_measure(truth, shown)))
            scores["lpips"].append(float(lpips(truth, shown)))

    result = {key: float(np.mean(values)) for key, values in scores.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
