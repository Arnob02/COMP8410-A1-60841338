"""
prepare_scenes.py - Turns each phone video into a dataset. Run it inside the container.

    python prepare_scenes.py

3D models need to know where the camera was for every picture. ns-process-data takes
evenly spaced frames from a video, and COLMAP then works out each camera position from
matching points between frames.

COLMAP can split a video into separate pieces, called models, when it can't link one
part of the video to the next. Nerfstudio always reads the first piece, even if that
piece is tiny. This file picks the largest piece instead. If that still leaves too many
frames unplaced, it tries once more with a slower matching method.

This file then writes two dataset files for each scene. Both test on exactly the same
held-out frames, so their scores can be compared fairly:

    full.json       Trains on all the remaining frames.
    reduced.json    Trains on half of them, which simulates a quicker hotel filming visit.
"""

import json
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
VIDEOS = HERE / "videos"
PROCESSED = HERE / "processed"

# 150 frames keeps processing short and fits an 8 GB GPU.
FRAMES_TARGET = 150

# Every 8th frame is held out for testing, a common choice for this kind of benchmark.
TEST_EVERY = 8

# Phone videos are often 4K. COLMAP ran out of memory on 4K frames, so each video is
# first reduced to this length on its long side. The models use half-size images in
# any case, so this doesn't change what they train on by much.
LONG_SIDE_PIXELS = 1920

# Below this share of matched frames, the camera positions are unreliable.
MINIMUM_MATCHED_SHARE = 0.8

# Sequential matching compares each frame with the frames near it in the video, which is
# quick. Exhaustive matching compares every frame with every other frame, so it can link
# parts of a video that are far apart in time, but it takes several times longer.
FIRST_MATCHING_METHOD = "sequential"
RETRY_MATCHING_METHOD = "exhaustive"

# This file records the method a scene ended up with, once the retry has been tried.
# Its presence also stops the slow retry from running again on the next run.
MATCHING_RECORD = "matching_used.txt"


def resize_video(video: Path) -> Path:
    """Save a smaller copy of the video, keeping its shape, and return the copy's path."""
    resized = PROCESSED / "_resized" / f"{video.stem}.mp4"
    if resized.exists():
        return resized
    resized.parent.mkdir(parents=True, exist_ok=True)
    # The long side becomes LONG_SIDE_PIXELS, whether the video is upright or sideways.
    # -2 keeps the shape and gives an even number, which video files need.
    size = f"scale='if(gt(iw,ih),{LONG_SIDE_PIXELS},-2)':'if(gt(iw,ih),-2,{LONG_SIDE_PIXELS})'"
    temporary = resized.with_suffix(".partial.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vf", size, "-crf", "18", "-preset", "veryfast", "-an", str(temporary)],
        check=True,
    )
    temporary.rename(resized)  # Only a finished file gets the final name.
    return resized


def run_colmap(video: Path, output: Path, method: str) -> None:
    """Extract frames from a video and estimate the camera positions."""
    subprocess.run(
        [
            "ns-process-data", "video",
            "--data", str(resize_video(video)),
            "--output-dir", str(output),
            "--num-frames-target", str(FRAMES_TARGET),
            "--matching-method", method,
            # COLMAP's GPU mode needs a display, which the container doesn't have.
            "--no-gpu",
        ],
        check=True,
    )


def registered_images(model: Path) -> int:
    """Return how many frames a COLMAP model placed."""
    # images.bin starts with this count, stored as an 8-byte whole number.
    with (model / "images.bin").open("rb") as file:
        return struct.unpack("<Q", file.read(8))[0]


def largest_model(output: Path) -> Optional[Path]:
    """Return the COLMAP model that placed the most frames, or None if there are none."""
    sparse = output / "colmap" / "sparse"
    if not sparse.exists():
        return None
    # Sorting by name means a tie always gives the same answer.
    models = sorted(folder for folder in sparse.iterdir() if (folder / "images.bin").exists())
    return max(models, key=registered_images, default=None)


def frames_in_dataset(output: Path) -> int:
    """Return how many frames the dataset file holds, or 0 if there is no such file."""
    transforms = output / "transforms.json"
    return len(json.loads(transforms.read_text())["frames"]) if transforms.exists() else 0


def matched_share(output: Path) -> float:
    """Return the share of extracted frames that have a camera position."""
    # Frames that COLMAP couldn't place are missing from transforms.json.
    return frames_in_dataset(output) / len(list((output / "images").glob("*")))


def use_largest_model(output: Path) -> None:
    """Rebuild the dataset from the largest COLMAP model, if it isn't already in use."""
    model = largest_model(output)
    if model is None or registered_images(model) == frames_in_dataset(output):
        return
    print(f"{output.name}: using {model.name}, which placed {registered_images(model)} frames")

    # Nerfstudio's last COLMAP step sharpens the lens settings of model 0 only, so the
    # chosen model gets the same step here. Otherwise it would be treated differently.
    refine = subprocess.run(
        [
            "colmap", "bundle_adjuster",
            "--input_path", str(model),
            "--output_path", str(model),
            "--BundleAdjustment.refine_principal_point", "1",
        ],
        capture_output=True,  # COLMAP prints a lot, so it only shows if something fails.
        text=True,
    )
    if refine.returncode != 0:
        raise SystemExit(refine.stderr[-2000:])

    # This writes transforms.json and sparse_pc.ply again from the chosen model, without
    # repeating COLMAP. Splatfacto starts from the points in sparse_pc.ply.
    subprocess.run(
        [
            "ns-process-data", "images",
            "--data", str(output / "images"),
            "--output-dir", str(output),
            "--skip-colmap",
            "--skip-image-processing",
            "--colmap-model-path", str(model.relative_to(output)),
        ],
        check=True,
    )


def retry_matching(video: Path, output: Path) -> None:
    """Run COLMAP again with slower matching, and keep whichever result placed more."""
    retry = output.with_name(f"{output.name}_retry")
    # Remove anything left by an earlier retry that stopped partway.
    if retry.exists():
        shutil.rmtree(retry)
    run_colmap(video, retry, RETRY_MATCHING_METHOD)
    use_largest_model(retry)

    # The first result stays in place until the retry has been shown to be better.
    if matched_share(retry) > matched_share(output):
        shutil.rmtree(output)
        retry.rename(output)
        (output / MATCHING_RECORD).write_text(RETRY_MATCHING_METHOD)
    else:
        shutil.rmtree(retry)
        (output / MATCHING_RECORD).write_text(FIRST_MATCHING_METHOD)


def process_video(video: Path, output: Path) -> None:
    """Make the dataset for one video, doing only the steps that aren't already done."""
    if (output / "transforms.json").exists():
        print(f"{video.stem}: COLMAP has already run")
    else:
        # Remove anything left by an earlier run that stopped partway.
        if output.exists():
            shutil.rmtree(output)
        run_colmap(video, output, FIRST_MATCHING_METHOD)

    use_largest_model(output)

    if matched_share(output) < MINIMUM_MATCHED_SHARE and not (output / MATCHING_RECORD).exists():
        print(f"{video.stem}: {matched_share(output):.0%} of frames placed, so trying {RETRY_MATCHING_METHOD} matching")
        retry_matching(video, output)

    if frames_in_dataset(output) == 0:
        raise SystemExit(f"COLMAP couldn't place any frames in {video.stem}. Film it again more slowly.")


def write_split(meta: dict, output: Path, name: str, train: list[str], test: list[str]) -> None:
    """Save a copy of the dataset file that names its own train and test frames."""
    split = dict(meta)
    split["train_filenames"] = train
    split["val_filenames"] = test  # Used for progress checks during training.
    split["test_filenames"] = test  # Used for the final scores.
    (output / f"{name}.json").write_text(json.dumps(split, indent=2))


def main() -> None:
    videos = sorted(path for path in VIDEOS.iterdir() if path.suffix.lower() in {".mp4", ".mov"})
    if not videos:
        raise SystemExit(f"No videos found. Put your .mp4 or .mov files in {VIDEOS}")

    summary = []
    for video in videos:
        output = PROCESSED / video.stem
        process_video(video, output)

        meta = json.loads((output / "transforms.json").read_text())
        names = sorted(frame["file_path"] for frame in meta["frames"])
        share = matched_share(output)
        record = output / MATCHING_RECORD

        test = names[::TEST_EVERY]
        train = [name for name in names if name not in test]
        write_split(meta, output, "full", train, test)
        write_split(meta, output, "reduced", train[::2], test)

        row = {
            "scene": video.stem,
            "frames_extracted": len(list((output / "images").glob("*"))),
            "frames_matched": len(names),
            "matched_share": round(share, 3),
            "matching": record.read_text() if record.exists() else FIRST_MATCHING_METHOD,
            "train_frames_full": len(train),
            "train_frames_reduced": len(train[::2]),
            "test_frames": len(test),
        }
        summary.append(row)
        print(row)
        if share < MINIMUM_MATCHED_SHARE:
            print(f"WARNING: only {share:.0%} of frames matched in {video.stem}. Film it again more slowly.")

    (PROCESSED / "scene_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
