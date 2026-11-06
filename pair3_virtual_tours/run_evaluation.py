"""
run_evaluation.py - Trains and tests each 3D model on each scene. Run it in the container.

Full run:             python run_evaluation.py
Short setup check:    python run_evaluation.py --quick

The two platforms are compared on the same scenes, images, test views and training
length. A full run trains 8 models, which takes hours, so leave it running. After each
model it saves a result, and if it stops, the same command continues from there.
A --quick run saves to results_quick, and those numbers must not be reported.
"""

import argparse
import csv
import json
import subprocess
import threading
import time
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROCESSED = HERE / "processed"

# Half-size images keep both models within an 8 GB GPU. Both use the same size.
DOWNSCALE = 2

# A fixed seed makes a run repeatable, as far as GPU calculations allow.
SEED = 42

# Options that apply to one model only.
MODEL_OPTIONS = {
    "nerfacto": [
        # By default, Nerfacto adjusts training cameras as it learns but can't adjust
        # test cameras, which lowers its test scores unfairly. Splatfacto's default is
        # already off, so this makes the comparison equal.
        "--pipeline.model.camera-optimizer.mode", "off",
    ],
    "splatfacto": [
        # Keeps the training images in normal memory, which leaves more GPU memory.
        "--pipeline.datamanager.cache-images", "cpu",
    ],
}

# Options for the data reader, which must come last on the command line.
DATA_OPTIONS = {
    "nerfacto": [],
    # Splatfacto starts from COLMAP's 3D points. This flag is stated directly, because
    # naming nerfstudio-data on the command line resets the method's own setting.
    "splatfacto": ["--load-3D-points", "True"],
}

COLUMNS = [
    "scene", "model", "frames", "status", "train_frames", "test_frames", "iterations",
    "psnr", "ssim", "lpips", "render_fps", "train_minutes", "peak_gpu_gb", "storage_mb",
]


class MemoryWatcher(threading.Thread):
    """Reads GPU memory every 2 seconds during training and keeps the highest value."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak_mb = None
        self.finished = threading.Event()

    def run(self):
        while not self.finished.wait(2):
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            )
            try:
                used = float(result.stdout.strip().splitlines()[0])
            except (ValueError, IndexError):
                continue  # A missed reading doesn't matter, so try again next time.
            self.peak_mb = max(self.peak_mb or 0, used)


def folder_size_mb(folder: Path) -> float:
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file()) / 1024**2


def train_command(model: str, data_file: Path, experiment: str, outputs: Path, iterations: int) -> list[str]:
    """Build the ns-train command. The fixed folder names let later steps find the model."""
    return [
        "ns-train", model,
        "--data", str(data_file),
        "--output-dir", str(outputs),
        "--experiment-name", experiment,
        "--timestamp", "run",
        "--max-num-iterations", str(iterations),
        "--machine.seed", str(SEED),
        "--vis", "tensorboard",  # No browser window, so the run finishes by itself.
        # Skips slow progress reports, because the final test below gives the scores.
        "--steps-per-eval-all-images", "1000000",
        *MODEL_OPTIONS[model],
        "nerfstudio-data",
        "--downscale-factor", str(DOWNSCALE),
        *DATA_OPTIONS[model],
    ]


def run_logged(command: list[str], log_path: Path) -> bool:
    """Run a command, save its output to a log, and return True if it succeeded."""
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()
        return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode == 0


def run_model(scene: Path, model: str, frames: str, iterations: int, outputs: Path, results: Path) -> dict:
    """Train one model, test it, and return one row of results."""
    experiment = f"{scene.name}_{frames}"
    run_name = f"{scene.name}__{model}__{frames}"
    logs = results / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / f"{run_name}.log"

    data_file = scene / f"{frames}.json"
    meta = json.loads(data_file.read_text())
    row = {
        "scene": scene.name, "model": model, "frames": frames, "status": "failed",
        "train_frames": len(meta["train_filenames"]), "test_frames": len(meta["test_filenames"]),
        "iterations": iterations,
    }

    # 1. Train, watching GPU memory and time.
    print(f"Training {run_name}. Follow progress in {log_path}")
    watcher = MemoryWatcher()
    watcher.start()
    started = time.perf_counter()
    trained = run_logged(train_command(model, data_file, experiment, outputs, iterations), log_path)
    row["train_minutes"] = round((time.perf_counter() - started) / 60, 2)
    watcher.finished.set()
    watcher.join()
    row["peak_gpu_gb"] = round(watcher.peak_mb / 1024, 2) if watcher.peak_mb else None
    if not trained:
        print(f"FAILED: {run_name}. See the end of {log_path}")
        return row

    run_dir = outputs / experiment / model / "run"
    config = run_dir / "config.yml"
    row["storage_mb"] = round(folder_size_mb(run_dir / "nerfstudio_models"), 1)

    # 2. Score the held-out test views, and save the images for visual checks.
    eval_file = results / "evals" / f"{run_name}.json"
    eval_file.parent.mkdir(exist_ok=True)
    if not run_logged(
        ["ns-eval", "--load-config", str(config), "--output-path", str(eval_file), "--render-output-path", str(results / "renders" / run_name)],
        log_path,
    ):
        print(f"FAILED during testing: {run_name}. See the end of {log_path}")
        return row
    scores = json.loads(eval_file.read_text())["results"]
    row.update(psnr=scores["psnr"], ssim=scores["ssim"], lpips=scores["lpips"], status="done")

    # 3. Measure drawing speed. A failure here keeps the scores and leaves this blank.
    speed = subprocess.run(["python", str(HERE / "measure_speed.py"), str(config)], capture_output=True, text=True)
    try:
        row["render_fps"] = json.loads(speed.stdout.strip().splitlines()[-1])["render_fps"]
    except (ValueError, IndexError, KeyError):
        print(f"Couldn't measure drawing speed for {run_name}")
        with log_path.open("a", encoding="utf-8") as log:
            log.write(speed.stdout + speed.stderr)

    # 4. Make a short walkthrough video for the presentation, from the full data only.
    if frames == "full":
        video = results / "videos" / f"{run_name}.mp4"
        video.parent.mkdir(exist_ok=True)
        run_logged(
            ["ns-render", "interpolate", "--load-config", str(config), "--output-path", str(video), "--pose-source", "eval", "--interpolation-steps", "6", "--frame-rate", "24"],
            log_path,
        )
    return row


def run_baseline(scene: Path, frames: str, results: Path) -> dict:
    """Score the nearest-frame baseline, which needs no training."""
    run_name = f"{scene.name}__nearest-frame__{frames}"
    output = results / "evals" / f"{run_name}.json"
    finished = subprocess.run(["python", str(HERE / "baseline_nearest_frame.py"), str(scene), frames, "--output", str(output)])
    meta = json.loads((scene / f"{frames}.json").read_text())
    row = {
        "scene": scene.name, "model": "nearest-frame", "frames": frames, "status": "failed",
        "train_frames": len(meta["train_filenames"]), "test_frames": len(meta["test_filenames"]),
    }
    if finished.returncode == 0:
        row.update(json.loads(output.read_text()), status="done")
    return row


def save_table(rows: list[dict], path: Path) -> None:
    """Rewrite the results table, which is safe to do after every run."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pair 3 virtual tour evaluation")
    # Both models use Nerfstudio's standard training length.
    parser.add_argument("--iterations", type=int, default=30000, help="training steps per model")
    parser.add_argument("--quick", action="store_true", help="short setup check, not for reporting")
    args = parser.parse_args()

    scenes = sorted(path for path in PROCESSED.iterdir() if (path / "full.json").exists())
    if not scenes:
        raise SystemExit("No prepared scenes found. Run prepare_scenes.py first.")

    frame_levels = ["full", "reduced"]
    iterations = args.iterations
    if args.quick:
        scenes, frame_levels, iterations = scenes[:1], ["full"], 300

    results = HERE / ("results_quick" if args.quick else "results")
    outputs = HERE / ("outputs_quick" if args.quick else "outputs")
    results.mkdir(exist_ok=True)
    table = results / "metrics.csv"

    # Earlier finished runs are kept. Failed runs are tried again.
    rows = []
    if table.exists():
        with table.open(encoding="utf-8") as file:
            rows = [row for row in csv.DictReader(file) if row["status"] == "done"]
    finished = {(row["scene"], row["model"], row["frames"]) for row in rows}

    for scene in scenes:
        for frames in frame_levels:
            if (scene.name, "nearest-frame", frames) not in finished:
                (results / "evals").mkdir(exist_ok=True)
                rows.append(run_baseline(scene, frames, results))
                save_table(rows, table)
            for model in MODEL_OPTIONS:
                if (scene.name, model, frames) in finished:
                    print(f"Already finished: {scene.name} {model} {frames}")
                    continue
                rows.append(run_model(scene, model, frames, iterations, outputs, results))
                save_table(rows, table)

    # Record the exact software versions for the report.
    # The version comes from Python itself, because the container has no pip command.
    try:
        nerfstudio_version = metadata.version("nerfstudio")
    except metadata.PackageNotFoundError:
        nerfstudio_version = None  # A missing version shouldn't stop a finished run.
    colmap = subprocess.run(["colmap", "-h"], capture_output=True, text=True).stdout.splitlines()[:2]
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    environment = {
        "nerfstudio": nerfstudio_version,
        "colmap": colmap,
        "gpu": gpu,
        "settings": {"iterations": iterations, "downscale": DOWNSCALE, "seed": SEED, "model_options": MODEL_OPTIONS, "data_options": DATA_OPTIONS},
    }
    (results / "environment.json").write_text(json.dumps(environment, indent=2))

    failed = [row for row in rows if row["status"] != "done"]
    print(f"\nFinished. Results are in {table}. Failed runs: {len(failed)}")


if __name__ == "__main__":
    main()
