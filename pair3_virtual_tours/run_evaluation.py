# Trains and tests each model on each scene. Runs inside the container. --quick is a short check, not for reporting.
import argparse, csv, json, subprocess, threading, time
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROCESSED = HERE / "processed"
DOWNSCALE, SEED = 2, 42  # Half-size images keep both models within an 8 GB GPU.
MODEL_OPTIONS = {"nerfacto": ["--pipeline.model.camera-optimizer.mode", "off"],
                 "splatfacto": ["--pipeline.datamanager.cache-images", "cpu"]}  # Keeps images in RAM, saving GPU memory.
DATA_OPTIONS = {"nerfacto": [], "splatfacto": ["--load-3D-points", "True"]}  # Splatfacto starts from COLMAP's points.
COLUMNS = ["scene", "model", "frames", "status", "train_frames", "test_frames", "iterations", "psnr", "ssim", "lpips",
           "render_fps", "train_minutes", "peak_gpu_gb", "storage_mb"]


def watch_gpu(peak, stop):
    """Every 2 seconds, keep the highest GPU memory in use (MB)."""
    while not stop.wait(2):
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        try:
            peak[0] = max(peak[0] or 0, float(out.strip().splitlines()[0]))
        except (ValueError, IndexError):
            pass  # A missed reading doesn't matter.


def run_logged(command, log_path):
    """Run a command, append its output to a log, and return True if it succeeded."""
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()
        return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode == 0


def run_model(scene, model, frames, iterations, outputs, results):
    """Train one model, test it, time its drawing, and return one row of results."""
    experiment, name = f"{scene.name}_{frames}", f"{scene.name}__{model}__{frames}"
    (results / "logs").mkdir(exist_ok=True)
    log = results / "logs" / f"{name}.log"
    data_file = scene / f"{frames}.json"
    meta = json.loads(data_file.read_text())
    row = {"scene": scene.name, "model": model, "frames": frames, "status": "failed", "train_frames": len(meta["train_filenames"]),
           "test_frames": len(meta["test_filenames"]), "iterations": iterations}

    print(f"Training {name}. Follow progress in {log}")
    peak, stop = [None], threading.Event()
    watcher = threading.Thread(target=watch_gpu, args=(peak, stop), daemon=True)
    watcher.start()
    started = time.perf_counter()
    trained = run_logged(["ns-train", model, "--data", str(data_file), "--output-dir", str(outputs), "--experiment-name", experiment,
                          "--timestamp", "run", "--max-num-iterations", str(iterations), "--machine.seed", str(SEED),
                          "--vis", "tensorboard",  # No browser window, so the run finishes by itself.
                          "--steps-per-eval-all-images", "1000000",  # Skips slow progress checks; ns-eval gives the scores.
                          *MODEL_OPTIONS[model], "nerfstudio-data", "--downscale-factor", str(DOWNSCALE), *DATA_OPTIONS[model]], log)
    row["train_minutes"] = round((time.perf_counter() - started) / 60, 2)
    stop.set()
    watcher.join()
    row["peak_gpu_gb"] = round(peak[0] / 1024, 2) if peak[0] else None
    if not trained:
        print(f"FAILED: {name}. See the end of {log}")
        return row

    run_dir = outputs / experiment / model / "run"
    config = run_dir / "config.yml"
    row["storage_mb"] = round(sum(p.stat().st_size for p in (run_dir / "nerfstudio_models").rglob("*") if p.is_file()) / 1024**2, 1)
    eval_file = results / "evals" / f"{name}.json"
    eval_file.parent.mkdir(exist_ok=True)
    if not run_logged(["ns-eval", "--load-config", str(config), "--output-path", str(eval_file), "--render-output-path", str(results / "renders" / name)], log):
        print(f"FAILED during testing: {name}. See the end of {log}")
        return row
    scores = json.loads(eval_file.read_text())["results"]
    row.update(psnr=scores["psnr"], ssim=scores["ssim"], lpips=scores["lpips"], status="done")

    speed = subprocess.run(["python", str(HERE / "measure_speed.py"), str(config)], capture_output=True, text=True)
    try:
        row["render_fps"] = json.loads(speed.stdout.strip().splitlines()[-1])["render_fps"]
    except (ValueError, IndexError, KeyError):  # A failure here keeps the scores.
        print(f"Couldn't measure drawing speed for {name}")
        with log.open("a", encoding="utf-8") as file:
            file.write(speed.stdout + speed.stderr)
    if frames == "full":  # A walkthrough video for the presentation.
        video = results / "videos" / f"{name}.mp4"
        video.parent.mkdir(exist_ok=True)
        run_logged(["ns-render", "interpolate", "--load-config", str(config), "--output-path", str(video), "--pose-source", "eval",
                    "--interpolation-steps", "6", "--frame-rate", "24"], log)
    return row


def run_baseline(scene, frames, results):
    """Score the nearest-frame baseline, which needs no training."""
    output = results / "evals" / f"{scene.name}__nearest-frame__{frames}.json"
    finished = subprocess.run(["python", str(HERE / "baseline_nearest_frame.py"), str(scene), frames, "--output", str(output)])
    meta = json.loads((scene / f"{frames}.json").read_text())
    row = {"scene": scene.name, "model": "nearest-frame", "frames": frames, "status": "failed",
           "train_frames": len(meta["train_filenames"]), "test_frames": len(meta["test_filenames"])}
    if finished.returncode == 0:
        row.update(json.loads(output.read_text()), status="done")
    return row


def save_table(rows, path):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int, default=30000)  # Nerfstudio's standard length for both models.
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()
scenes = sorted(p for p in PROCESSED.iterdir() if (p / "full.json").exists())
if not scenes:
    raise SystemExit("No prepared scenes found. Run prepare_scenes.py first.")
levels, iterations = ["full", "reduced"], args.iterations
if args.quick:
    scenes, levels, iterations = scenes[:1], ["full"], 300
results, outputs = HERE / ("results_quick" if args.quick else "results"), HERE / ("outputs_quick" if args.quick else "outputs")
results.mkdir(exist_ok=True)
table = results / "metrics.csv"
rows = [r for r in csv.DictReader(table.open(encoding="utf-8")) if r["status"] == "done"] if table.exists() else []
finished = {(r["scene"], r["model"], r["frames"]) for r in rows}  # Finished runs are kept; failed runs are retried.
for scene in scenes:
    for frames in levels:
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

try:
    version = metadata.version("nerfstudio")  # The container has no pip, so ask Python for the version.
except metadata.PackageNotFoundError:
    version = None
environment = {"nerfstudio": version, "colmap": subprocess.run(["colmap", "-h"], capture_output=True, text=True).stdout.splitlines()[:2],
               "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
               "settings": {"iterations": iterations, "downscale": DOWNSCALE, "seed": SEED, "model_options": MODEL_OPTIONS, "data_options": DATA_OPTIONS}}
(results / "environment.json").write_text(json.dumps(environment, indent=2))
print(f"\nFinished. Results are in {table}. Failed runs: {sum(r['status'] != 'done' for r in rows)}")
