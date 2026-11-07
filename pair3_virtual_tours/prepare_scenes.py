
import json, shutil, struct, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIDEOS, PROCESSED = HERE / "videos", HERE / "processed"
FRAMES, TEST_EVERY, LONG_SIDE, MIN_SHARE = 150, 8, 1920, 0.8
FIRST, RETRY, RECORD = "sequential", "exhaustive", "matching_used.txt"  


def resize(video):
    out = PROCESSED / "_resized" / f"{video.stem}.mp4"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        size = f"scale='if(gt(iw,ih),{LONG_SIDE},-2)':'if(gt(iw,ih),-2,{LONG_SIDE})'"  
        partial = out.with_suffix(".partial.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vf", size, "-crf", "18", "-preset", "veryfast", "-an", str(partial)], check=True)
        partial.rename(out)  
    return out


def colmap(video, output, method):
    subprocess.run(["ns-process-data", "video", "--data", str(resize(video)), "--output-dir", str(output),
                    "--num-frames-target", str(FRAMES), "--matching-method", method, "--no-gpu"], check=True)


def placed(model):
    return struct.unpack("<Q", (model / "images.bin").read_bytes()[:8])[0]  


def frames_in(output):
    file = output / "transforms.json"
    return len(json.loads(file.read_text())["frames"]) if file.exists() else 0


def share(output):
    return frames_in(output) / len(list((output / "images").glob("*")))  


def use_largest_model(output):
    sparse = output / "colmap" / "sparse"
    models = sorted(f for f in sparse.iterdir() if (f / "images.bin").exists()) if sparse.exists() else []
    model = max(models, key=placed, default=None)
    if model is None or placed(model) == frames_in(output):
        return
    print(f"{output.name}: using {model.name}, which placed {placed(model)} frames")
    refine = subprocess.run(["colmap", "bundle_adjuster", "--input_path", str(model), "--output_path", str(model),
                             "--BundleAdjustment.refine_principal_point", "1"], capture_output=True, text=True)
    if refine.returncode != 0:
        raise SystemExit(refine.stderr[-2000:])
    subprocess.run(["ns-process-data", "images", "--data", str(output / "images"), "--output-dir", str(output), "--skip-colmap",
                    "--skip-image-processing", "--colmap-model-path", str(model.relative_to(output))], check=True)


def prepare(video, output):
    if (output / "transforms.json").exists():
        print(f"{video.stem}: COLMAP has already run")
    else:
        if output.exists():
            shutil.rmtree(output) 
        colmap(video, output, FIRST)
    use_largest_model(output)
    if share(output) < MIN_SHARE and not (output / RECORD).exists():
        print(f"{video.stem}: {share(output):.0%} of frames placed, so trying {RETRY} matching")
        retry = output.with_name(f"{output.name}_retry")
        if retry.exists():
            shutil.rmtree(retry)
        colmap(video, retry, RETRY)
        use_largest_model(retry)
        better = share(retry) > share(output)  
        shutil.rmtree(output if better else retry)
        if better:
            retry.rename(output)
        (output / RECORD).write_text(RETRY if better else FIRST)
    if frames_in(output) == 0:
        raise SystemExit(f"COLMAP couldn't place any frames in {video.stem}. Film it again more slowly.")


videos = sorted(p for p in VIDEOS.iterdir() if p.suffix.lower() in {".mp4", ".mov"})
if not videos:
    raise SystemExit(f"No videos found. Put your .mp4 or .mov files in {VIDEOS}")
summary = []
for video in videos:
    output = PROCESSED / video.stem
    prepare(video, output)
    meta = json.loads((output / "transforms.json").read_text())
    names = sorted(f["file_path"] for f in meta["frames"])
    test = names[::TEST_EVERY]
    train = [n for n in names if n not in test]
    for name, frames in [("full", train), ("reduced", train[::2])]: 
        (output / f"{name}.json").write_text(json.dumps({**meta, "train_filenames": frames, "val_filenames": test, "test_filenames": test}, indent=2))
    record = output / RECORD
    row = {"scene": video.stem, "frames_extracted": len(list((output / "images").glob("*"))), "frames_matched": len(names),
           "matched_share": round(share(output), 3), "matching": record.read_text() if record.exists() else FIRST,
           "train_frames_full": len(train), "train_frames_reduced": len(train[::2]), "test_frames": len(test)}
    summary.append(row)
    print(row)
    if share(output) < MIN_SHARE:
        print(f"WARNING: only {share(output):.0%} of frames matched in {video.stem}. Film it again more slowly.")
(PROCESSED / "scene_summary.json").write_text(json.dumps(summary, indent=2))
