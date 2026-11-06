"""
measure_speed.py - Measures how fast a trained model draws new views.

A virtual tour must respond smoothly, so this is a business measure. ns-eval also reports
frames per second, but its timer includes the image metrics and saving files, which hides
the speed of a very fast model. This file times the drawing step alone.

run_evaluation.py calls it. To run it yourself, inside the container:
    python measure_speed.py outputs/place1_full/splatfacto/run/config.yml
"""

import json
import sys
import time
from pathlib import Path

import torch
from nerfstudio.utils.eval_utils import eval_setup


def main() -> None:
    _, pipeline, _, _ = eval_setup(Path(sys.argv[1]), test_mode="test")
    timings = []
    with torch.no_grad():
        for index, (camera, _) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            torch.cuda.synchronize()  # Wait for earlier GPU work, so the timer is exact.
            started = time.perf_counter()
            pipeline.model.get_outputs_for_camera(camera=camera)
            torch.cuda.synchronize()
            # The first view includes one-off setup work, so it isn't counted.
            if index > 0:
                timings.append(time.perf_counter() - started)

    # run_evaluation.py reads this last line.
    print(json.dumps({"render_fps": len(timings) / sum(timings)}))


if __name__ == "__main__":
    main()
