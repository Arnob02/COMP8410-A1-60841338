
import json, sys, time
from pathlib import Path

import torch
from nerfstudio.utils.eval_utils import eval_setup

_, pipeline, _, _ = eval_setup(Path(sys.argv[1]), test_mode="test")
timings = []
with torch.no_grad():
    for index, (camera, _) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
        torch.cuda.synchronize()  
        started = time.perf_counter()
        pipeline.model.get_outputs_for_camera(camera=camera)
        torch.cuda.synchronize()
        if index > 0:  
            timings.append(time.perf_counter() - started)
print(json.dumps({"render_fps": len(timings) / sum(timings)})) 
