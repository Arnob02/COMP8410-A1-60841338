## What is in this folder

1. pair3_virtual_tours.ipynb: the notebook.
2. prepare_scenes.py, run_evaluation.py, baseline_nearest_frame.py and measure_speed.py: the scripts that run inside the Nerfstudio container. Section 1 of the notebook writes them so running the notebook recreates them.
3. results/: the scores in results/metrics.csv, the test views drawn by each model in results/renders, the blind rating images, sheet and key in results/rating, the charts in results/figures, and the walkthrough videos in results/videos.
4. requirements.txt: the Python packages.

Keep the notebook and results together in this folder. The notebook reads and writes only inside its own folder.

## How to run it

1. Install Python 3.11. 
2. Install the packages from this folder: pip install -r requirements.txt
3. Open pair3_virtual_tours.ipynb in VS Code or Jupyter. When asked for a kernel, choose the Python environment from steps 1 and 2.
4. Restart the kernel then run all cells.

The notebook sees that all 12 runs are saved in results/metrics.csv and skips the experiment. It then shows the results, reruns the colour check from the saved test views, redraws the charts, summarises the blind ratings and shows the demo. It needs no GPU no Docker and no internet.

If the processed folder isn't included, Section 3 says it can't show the scene table. The other results are unaffected.