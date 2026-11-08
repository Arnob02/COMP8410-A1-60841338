## Whats in this folder

1. pair2_destination_video.ipynb: the notebook.
2. workflows/wan21.json and workflows/wan22.json: the exact ComfyUI workflows used for each model.
3. results/: the 20 generated clips in results/clips, their scores, the blind rating sheet and key in results/rating, the summary tables, the charts in results/figures, and the demo clips in results/demo.
4. requirements.txt: the Python packages.

## How to run it

1. Install Python 3.11. 
2. Install the packages from this folder: pip install -r requirements.txt. A CPU-only PyTorch is enough.
3. Open pair2_destination_video.ipynb in VS Code or Jupyter. When asked for a kernel, choose the Python environment from steps 1 and 2.
4. Restart the kernel then run all cells.

By default RUN_GENERATION, RUN_SCORING and RUN_DEMO are all False. In this mode the notebook uses the saved clips, scores and ratings, rebuilds the result tables and charts and shows the saved demo clips in Section 7. It needs no GPU, no ComfyUI and no model files. The first cell runs pip install which needs internet only if a package is missing.