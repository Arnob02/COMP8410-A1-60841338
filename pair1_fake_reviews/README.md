## Whats in this folder

1. pair1_fake_reviews.ipynb: the notebook. It holds all the code for this pair.
2. data/YelpChi.zip: the YelpChi dataset. The notebook unpacks it the first time it runs.
3. results/: the saved results of the final run, the trained models in results/models, and the charts in results/figures.
4. requirements.txt: the Python packages


## How to run it

Keep the notebook, data and results together in this folder. The notebook reads and writes only inside its own folder. 

1. Install Python 3.11
2. Install PyTorch - With an NVIDIA GPU, which the final run used: `pip install torch --index-url https://download.pytorch.org/whl/cu126`
   Without a GPU: `pip install torch`
3. Install the other packages from this folder: `pip install -r requirements.txt`
4. Open `pair1_fake_reviews.ipynb` in VS Code or Jupyter. When asked for a kernel choose the Python environment from steps 1 to 3. The notebook was saved with a kernel called "COMP8410 venv" which only exists on the author's PC.
5. Run all cells.

By default `RUN_TRAINING = False`. In this mode the notebook loads the saved results and models, redraws the charts into `results/figures`, and runs the input-to-output demo in Section 8. It needs no GPU and no internet, and took under a minute in testing.

