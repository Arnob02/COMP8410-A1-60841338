## How to run it

1. Install Python 3.11
2. nstall PyTorch - With an NVIDIA GPU, which the final run used: `pip install torch --index-url https://download.pytorch.org/whl/cu126`
   Without a GPU: `pip install torch`
3. Install the other packages from this folder: `pip install -r requirements.txt`
4. Open `pair1_fake_reviews.ipynb` in VS Code or Jupyter. When asked for a kernel choose
   the Python environment from steps 1 to 3. The notebook was saved with a kernel called
   "COMP8410 venv" which only exists on the author's PC.
5. Run all cells.

By default `RUN_TRAINING = False`. In this mode the notebook loads the saved results and models,
redraws the charts into `results/figures`, and runs the input-to-output demo in Section 8. It
needs no GPU and no internet, and took under a minute in testing.