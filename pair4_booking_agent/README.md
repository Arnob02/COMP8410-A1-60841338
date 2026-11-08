## Whats in this folder

1. pair4_booking_agent.ipynb: the notebook. It contains the mock booking system, the 30 test tasks, the marking code, the agents and the rule-based baseline.
2. results/: every saved agent run in results/traces.jsonl, the marked results and summary tables. The model settings in results/environment.json and the charts in results/figures.
3. requirements.txt: the Python packages.

Keep the notebook and results together in this folder. The notebook reads and writes only inside its own folder.

## How to run it

1. Install Python 3.11. 
2. Install the packages from this folder: pip install -r requirements.txt
3. Open pair4_booking_agent.ipynb in VS Code or Jupyter. When asked for a kernel, choose the Python environment from steps 1 and 2.
4. Restart the kernel then run all cells.

By default RUN_EVALUATION = False and RUN_DEMO = False. In this mode the notebook tests its own marking code, marks all 270 saved runs again, rebuilds the result tables and charts and replays two saved runs per agent in the demo in Section 8. The rule-based baseline runs live because it needs no model. It needs no GPU no Ollama and no internet and the fictional booking data is built inside the notebook.

