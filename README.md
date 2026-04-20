# Fair RIS Assignment
This project implements and evaluates **reinforcement learning (RL)** and heuristic methods for fair **Reconfigurable Intelligent Surface (RIS)** allocation in wireless networks.

## Core modules:
- `cli/train.py` – train and save RL models.
- `cli/test.py` – run experiments and compare RL vs. heuristic baselines.
- `cli/evaluate.py` – aggregate results provided by testing.
- `cli/plots.ipynb` – visualize performance metrics.
- `src/config.py` – configurate the environment, training and testing.
- `src/` – environment, metrics, and utility functions (e.g., plotting).

## Running the Code
Python version 3.12.7 is required.
1. Install the packages with: `pip install -r requirements.txt`
2. Run the modules as Python packages: `python -m cli.train` and `python -m cli.evaluate`

## Running with Apptainer container
1. Build container with: `apptainer build my-container.sif image.def`
2. Activate container with: `apptainer shell -e my-container.sif`
3. Then use the regular commands to run the code.