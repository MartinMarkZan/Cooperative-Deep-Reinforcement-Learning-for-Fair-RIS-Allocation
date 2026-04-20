#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --mail-user=martin.zan@tuwien.ac.at
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Usage:
#   sbatch run_job.sh train
#   sbatch run_job.sh test
#   sbatch run_job.sh both

# --- Argument parsing ---
MODE=$1

if [ -z "$MODE" ]; then
  echo "Error: Please specify 'train', 'test', or 'both'"
  exit 1
fi

# --- Load environment / container ---
# Run commands inside the container non-interactively
# TODO: Error here. In "both" case if multiple models are trained in parallel, 
# then in test all of them will use to config of the last one.
apptainer exec -e my-container.sif bash -c "
  set -e
  case '$MODE' in
    train)
      python -m cli.train
      ;;
    test)
      python -m cli.test
      ;;
    both)
      python -m cli.train
      python -m cli.test
      ;;
    *)
      echo 'Invalid mode: $MODE. Use train, test, or both.'
      exit 1
      ;;
  esac
"
