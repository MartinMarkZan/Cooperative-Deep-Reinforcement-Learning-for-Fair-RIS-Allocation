#!/bin/bash

# Usage:
#   submit.sh train
#   submit.sh test
#   submit.sh both

# Configuration
CLUSTER_USER="martin.zan"
CLUSTER_HOST="cluster.datalab.tuwien.ac.at"
REMOTE_DIR="/home/$CLUSTER_USER"

# --- Argument parsing ---
MODE=$1

if [ -z "$MODE" ]; then
  echo "Error: Please specify 'train', 'test', or 'both'"
  exit 1
fi

# Sync code
echo "Syncing code to cluster..."
ssh $CLUSTER_USER@$CLUSTER_HOST "rm -rf $REMOTE_DIR/cli $REMOTE_DIR/src $REMOTE_DIR/results"
scp -rq ./cli ./src ./results $CLUSTER_USER@$CLUSTER_HOST:$REMOTE_DIR/
# scp -rq ./src $CLUSTER_USER@$CLUSTER_HOST:$REMOTE_DIR/

# Submit the job
echo "Submitting job..."
ssh $CLUSTER_USER@$CLUSTER_HOST "cd $REMOTE_DIR && sbatch src/run_job.sh $MODE"
