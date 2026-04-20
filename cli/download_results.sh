#!/bin/bash

# Configuration
CLUSTER_USER="martin.zan"
CLUSTER_HOST="cluster.datalab.tuwien.ac.at"
REMOTE_DIR="/home/$CLUSTER_USER"
LOCAL_RESULTS_DIR="./results"

# --- Make sure local results directory exists ---
mkdir -p "$LOCAL_RESULTS_DIR"

# --- Download the results folder ---
echo "Downloading results from cluster..."
scp -rq $CLUSTER_USER@$CLUSTER_HOST:$REMOTE_DIR/results/* $LOCAL_RESULTS_DIR/

# --- Done ---
echo "Results successfully downloaded to: $LOCAL_RESULTS_DIR"
