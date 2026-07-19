#!/bin/bash

# overrides
data=WOS-150-H2
model=HGCLR

START_FOLD=$1
END_FOLD=$2
TASKS_ARG=$3

# Helper function to check if a specific task is in the requested tasks list
# Usage: if should_run "task_name"; then ... fi
should_run() {
    local task_name=$1
    # Check if TASKS_ARG contains the task_name (surrounded by commas to avoid partial matches)
    if [[ ",$TASKS_ARG," == *",$task_name,"* ]]; then
        return 0
    else
        return 1
    fi
}

# preprocess
if should_run "preprocess"; then
  for fold_idx in $(seq $1 $2);
  do
    time_start=$(date '+%Y-%m-%d %H:%M:%S')
    python main.py \
      tasks=[preprocess] \
      data=$data \
      data.folds=[$fold_idx] \
      model=$model
    time_end=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$time_start,$time_end" > resource/time/preprocess_${data}_${fold_idx}.tmr
  done
fi

# fit
if should_run "fit"; then
  for fold_idx in $(seq $1 $2);
  do
    time_start=$(date '+%Y-%m-%d %H:%M:%S')
    python train.py \
    --data WOS-150-H2 \
    --fold $fold_idx \
    --epochs 16 \
    --early-stop 3 \
    --batch 12 \
    --lamb 0.05 \
    --thre 0.02
    echo "$time_start,$time_end" > resource/time/fit_${data}_${fold_idx}.tmr
  done
fi

# predict
if should_run "predict"; then
  for fold_idx in $(seq $1 $2);
  do
    time_start=$(date '+%Y-%m-%d %H:%M:%S')
    python test.py \
    --data WOS-150-H2 \
    --fold $fold_idx
  done
fi

# eval
if should_run "eval"; then
  for fold_idx in $(seq $1 $2);
  do
    time_start=$(date '+%Y-%m-%d %H:%M:%S')
    python main.py \
      tasks=[eval] \
      data=$data \
      data.folds=[$fold_idx] \
      model=$model
    time_end=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$time_start,$time_end" > resource/time/eval_${data}_${fold_idx}.tmr
  done
fi
