#!/usr/bin/env bash
set -euo pipefail

# RCV1 is isolated from the legacy WOS runner.  In particular, it prepares an
# official taxonomy and always writes rankings keyed by text_idx.
data=RCV1-103-H3
model=HGCLR
START_FOLD=${1:?usage: $0 START_FOLD END_FOLD prepare,preprocess,fit,predict,eval}
END_FOLD=${2:?usage: $0 START_FOLD END_FOLD prepare,preprocess,fit,predict,eval}
TASKS_ARG=${3:?usage: $0 START_FOLD END_FOLD prepare,preprocess,fit,predict,eval}

should_run() {
    [[ ",$TASKS_ARG," == *",$1,"* ]]
}

mkdir -p resource/time

if should_run "prepare"; then
    python source/helper/rcv1_preparation.py --data-dir "resource/dataset/$data"
fi

if should_run "preprocess"; then
    # Preprocessing writes shared dataset artifacts (vocab, token binaries and
    # slot.pt), so it must run once before independent fold Pods start.
    folds=$(seq -s, "$START_FOLD" "$END_FOLD")
    time_start=$(date '+%Y-%m-%d %H:%M:%S')
    python main.py \
      tasks=[preprocess] \
      data=$data \
      data.folds=[$folds] \
      model=$model
    time_end=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$time_start,$time_end" > "resource/time/preprocess_${data}_${START_FOLD}-${END_FOLD}.tmr"
fi

if should_run "fit"; then
    for fold_idx in $(seq "$START_FOLD" "$END_FOLD"); do
        time_start=$(date '+%Y-%m-%d %H:%M:%S')
        python train.py \
          --data "$data" \
          --fold "$fold_idx" \
          --epochs 16 \
          --early-stop 3 \
          --batch 12 \
          --lamb 0.05 \
          --thre 0.02
        time_end=$(date '+%Y-%m-%d %H:%M:%S')
        echo "$time_start,$time_end" > "resource/time/fit_${data}_${fold_idx}.tmr"
    done
fi

if should_run "predict"; then
    for fold_idx in $(seq "$START_FOLD" "$END_FOLD"); do
        time_start=$(date '+%Y-%m-%d %H:%M:%S')
        python test.py \
          --data "$data" \
          --fold "$fold_idx" \
          --ranking-id text_idx
        time_end=$(date '+%Y-%m-%d %H:%M:%S')
        echo "$time_start,$time_end" > "resource/time/predict_${data}_${fold_idx}.tmr"
    done
fi

if should_run "eval"; then
    for fold_idx in $(seq "$START_FOLD" "$END_FOLD"); do
        time_start=$(date '+%Y-%m-%d %H:%M:%S')
        python main.py \
          tasks=[eval] \
          data=$data \
          data.folds=[$fold_idx] \
          model=$model
        time_end=$(date '+%Y-%m-%d %H:%M:%S')
        echo "$time_start,$time_end" > "resource/time/eval_${data}_${fold_idx}.tmr"
    done
fi
