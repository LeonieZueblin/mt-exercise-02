#! /bin/bash

scripts=$(dirname "$0")
base=$(realpath $scripts/..)

models=$base/models
data=$base/data
tools=$base/tools

mkdir -p $models

num_threads=4

SECONDS=0

(cd $tools/pytorch-examples/word_language_model &&
    OMP_NUM_THREADS=$num_threads python main.py --data $data/onion_news --accel\
        --epochs 40 \
        --log-interval 100 \
        --emsize 200 --nhid 200 --dropout 0.5 --tied \
        --save $models/model.pt
)

echo "time taken:"
echo "$SECONDS seconds"
