#! /bin/bash
set -euo pipefail

scripts=$(dirname "$0")
base=$(realpath $scripts/..)

models=$base/models
data=$base/data
tools=$base/tools
samples=$base/samples

mkdir -p $samples

num_threads=4

(cd $tools/pytorch-examples/word_language_model &&
    OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/onion_news \
        --words 100 \
        --checkpoint $models/model_dropout_0_0.pt \
        --outf $samples/sample
)

echo "Generated text ($samples/sample):"
cat $samples/sample
