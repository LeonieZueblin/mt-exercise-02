#! /bin/bash
set -euo pipefail

scripts=$(dirname "$0")
base=$(realpath "$scripts/..")

models="$base/models"
data="$base/data"
tools="$base/tools"

mkdir -p "$models"

num_threads=4
dropouts=(0.0 0.2 0.4 0.6 0.8)
train_ppl_log_file="$models/dropout_train_perplexities.tsv"
val_ppl_log_file="$models/dropout_val_perplexities.tsv"
test_ppl_log_file="$models/dropout_test_perplexities.tsv"

rm -f "$train_ppl_log_file" "$val_ppl_log_file" "$test_ppl_log_file"

SECONDS=0

for dropout in "${dropouts[@]}"; do
    dropout_tag=${dropout/./_}
    save_path="$models/model_dropout_${dropout_tag}.pt"

    echo "Training with dropout=$dropout ..."
    (cd "$tools/pytorch-examples/word_language_model" &&
        OMP_NUM_THREADS=$num_threads python main.py --data "$data/onion_news" --accel\
            --epochs 20 \
            --log-interval 50 \
            --emsize 300 --nhid 300 --dropout "$dropout" --tied \
            --save "$save_path" \
            --train-ppl-log-file "$train_ppl_log_file" \
            --val-ppl-log-file "$val_ppl_log_file" \
            --test-ppl-log-file "$test_ppl_log_file"
    )
done

echo "time taken:"
echo "$SECONDS seconds"
echo "Train perplexity table saved to: $train_ppl_log_file"
echo "Validation perplexity table saved to: $val_ppl_log_file"
echo "Test perplexity table saved to: $test_ppl_log_file"

python "$base/scripts/plot_perplexities.py" \
    --train-log "$train_ppl_log_file" \
    --val-log "$val_ppl_log_file" \
    --test-log "$test_ppl_log_file" \
    --out-dir "$base/plots"