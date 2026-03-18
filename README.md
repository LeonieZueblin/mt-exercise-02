# MT Exercise 2: Pytorch RNN Language Models

Submission for LEZUB for exercise 02 for the MT course. Thanks to Emma van den Bold, the original author of these scripts. 

# Requirements

- This only works on a Unix-like system, with bash.
- Python 3 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`

# Steps

Clone this repository in the desired place:

    git clone https://github.com/LeonieZueblin/mt-exercise-02
    cd mt-exercise-02

Create a new virtualenv that uses Python 3. Please make sure to run this command outside of any virtual Python environment:

    ./scripts/make_virtualenv.sh

**Important**: Then activate the env by executing the `source` command that is output by the shell script above.

Download and install required software:

    ./scripts/install_packages.sh

Download and preprocess data:

    ./scripts/download_data.sh

The download data script has been adapted to automatically use the `huggingface.co/datasets/Biddls/Onion_News` dataset.

Train a model on the baseline settings:

    ./scripts/train.sh

If you are not using a version of PyTorch with CUDA enabled or do not have a CUDA GPU available, you may need to remove the 
--accel flag from the training scripts (scripts/train.sh and scripts train_dropout_sweep.sh) as the original code uses a depreceated syntax to check for CUDA.

To train a model on different dropout values, use the following script:

    ./scripts/train_dropout_sweep.sh

The training process can be interrupted at any time, and the best checkpoint will always be saved. Training takes circa 15-20 minutes on my GPU.  

Generate (sample) some text from a trained model with:

    ./scripts/generate.sh

Edit the model pointer in the generate.sh script to use a different model for generation. 

Use the script:

    python3 scripts/plot_perplexities.py 

to generate the tables and charts for the perplexities from the data written to /models. 