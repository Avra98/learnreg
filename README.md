# Learning Sparsifying Transforms (1D)

## Table of Contents
- One-time Setup
- Updating
- Working with the Code
- Learning the Transform on 1D Signals
- Learning the Transform on 2D Images
- Gradients and Time Comparisons

## One-time Setup
1. Install a working version of conda.
2. Create a conda environment: `conda env create --file environment.yml`
3. Activate the conda environment: `conda activate learnreg1d`

## Updating
1. Pull the latest changes: `git pull`
2. Update the conda environment: `conda env update -f environment.yml`

## Working with the Code
Activate the conda environment before working with the code: `conda activate learnreg1d`

## Learning the Transform on 1D Signals

![Transforms learnt on 1D signal pairs](data/lertran2.png)

Run the following command to learn the transform on 1D signals:

python scripts/1D_learn.py --learning_rate 0.001 --num_steps 1e6 --sign_threshold 0.0001 --signal_type piecewise_constant --n 64 --noise_sigma 0.01 --forward_model_type identity --num_training 10000 --transform_type identity --transform_scale 0.01 --num_testing 5 --seed 0 --batch_size 5


## Learning the Transform on 2D Images

![Transforms learnt on 2D image pairs](data/image_Strips.png)

Run the following command to learn the transform on 2D images:

python script_name.py --filename "barbara_gray.bmp" --patch_size 8 --forward_model_type identity --noise_sigma 0.01 --transform_type identity --transform_scale 0.01 --SEED 0 --learning_rate 0.001 --num_steps 100000 --sign_threshold 0.0001 --batch_size 5


## Gradients and Time Comparisons

![Comparisons with autodiff solvers](data/grad_comb.png)

Run the notebook `scripts/check_gradients.ipynb` for gradient and time comparisons with autodiff solvers.

## Adding Images to README
To add images to this README, upload the image files to your repository and then use the following Markdown syntax: `![Description](path/to/image)`


