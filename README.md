# Learning Sparsifying Transforms (1D)

## One-time setup
1. get a working conda install
1. conda env create --file environment.yml
1. conda activate learnreg1d

## Updating
1. `git pull`
1. `conda env update -f environment.yml`

## Working with the code
* `conda activate learnreg1d`

## Learning the transform on 1D signals
python scripts/1D_learn.py \
  --learning_rate 0.001 \
  --num_steps 1e6 \
  --sign_threshold 0.0001 \
  --signal_type piecewise_constant \
  --n 64 \
  --noise_sigma 0.01 \
  --forward_model_type identity \
  --num_training 10000 \
  --transform_type identity \
  --transform_scale 0.01 \
  --num_testing 5 \
  --seed 0 \
  --batch_size 5

## Learning the transform on 2D images
  
python script_name.py \
  --filename "barbara_gray.bmp" \
  --patch_size 8 \
  --forward_model_type identity \
  --noise_sigma 0.01 \
  --transform_type identity \
  --transform_scale 0.01 \
  --SEED 0 \
  --learning_rate 0.001 \
  --num_steps 100000 \
  --sign_threshold 0.0001 \
  --batch_size 5


## Gradients and time comparisons with autodiff solvers

Run the notebook scripts/check_gradients.ipynb



