"""

add the flag "-m monogodb" to save to the monogodb (assuming it is running)

quick example:
>>python scripts/run_exp.py with basic 'num_steps=100' 'num_testing=1'


"""

import sacred

import learnreg as lr


ex = sacred.Experiment()

# define named configs to be used from the command line
@ex.named_config
def basic():
    learning_rate = 1e-3
    num_steps = int(1e6)
    sign_threshold = 1e-4

    signal_type = 'piecewise_constant'
    n = 64
    noise_sigma = 1e-1

    forward_model_type = 'identity'
    num_training = 10000
    batch_size = 1

    transform_type = 'identity'
    k = 64
    transform_scale = 1e-2

    num_testing = 50
    seed = 0


@ex.named_config
def baseline(n, signal_type):
    learning_rate = 0.0
    num_steps = 0

    if signal_type == 'piecewise_constant':
        transform_type = 'TV'
        k = n - 1
    elif signal_type == 'DCT-sparse':
        transform_type = 'DCT'
        k = n
    else:
        raise ValueError(signal_type)

    transform_scale = 1.0


ex.main(lr.main)
ex.run_commandline()
