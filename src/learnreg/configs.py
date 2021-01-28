
def basic():
    learning_rate = 1e-3
    num_steps = int(1e5)
    sign_threshold = 1e-4

    signal_type = 'piecewise_constant'
    n = 64
    noise_sigma = 1e-1

    forward_model_type = 'identity'
    num_training = 10000
    batch_size = 10

    transform_type = 'identity'
    k = 64
    transform_scale = 1e-2

    num_testing = 50
    SEED = 0


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
