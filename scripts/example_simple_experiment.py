"""
example of how to run the system in the simplest way
"""

import learnreg as lr

MSE, beta_W, W = lr.main(
    learning_rate=1e-3,
    num_steps=int(1e3),  # 1e5 takes a 30 minutes, can give okay results
    sign_threshold=1e-4,
    signal_type='DCT-sparse',
    n=64,
    noise_sigma=1e-1,
    forward_model_type='identity',
    k=63,
    num_training=10000,
    transform_type='identity',
    transform_scale=1e-2,
    num_testing=5,  # 50 is better for making comparisons
    SEED=0
    )

print(f'MSE was {MSE:.3e}')

# save or plot your W
