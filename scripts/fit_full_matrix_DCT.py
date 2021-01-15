"""
learn a W on a training set of piecewise constant signals
"""

import sacred
import numpy as np

import learnreg as lr

ex = sacred.Experiment()
ex.observers.append(sacred.observers.MongoObserver())
@ex.config
def config():
    learning_rate = 1e-3
    num_steps = int(1e6) # 1e5 takes a 30 minutes, can give good results
    sign_threshold = 1e-4

    signal_type = 'DCT-sparse'
    n = 64
    noise_sigma = 1e-1

    forward_model_type = 'identity'
    num_training = 10000
    transform_type = 'identity' # start with identity for learning
    transform_scale = 1e-2 # 1e-2 seems usually good
    num_testing = 50
    SEED = 0


ex.automain(lr.main)
ex.run_commandline()
