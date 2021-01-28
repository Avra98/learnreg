"""
learn a W on a training set of piecewise constant signals
"""

import sacred
import numpy as np

import learnreg as lr

ex = sacred.Experiment()
ex.observers.append(sacred.observers.MongoObserver())

ex.config(lr.configs.basic)
ex.config(lr.configs.baseline)

ex.automain(lr.main)
ex.run_commandline()
