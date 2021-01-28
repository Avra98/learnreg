"""

quick example:
>> python scripts/run_exp.py with basic 'num_steps=1000' 'num_testing=1'
"""

import sacred

import learnreg as lr

ex = sacred.Experiment()
#ex.observers.append(sacred.observers.MongoObserver())

# extra named configs, used via "with <config name>" command line option
ex.named_config(lr.configs.basic)
ex.named_config(lr.configs.baseline)

ex.main(lr.main)

ex.run_commandline()

#ex.automain(f)
