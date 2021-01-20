import numpy as np
import sacred

import learnreg as lr

runs = lr.reports.get_runs_from_db()

if runs.find_one({'experiment.name': 'piecewise_constant_baseline'}) is None:
    ex = sacred.Experiment('piecewise_constant_baseline')
    ex.observers.append(sacred.observers.MongoObserver())

    ex.config(lr.configs.basic)
    ex.config(lr.configs.baseline)

    ex.automain(lr.main)
    #ex.run_commandline(['', 'print_config'])
    ex.run_commandline()

if runs.find_one({'experiment.name': 'DCT_baseline'}) is None:
    ex = sacred.Experiment('DCT_baseline')
    ex.observers.append(sacred.observers.MongoObserver())

    ex.config(lr.configs.basic)
    ex.add_config({'signal_type': 'DCT-sparse'})
    ex.config(lr.configs.baseline)

    ex.automain(lr.main)
    #ex.run_commandline(['', 'print_config'])
    ex.run_commandline()
