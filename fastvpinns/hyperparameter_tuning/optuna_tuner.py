# optuna_tuner.py
"""
Optuna-based hyperparameter tuner for FastVPINNs.

This module provides the OptunaTuner class, which implements hyperparameter
tuning using the Optuna optimization framework. It allows for efficient
exploration of the hyperparameter space to find optimal configurations
for FastVPINNs models.

Classes:
    OptunaTuner: Manages the hyperparameter tuning process using Optuna.

Usage:
    tuner = OptunaTuner(n_trials=100, study_name="my_optimization")
    best_params = tuner.run()

Note:
    This module requires the 'optuna' package to be installed.
"""

import optuna
from .objective import objective

class OptunaTuner:
    def __init__(self, n_trials=100, study_name="fastvpinns_optimization"):
        self.n_trials = n_trials
        self.study_name = study_name

    def run(self):
        study = optuna.create_study(direction="minimize", study_name=self.study_name)
        study.optimize(objective, n_trials=self.n_trials)

        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        return study.best_params