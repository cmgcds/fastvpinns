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
import tensorflow as tf
import os


class OptunaTuner:
    def __init__(
        self, n_trials=100, study_name="fastvpinns_optimization", n_jobs=-1, n_epochs=5000
    ):
        self.n_trials = n_trials
        self.study_name = study_name
        self.n_jobs = n_jobs
        self.n_epochs = n_epochs
        self.gpus = tf.config.list_physical_devices('GPU')
        print(f"Available GPUs: {len(self.gpus)}")

    def objective_wrapper(self, trial):
        """
        Wrapper function to run the objective function on a specific GPU.
        """

        gpu_id = trial.number % len(self.gpus)
        with tf.device(f'/device:GPU:{gpu_id}'):
            return objective(trial, self.n_epochs)

    def run(self):
        study = optuna.create_study(
            study_name=self.study_name,
            direction="minimize",
            storage="sqlite:///fastvpinns_optuna.db",
            load_if_exists=True,
        )
        study.optimize(
            self.objective_wrapper, n_trials=self.n_trials, n_jobs=min(len(self.gpus), self.n_jobs)
        )

        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        return study.best_params
