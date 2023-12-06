import optuna

from cleanrl_utils.tuner import Tuner


def test_tuner():
    tuner = Tuner(
        script="cleanrl/ppo.py",
        metric="charts/episodic_return",
        metric_last_n_average_window=50,
        direction="maximize",
        target_scores={
            "CartPole-v1": [0, 500],
            "Acrobot-v1": [-500, 0],
        },
        params_fn=lambda trial: {
            "learning-rate": trial.suggest_float("learning-rate", 0.0003, 0.003, log=True),
            "num-minibatches": trial.suggest_categorical("num-minibatches", [1, 2, 4]),
            "update-epochs": trial.suggest_categorical("update-epochs", [1, 2, 4]),
            "num-steps": trial.suggest_categorical("num-steps", [1200]),
            "vf-coef": trial.suggest_float("vf-coef", 0, 5),
            "max-grad-norm": trial.suggest_float("max-grad-norm", 0, 5),
            "total-timesteps": 1200,
            "num-envs": 1,
        },
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        sampler=optuna.samplers.TPESampler(),
        # wandb_kwargs={"project": "cleanrl"},
    )
    tuner.tune(
        num_trials=1,
        num_seeds=1,
    )
