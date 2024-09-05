import wandb

sweep_config = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "episodic_return"},
    "parameters": {
        "vae_lr": {
            "distribution": "log_uniform_values",
            "max": 1e-2,
            "min": 1e-5,
        },
        "vae_frequency": {
            "distribution": "q_uniform",
            "max": 1000,
            "min": 100,
            "q": 100,
        },
        "coef_intrinsic": {
            "distribution": "log_uniform_values",
            "max": 100.0,
            "min": 0.1,
        },
        "coef_extrinsic": {
            "distribution": "log_uniform_values",
            "max": 100.0,
            "min": 0.1,
        },
        'number_of_attempts': {
            'value': 5
        },
        'capture_video': {
            'value': False
        },

        
    },
}

sweep_id = wandb.sweep(sweep_config, project="aux hyperparameters optimization")
print(sweep_id)