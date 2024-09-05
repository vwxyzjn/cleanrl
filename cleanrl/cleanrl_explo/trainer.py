import wandb
import importlib
import multiprocessing
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Sweep_Args():

    available_methods = ["aux", "icm", "ngu", "rnd"]
    "All the methods available for training"
    method = "aux"
    "The method to use for training"
    fichier = f"sac_{method}"
    "The file to run for training" 
    project: str = f"{method} sweep"
    "The project name to use in wandb"
    nb_of_attempts: int = 5
    "Every hyperparameter combination will be tried this many times, the average will be used"
    nb_of_parallel_jobs: int = 3
    "The number of parallel agents to run (remember that several environments will already be run in parallel for every single agent)"
    count: int = 8
    "The number of hyperparameter combinations to try per agent"


    """
    In order to run the sweep, you must create a sweep configuration dictionnary.
    The documentation for the sweep configuration can be found here: https://docs.wandb.ai/guides/sweeps/configuration
    """

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
            "total_timesteps": {
                'value': 7000,   
            },
            'capture_video': {
                'value': False
            },
        },
    }


    assert method in available_methods, f"method must be in {available_methods}"

    sweep_id = wandb.sweep(sweep_config, project=project)
    "The sweep id to use for the sweep"

def train(args: Sweep_Args):

    try:
        module = importlib.import_module(args.fichier)
        values = []
        for i in range(args.nb_of_attempts):
            values.append(module.main(seed=i, sweep=True))


        values = np.array(values)
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        for step, (mean_, std_) in enumerate(zip(mean, std)):
            wandb.log({
                "episodic_return": mean_,
                "episodic_return_upper": mean_ + std_,
                "episodic_return_lower": mean_ - std_
            }, step=step)

        
    except ModuleNotFoundError:
        print(f"Erreur: le module '{args.fichier}' n'a pas été trouvé.")
    except AttributeError:
        print(f"Erreur: le module '{args.fichier}' n'a pas de fonction 'main'.")
    except Exception as e:
        print(f"Erreur: {e}")


def agent(index: int, args: Sweep_Args):
    print(f"Agent {index} started.")
    
    wandb.agent(args.sweep_id, function=lambda: train(args), project=args.project, count=args.count)
    
    print(f"Agent {index} finished.")


if __name__ == "__main__":

    args = Sweep_Args()
    processes = []
    for i in range(args.nb_of_parallel_jobs):
        p = multiprocessing.Process(target=agent, args=(i, args))
        p.start()
        processes.append(p)


    for p in processes:
        p.join()

    print("All processes have finished.")