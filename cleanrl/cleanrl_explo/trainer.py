import wandb
import importlib
import multiprocessing
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Sweep_Args():

    
    ############################### IMPORTANT ################################
    """
    This code produces a sweep for training a SAC-exploration agent with different hyperparameters.
    It will print an id. This id can be used to run the same sweep in parellel on different machines.
    Thus, you can run the same sweep on different machines and the results will be aggregated in the same wandb project,
    therefore speeding up the hyperparameter search.

    To do so you must run same script than this one on the other machines, but with the same sweep id. 
    So you must copy the sweep id from the output of this script and paste it in the other scripts.
    """

    ###########################################################################

    available_methods = ["aux", "icm", "ngu", "rnd", "apt", "our_method"]
    "All the methods available for training"


    method = "our_method"
    "The method to use for training"
    environment = "LilMaze"
    "The environment to use for training"
    nb_of_attempts: int = 1
    "Every hyperparameter combination will be tried this many times, the average will be used"
    nb_of_parallel_jobs: int = 1
    "The number of parallel agents to run (remember that several environments will already be run for every single agent)"
    count: int = 10
    "The number of hyperparameter combinations to try per agent"


    fichier = f"sac_{method}"
    "The file to run for training" 
    project: str = f"{method} sweep {environment}"
    "The project name to use in wandb"
    


    """
    In order to run the sweep, you must create a sweep configuration dictionnary.
    The documentation for the sweep configuration can be found here: https://docs.wandb.ai/guides/sweeps/configuration
    """

    sweep_config = {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "episodic_return"},
        "parameters": {
            "classifier_lr": {
                "distribution": "log_uniform_values",
                "max": 1e-2,
                "min": 1e-5,
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
                'value': 200000,   
            },
            'capture_video': {
                'value': False
            },
            'keep_extrinsic_reward': {
                'value': False
            },
            'env_id': {
                'value': f"{environment}"
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
        steps = []
        for i in range(args.nb_of_attempts):
            v, t = module.main(seed=i, sweep=True)

            values += v
            steps += t



        values = np.array(values)
        steps = np.array(steps).reshape(-1, 1)


        # We use the quantile regression to get the median and the 95% confidence interval
        

        from sklearn.ensemble import GradientBoostingRegressor

        gbm_median = GradientBoostingRegressor(loss="quantile", alpha=0.5, n_estimators=100)
        gbm_median.fit(steps, values)

        gbm_upper = GradientBoostingRegressor(loss="quantile", alpha=0.975, n_estimators=100)
        gbm_upper.fit(steps, values)

        gbm_lower = GradientBoostingRegressor(loss="quantile", alpha=0.025, n_estimators=100)
        gbm_lower.fit(steps, values)


        plot_steps = np.linspace(steps.min(), steps.max(), 200)[:, np.newaxis]

        y_pred_median = gbm_median.predict(plot_steps).ravel()
        y_pred_upper = gbm_upper.predict(plot_steps).ravel()
        y_pred_lower = gbm_lower.predict(plot_steps).ravel()


        for t, min, median, max in list(zip(steps, y_pred_lower, y_pred_median, y_pred_upper)):
            wandb.log({
                "episodic_return": median,
                "episodic_return_upper": max,
                "episodic_return_lower": min
            }, step=t[0])

        
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