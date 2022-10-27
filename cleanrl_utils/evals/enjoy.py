import random
from typing import Callable

import gym
import numpy as np
import torch
from huggingface_hub import hf_hub_download


import cleanrl.dqn
import cleanrl.dqn_atari
import cleanrl_utils.evals.dqn_eval

def parse_args():
    import argparse
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="dqn_atari",
        help="the name of this experiment (e.g., ppo, dqn_atari)")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")

    parser.add_argument("--hf-entity", type=str, default="cleanrl",
        help="the user or org name of the model repository from the Hugging Face Hub")
    parser.add_argument("--hf-repo", type=str, default="",
        help="the hf repo (e.g., cleanrl/BreakoutNoFrameskip-v4-dqn_atari-seed1)")
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4",
        help="the id of the environment")
    return parser.parse_args()


MODELS = {
    "dqn": (cleanrl.dqn.QNetwork, cleanrl.dqn.make_env, cleanrl_utils.evals.dqn_eval.evaluate),
    "dqn_atari": (cleanrl.dqn_atari.QNetwork, cleanrl.dqn_atari.make_env, cleanrl_utils.evals.dqn_eval.evaluate),
}


if __name__ == "__main__":
    args = parse_args()
    Model, make_env, evaluate = MODELS[args.exp_name]
    if not args.hf_repo:
        args.hf_repo = f"{args.hf_entity}/{args.env_id}-{args.exp_name}-seed{args.seed}"
    print(args.hf_repo)
    model_path = hf_hub_download(repo_id=args.hf_repo, filename="q_network.pth")
    evaluate(
        model_path,
        make_env,
        args.env_id,
        eval_episodes=10,
        run_name=f"eval",
        Model=Model,
        device="cpu",
        capture_video=False,
    )
