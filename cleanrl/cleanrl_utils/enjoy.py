import argparse

from huggingface_hub import hf_hub_download

from cleanrl_utils.evals import MODELS


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="dqn_atari",
        help="the name of this experiment (e.g., ppo, dqn_atari)")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--hf-entity", type=str, default="cleanrl",
        help="the user or org name of the model repository from the Hugging Face Hub")
    parser.add_argument("--hf-repository", type=str, default="",
        help="the huggingface repo (e.g., cleanrl/BreakoutNoFrameskip-v4-dqn_atari-seed1)")
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--eval-episodes", type=int, default=10,
        help="the number of evaluation episodes")
    args = parser.parse_args()
    # fmt: on
    return args


if __name__ == "__main__":
    args = parse_args()
    Model, make_env, evaluate = MODELS[args.exp_name]()
    if not args.hf_repository:
        args.hf_repository = f"{args.hf_entity}/{args.env_id}-{args.exp_name}-seed{args.seed}"
    print(f"loading saved models from {args.hf_repository}...")
    model_path = hf_hub_download(repo_id=args.hf_repository, filename=f"{args.exp_name}.cleanrl_model")
    evaluate(
        model_path,
        make_env,
        args.env_id,
        eval_episodes=args.eval_episodes,
        run_name=f"eval",
        Model=Model,
        capture_video=args.capture_video,
    )
