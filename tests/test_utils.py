import subprocess


def test_submit_exp():
    subprocess.run(
        "poetry run python -m cleanrl_utils.submit_exp --docker-tag vwxyzjn/cleanrl:latest --wandb-key xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        shell=True,
        check=True,
    )
