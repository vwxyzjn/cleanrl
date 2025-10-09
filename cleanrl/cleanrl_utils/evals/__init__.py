def dqn():
    import cleanrl.dqn
    import cleanrl_utils.evals.dqn_eval

    return cleanrl.dqn.QNetwork, cleanrl.dqn.make_env, cleanrl_utils.evals.dqn_eval.evaluate


def dqn_atari():
    import cleanrl.dqn_atari
    import cleanrl_utils.evals.dqn_eval

    return cleanrl.dqn_atari.QNetwork, cleanrl.dqn_atari.make_env, cleanrl_utils.evals.dqn_eval.evaluate


def dqn_jax():
    import cleanrl.dqn_jax
    import cleanrl_utils.evals.dqn_jax_eval

    return cleanrl.dqn_jax.QNetwork, cleanrl.dqn_jax.make_env, cleanrl_utils.evals.dqn_jax_eval.evaluate


def dqn_atari_jax():
    import cleanrl.dqn_atari_jax
    import cleanrl_utils.evals.dqn_jax_eval

    return cleanrl.dqn_atari_jax.QNetwork, cleanrl.dqn_atari_jax.make_env, cleanrl_utils.evals.dqn_jax_eval.evaluate


def c51():
    import cleanrl.c51
    import cleanrl_utils.evals.c51_eval

    return cleanrl.c51.QNetwork, cleanrl.c51.make_env, cleanrl_utils.evals.c51_eval.evaluate


def c51_atari():
    import cleanrl.c51_atari
    import cleanrl_utils.evals.c51_eval

    return cleanrl.c51_atari.QNetwork, cleanrl.c51_atari.make_env, cleanrl_utils.evals.c51_eval.evaluate


def c51_jax():
    import cleanrl.c51_jax
    import cleanrl_utils.evals.c51_jax_eval

    return cleanrl.c51_jax.QNetwork, cleanrl.c51_jax.make_env, cleanrl_utils.evals.c51_jax_eval.evaluate


def c51_atari_jax():
    import cleanrl.c51_atari_jax
    import cleanrl_utils.evals.c51_jax_eval

    return cleanrl.c51_atari_jax.QNetwork, cleanrl.c51_atari_jax.make_env, cleanrl_utils.evals.c51_jax_eval.evaluate


def ppo_atari_envpool_xla_jax_scan():
    import cleanrl.ppo_atari_envpool_xla_jax_scan
    import cleanrl_utils.evals.ppo_envpool_jax_eval

    return (
        (
            cleanrl.ppo_atari_envpool_xla_jax_scan.Network,
            cleanrl.ppo_atari_envpool_xla_jax_scan.Actor,
            cleanrl.ppo_atari_envpool_xla_jax_scan.Critic,
        ),
        cleanrl.ppo_atari_envpool_xla_jax_scan.make_env,
        cleanrl_utils.evals.ppo_envpool_jax_eval.evaluate,
    )


MODELS = {
    "dqn": dqn,
    "dqn_atari": dqn_atari,
    "dqn_jax": dqn_jax,
    "dqn_atari_jax": dqn_atari_jax,
    "c51": c51,
    "c51_atari": c51_atari,
    "c51_jax": c51_jax,
    "c51_atari_jax": c51_atari_jax,
    "ppo_atari_envpool_xla_jax_scan": ppo_atari_envpool_xla_jax_scan,
}
