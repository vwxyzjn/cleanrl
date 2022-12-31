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


MODELS = {
    "dqn": dqn,
    "dqn_atari": dqn_atari,
    "dqn_jax": dqn_jax,
    "dqn_atari_jax": dqn_atari_jax,
    "c51": c51,
}
