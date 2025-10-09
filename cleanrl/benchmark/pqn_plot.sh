
python -m openrlbenchmark.rlops \
    --filters '?we=rogercreus&wpn=cleanRL&ceik=env_id&cen=exp_name&metric=charts/episodic_return' \
        'pqn?tag=pr-494&cl=CleanRL PQN' \
    --env-ids CartPole-v1 Acrobot-v1 MountainCar-v0 \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 2 \
    --output-filename benchmark/cleanrl/pqn \
    --scan-history

python -m openrlbenchmark.rlops \
    --filters '?we=rogercreus&wpn=cleanRL&ceik=env_id&cen=exp_name&metric=charts/episodic_return' \
        'pqn_atari_envpool?tag=pr-494&cl=CleanRL PQN' \
    --env-ids Breakout-v5 SpaceInvaders-v5 BeamRider-v5 Pong-v5 MsPacman-v5 \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 3 \
    --rliable \
    --rc.score_normalization_method maxmin \
    --rc.normalized_score_threshold 1.0 \
    --rc.sample_efficiency_plots \
    --rc.sample_efficiency_and_walltime_efficiency_method Median \
    --rc.performance_profile_plots  \
    --rc.aggregate_metrics_plots  \
    --rc.sample_efficiency_num_bootstrap_reps 10 \
    --rc.performance_profile_num_bootstrap_reps 10 \
    --rc.interval_estimates_num_bootstrap_reps 10 \
    --output-filename static/0compare \
    --scan-history

python -m openrlbenchmark.rlops \
    --filters '?we=rogercreus&wpn=cleanRL&ceik=env_id&cen=exp_name&metric=charts/episodic_return' \
        'pqn_atari_envpool_lstm?tag=pr-494&cl=CleanRL PQN' \
    --env-ids Breakout-v5 SpaceInvaders-v5 BeamRider-v5 Pong-v5 MsPacman-v5 \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 3 \
    --rliable \
    --rc.score_normalization_method maxmin \
    --rc.normalized_score_threshold 1.0 \
    --rc.sample_efficiency_plots \
    --rc.sample_efficiency_and_walltime_efficiency_method Median \
    --rc.performance_profile_plots  \
    --rc.aggregate_metrics_plots  \
    --rc.sample_efficiency_num_bootstrap_reps 10 \
    --rc.performance_profile_num_bootstrap_reps 10 \
    --rc.interval_estimates_num_bootstrap_reps 10 \
    --output-filename static/0compare \
    --scan-history
