python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=cleanrl&ceik=env_id&cen=exp_name&metric=charts/episodic_return' \
        'sac_continuous_action?tag=pr-424' \
    --env-ids HalfCheetah-v4 Walker2d-v4 Hopper-v4 InvertedPendulum-v4 Humanoid-v4 Pusher-v4 \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 2 \
    --output-filename benchmark/cleanrl/sac \
    --scan-history
