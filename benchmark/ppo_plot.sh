python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=cleanrl&ceik=env_id&cen=exp_name&metric=charts/episodic_return' \
        'ppo?tag=pr-424' \
    --env-ids CartPole-v1 Acrobot-v1 MountainCar-v0 \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 2 \
    --output-filename benchmark/cleanrl/ppo \
    --scan-history

python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=cleanrl&ceik=env_id&cen=exp_name&metric=charts/episodic_return' \
        'ppo_atari?tag=pr-424' \
    --env-ids PongNoFrameskip-v4 BeamRiderNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 2 \
    --output-filename benchmark/cleanrl/ppo_atari \
    --scan-history

python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=cleanrl&ceik=env_id&cen=exp_name&metric=charts/episodic_return' \
        'ppo_continuous_action?tag=pr-424' \
    --env-ids HalfCheetah-v4 Walker2d-v4 Hopper-v4 InvertedPendulum-v4 Humanoid-v4 Pusher-v4 dm_control/acrobot-swingup-v0 dm_control/acrobot-swingup_sparse-v0 dm_control/ball_in_cup-catch-v0 \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 2 \
    --output-filename benchmark/cleanrl/ppo_continuous_action \
    --scan-history

python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=cleanrl&ceik=env_id&cen=exp_name&metric=charts/episodic_return' \
        'ppo_continuous_action?tag=v1.0.0-13-gcbd83f6' \
    --env-ids dm_control/acrobot-swingup-v0 dm_control/acrobot-swingup_sparse-v0 dm_control/ball_in_cup-catch-v0 dm_control/cartpole-balance-v0 dm_control/cartpole-balance_sparse-v0 dm_control/cartpole-swingup-v0 dm_control/cartpole-swingup_sparse-v0 dm_control/cartpole-two_poles-v0 dm_control/cartpole-three_poles-v0 dm_control/cheetah-run-v0 dm_control/dog-stand-v0 dm_control/dog-walk-v0 dm_control/dog-trot-v0 dm_control/dog-run-v0 dm_control/dog-fetch-v0 dm_control/finger-spin-v0 dm_control/finger-turn_easy-v0 dm_control/finger-turn_hard-v0 dm_control/fish-upright-v0 dm_control/fish-swim-v0 dm_control/hopper-stand-v0 dm_control/hopper-hop-v0 dm_control/humanoid-stand-v0 dm_control/humanoid-walk-v0 dm_control/humanoid-run-v0 dm_control/humanoid-run_pure_state-v0 dm_control/humanoid_CMU-stand-v0 dm_control/humanoid_CMU-run-v0 dm_control/lqr-lqr_2_1-v0 dm_control/lqr-lqr_6_2-v0 dm_control/manipulator-bring_ball-v0 dm_control/manipulator-bring_peg-v0 dm_control/manipulator-insert_ball-v0 dm_control/manipulator-insert_peg-v0 dm_control/pendulum-swingup-v0 dm_control/point_mass-easy-v0 dm_control/point_mass-hard-v0 dm_control/quadruped-walk-v0 dm_control/quadruped-run-v0 dm_control/quadruped-escape-v0 dm_control/quadruped-fetch-v0 dm_control/reacher-easy-v0 dm_control/reacher-hard-v0 dm_control/stacker-stack_2-v0 dm_control/stacker-stack_4-v0 dm_control/swimmer-swimmer6-v0 dm_control/swimmer-swimmer15-v0 dm_control/walker-stand-v0 dm_control/walker-walk-v0 dm_control/walker-run-v0 \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 2 \
    --output-filename benchmark/cleanrl/ppo_continuous_action_dm_control \
    --scan-history

python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=cleanrl&ceik=env_id&cen=exp_name&metric=charts/episodic_return' \
        'ppo_atari_lstm?tag=pr-424' \
    --env-ids PongNoFrameskip-v4 BeamRiderNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 2 \
    --output-filename benchmark/cleanrl/ppo_atari_lstm \
    --scan-history

python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=cleanrl&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' \
        'ppo_atari_envpool?tag=pr-424' \
    --filters '?we=openrlbenchmark&wpn=cleanrl&ceik=env_id&cen=exp_name&metric=charts/episodic_return' \
        'ppo_atari?tag=pr-424' \
    --env-ids Pong-v5 BeamRider-v5 Breakout-v5 \
    --env-ids PongNoFrameskip-v4 BeamRiderNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 2 \
    --output-filename benchmark/cleanrl/ppo_atari_envpool \
    --scan-history

python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=envpool-atari&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' \
        'ppo_atari_envpool_xla_jax' \
    --filters '?we=openrlbenchmark&wpn=baselines&ceik=env&cen=exp_name&metric=charts/episodic_return' \
        'baselines-ppo2-cnn' \
    --env-ids Alien-v5 Amidar-v5 Assault-v5 Asterix-v5 Asteroids-v5 Atlantis-v5 BankHeist-v5 BattleZone-v5 BeamRider-v5 Berzerk-v5 Bowling-v5 Boxing-v5 Breakout-v5 Centipede-v5 ChopperCommand-v5 CrazyClimber-v5 Defender-v5 DemonAttack-v5 DoubleDunk-v5 Enduro-v5 FishingDerby-v5 Freeway-v5 Frostbite-v5 Gopher-v5 Gravitar-v5 Hero-v5 IceHockey-v5 Jamesbond-v5 Kangaroo-v5 Krull-v5 KungFuMaster-v5 MontezumaRevenge-v5 MsPacman-v5 NameThisGame-v5 Phoenix-v5 Pitfall-v5 Pong-v5 PrivateEye-v5 Qbert-v5 Riverraid-v5 RoadRunner-v5 Robotank-v5 Seaquest-v5 Skiing-v5 Solaris-v5 SpaceInvaders-v5 StarGunner-v5 Surround-v5 Tennis-v5 TimePilot-v5 Tutankham-v5 UpNDown-v5 Venture-v5 VideoPinball-v5 WizardOfWor-v5 YarsRevenge-v5 Zaxxon-v5 \
    --env-ids AlienNoFrameskip-v4 AmidarNoFrameskip-v4 AssaultNoFrameskip-v4 AsterixNoFrameskip-v4 AsteroidsNoFrameskip-v4 AtlantisNoFrameskip-v4 BankHeistNoFrameskip-v4 BattleZoneNoFrameskip-v4 BeamRiderNoFrameskip-v4 BerzerkNoFrameskip-v4 BowlingNoFrameskip-v4 BoxingNoFrameskip-v4 BreakoutNoFrameskip-v4 CentipedeNoFrameskip-v4 ChopperCommandNoFrameskip-v4 CrazyClimberNoFrameskip-v4 DefenderNoFrameskip-v4 DemonAttackNoFrameskip-v4 DoubleDunkNoFrameskip-v4 EnduroNoFrameskip-v4 FishingDerbyNoFrameskip-v4 FreewayNoFrameskip-v4 FrostbiteNoFrameskip-v4 GopherNoFrameskip-v4 GravitarNoFrameskip-v4 HeroNoFrameskip-v4 IceHockeyNoFrameskip-v4 JamesbondNoFrameskip-v4 KangarooNoFrameskip-v4 KrullNoFrameskip-v4 KungFuMasterNoFrameskip-v4 MontezumaRevengeNoFrameskip-v4 MsPacmanNoFrameskip-v4 NameThisGameNoFrameskip-v4 PhoenixNoFrameskip-v4 PitfallNoFrameskip-v4 PongNoFrameskip-v4 PrivateEyeNoFrameskip-v4 QbertNoFrameskip-v4 RiverraidNoFrameskip-v4 RoadRunnerNoFrameskip-v4 RobotankNoFrameskip-v4 SeaquestNoFrameskip-v4 SkiingNoFrameskip-v4 SolarisNoFrameskip-v4 SpaceInvadersNoFrameskip-v4 StarGunnerNoFrameskip-v4 SurroundNoFrameskip-v4 TennisNoFrameskip-v4 TimePilotNoFrameskip-v4 TutankhamNoFrameskip-v4 UpNDownNoFrameskip-v4 VentureNoFrameskip-v4 VideoPinballNoFrameskip-v4 WizardOfWorNoFrameskip-v4 YarsRevengeNoFrameskip-v4 ZaxxonNoFrameskip-v4 \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 2 \
    --rliable \
    --rc.score_normalization_method atari \
    --rc.normalized_score_threshold 8.0 \
    --rc.sample_efficiency_plots \
    --rc.sample_efficiency_and_walltime_efficiency_method Median \
    --rc.performance_profile_plots  \
    --rc.aggregate_metrics_plots  \
    --rc.sample_efficiency_num_bootstrap_reps 50000 \
    --rc.performance_profile_num_bootstrap_reps 50000 \
    --rc.interval_estimates_num_bootstrap_reps 50000 \
    --output-filename benchmark/cleanrl/ppo_atari_envpool_xla_jax \
    --scan-history

python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=cleanrl&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' \
        'ppo_atari_envpool_xla_jax?tag=pr-424' \
        'ppo_atari_envpool_xla_jax_scan?tag=pr-424' \
    --env-ids Pong-v5 BeamRider-v5 Breakout-v5 \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 2 \
    --output-filename benchmark/cleanrl/ppo_atari_envpool_xla_jax_scan \
    --scan-history

python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=cleanrl&ceik=env_id&cen=exp_name&metric=charts/episodic_return' \
        'ppo_procgen?tag=pr-424' \
    --env-ids starpilot bossfight bigfish \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 2 \
    --output-filename benchmark/cleanrl/ppo_procgen \
    --scan-history

python -m openrlbenchmark.rlops \
    --filters '?we=openrlbenchmark&wpn=cleanrl&ceik=env_id&cen=exp_name&metric=charts/episodic_return' \
        'ppo_atari_multigpu?tag=pr-424' \
        'ppo_atari?tag=pr-424' \
    --env-ids PongNoFrameskip-v4 BeamRiderNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 2 \
    --output-filename benchmark/cleanrl/ppo_atari_multigpu \
    --scan-history
