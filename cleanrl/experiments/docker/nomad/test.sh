for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_action.py \
    --gym-id Walker2DBulletEnv-v0 \
    --total-timesteps 20000 \
    --wandb-project-name nomadtest \
    --wandb-entity cleanrl \
    --prod-mode \
    --norm-returns \
    --no-cuda \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done
