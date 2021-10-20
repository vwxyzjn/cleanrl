for seed in {1..2}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_car_racing.py \
    --wandb-project-name cleanrl.benchmark \
    --track \
    --wandb-entity cleanrl --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..1}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_procgen_visual.py \
    --wandb-project-name cleanrl.benchmark \
    --track \
    --wandb-entity cleanrl --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..1}
do
    (sleep 0.3 && nohup xvfb-run -a python ppg_atari_visual.py \
    --wandb-project-name cleanrl.benchmark \
    --track \
    --wandb-entity cleanrl --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

for seed in {1..1}
do
    (sleep 0.3 && nohup xvfb-run -a python ppg_procgen_visual.py \
    --wandb-project-name cleanrl.benchmark \
    --track \
    --wandb-entity cleanrl --cuda False \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done

# 16192 * 32
# Out[2]: 518144