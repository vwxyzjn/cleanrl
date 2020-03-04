# norm rewards, reset
%%!
git clone https://github.com/vwxyzjn/cleanrl.git 
cd cleanrl
pip install -e .
cd cleanrl
apt-get update
apt -qq install xvfb
pip install pybullet==2.6.5
wandb login 6603a1e99a016ac5002729a06b08e13931d4ee02

for seed in {2..2}
do
    (sleep 0.3 && nohup xvfb-run -a python3 ppo2_continuous_action.py \
    --gym-id AntBulletEnv-v0 \
    --total-timesteps 2000000 \
    --wandb-project-name ppo-kl \
    --wandb-entity cleanrl \
    --prod-mode \
    --kl --norm-returns \
    --no-cuda \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done
