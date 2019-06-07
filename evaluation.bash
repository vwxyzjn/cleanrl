# https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop
# evaluate the algorithm in parallel

for seed in {1..3}
do
    nohup python a2c.py --seed $seed >& /dev/null &
done