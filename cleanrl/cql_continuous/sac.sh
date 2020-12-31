#!/bin/bash
NUM_CORES=$(nproc --all)
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES

# CQL
for gym_id in "Hopper-v2"; do
    for offline_dataset_id in "medium-v0" "expert-v0"; do
        for seed in {1..2}; do
            for min_q_version in 2 3; do
                # # without lagrange
                # (sleep 0.3 && nohup xvfb-run -a -s "-screen :0 640x480x24" python sac_cql.py \
                #     --gym-id $gym_id \
                #     --offline-dataset-id $offline_dataset_id \
                #     --seed $seed \
                #     --min-q-version $min_q_version \
                #     --capture-video \
                #     --autotune \
                #     --prod-mode --wandb-project-name cleanrl.benchmark --wandb-entity cleanrl
                # ) >& /dev/null &

                # # with lagrange
                # (sleep 0.3 && nohup xvfb-run -a -s "-screen :0 640x480x24" python sac_cql.py \
                #     --gym-id $gym_id \
                #     --offline-dataset-id $offline_dataset_id \
                #     --seed $seed \
                #     --min-q-version $min_q_version \
                #     --capture-video \
                #     --autotune \
                #     --with-lagrange \
                #     --prod-mode --wandb-project-name cleanrl.benchmark --wandb-entity cleanrl
                # ) >& /dev/null &
            done
        done
    done
done