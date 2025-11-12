# !/bin/bash
echo "starting benchmarking script..."
#
GPU=0
model="ABMIL"
seeds="0 1 2 3 4 5 6 7 8 9"
studies="task1 task2 task3"
features="resnet50 uni uni2 conch chief gpfm mstar virchow2 conch15 phikon ctranspath phikon2 plip gigapath virchow h-optimus-0 h-optimus-1 musk hibou-l omiclip patho_clip"

for study in $studies
do
    for feature in $features
    do
        for seed in $seeds
        do
            echo "$(date): [study: ${study}]-[feature: ${feature}]-[seed: ${seed}]"
            CUDA_VISIBLE_DEVICES=$GPU python main.py --study $study \
                                                     --feature $feature \
                                                     --seed $seed \
                                                     --all_datasets ./splits/datasets.xlsx \
                                                     --model $model \
                                                     --excel_file ./splits/${study}_split.xlsx \
                                                     --num_epoch 50 \
                                                     --early_stop 10
            
        done
    done
done