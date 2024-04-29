#! /bin/bash

backbones=("hybrid" "gcn"  "gen" "gat")
ood_types=("structure" "feature" "label")
datasets=("cora" "amazon-photo" "coauthor-cs" "twitch" "arxiv")
use_reg_options=("" "--use_reg --m_in -5 --m_out -1")
use_prop_options=("" "--use_prop")
use_oc_options=("" "--use_oc")
use_gradnorm_options=("--grad_norm")
epochs=125
device=0
lamda1=0.01
lamda2=0.01
nu=0.1
eps=0.01

# Loop over different combinations of parameters
for backbone in "${backbones[@]}"; do
    for ood_type in "${ood_types[@]}"; do
        for use_reg_option in "${use_reg_options[@]}"; do
            for use_prop_option in "${use_prop_options[@]}"; do
                for use_oc_option in "${use_oc_options[@]}"; do
                    for use_gradnorm_option in "${use_gradnorm_options[@]}"; do
                        for dataset in "${datasets[@]}"; do
                            cmd="python3 main.py --method gnnsafe --dataset $dataset --backbone $backbone --ood_type $ood_type --epochs $epochs --device $device --lamda1 $lamda1 --lamda2 $lamda2 --nu $nu --eps $eps $use_reg_option $use_prop_option $use_oc_option $use_gradnorm_option"
                            echo $cmd
                            $cmd
                        done
                    done
                done
            done
        done
    done
done