#### Cora with structure ood
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --use_prop --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01

python main.py --method gnnsafe --backbone gen --dataset cora --ood_type structure --mode detect --use_bn --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gen --dataset cora --ood_type structure --mode detect --use_bn --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gen --dataset cora --ood_type structure --mode detect --use_bn --use_prop --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gen --dataset cora --ood_type structure --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 
python main.py --method gnnsafe --backbone gen --dataset cora --ood_type structure --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01


### Cora with feature ood
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type feature --mode detect --use_bn --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type feature --mode detect --use_bn --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type feature --mode detect --use_bn --use_prop --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type feature --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01

python main.py --method gnnsafe --backbone gen --dataset cora --ood_type feature --mode detect --use_bn --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gen --dataset cora --ood_type feature --mode detect --use_bn --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gen --dataset cora --ood_type feature --mode detect --use_bn --use_prop --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gen --dataset cora --ood_type feature --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0
python main.py --method gnnsafe --backbone gen --dataset cora --ood_type feature --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01

### Cora with label ood

python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type label --mode detect --use_bn --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type label --mode detect --use_bn --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type label --mode detect --use_bn --use_prop --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type label --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01

python main.py --method gnnsafe --backbone gen --dataset cora --ood_type label --mode detect --use_bn --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gen --dataset cora --ood_type label --mode detect --use_bn --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gen --dataset cora --ood_type label --mode detect --use_bn --use_prop --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gen --dataset cora --ood_type label --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0
python main.py --method gnnsafe --backbone gen --dataset cora --ood_type label --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01


### Amazon-photo with structure ood

python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type structure --mode detect --use_bn --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type structure --mode detect --use_bn --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type structure --mode detect --use_bn --use_prop --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type structure --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01

python main.py --method gnnsafe --backbone gen --dataset amazon-photo --ood_type structure --mode detect --use_bn --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gen --dataset amazon-photo --ood_type structure --mode detect --use_bn --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gen --dataset amazon-photo --ood_type structure --mode detect --use_bn --use_prop --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gen --dataset amazon-photo --ood_type structure --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 
python main.py --method gnnsafe --backbone gen --dataset amazon-photo --ood_type structure --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01


### Amazon-photo with feature ood
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type feature --mode detect --use_bn --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type feature --mode detect --use_bn --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type feature --mode detect --use_bn --use_prop --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type feature --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01

python main.py --method gnnsafe --backbone gen --dataset amazon-photo --ood_type feature --mode detect --use_bn --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gen --dataset amazon-photo --ood_type feature --mode detect --use_bn --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gen --dataset amazon-photo --ood_type feature --mode detect --use_bn --use_prop --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gen --dataset amazon-photo --ood_type feature --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0
python main.py --method gnnsafe --backbone gen --dataset amazon-photo --ood_type feature --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01


### Amazon-photo with label ood

python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type label --mode detect --use_bn --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type label --mode detect --use_bn --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type label --mode detect --use_bn --use_prop --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type label --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01

python main.py --method gnnsafe --backbone gen --dataset amazon-photo --ood_type label --mode detect --use_bn --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gen --dataset amazon-photo --ood_type label --mode detect --use_bn --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gen --dataset amazon-photo --ood_type label --mode detect --use_bn --use_prop --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gen --dataset amazon-photo --ood_type label --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0
python main.py --method gnnsafe --backbone gen --dataset amazon-photo --ood_type label --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01


### Coauthor with structure ood

python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type structure --mode detect --use_bn --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type structure --mode detect --use_bn --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type structure --mode detect --use_bn --use_prop --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type structure --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01

python main.py --method gnnsafe --backbone gen --dataset coauthor-cs --ood_type structure --mode detect --use_bn --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gen --dataset coauthor-cs --ood_type structure --mode detect --use_bn --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gen --dataset coauthor-cs --ood_type structure --mode detect --use_bn --use_prop --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gen --dataset coauthor-cs --ood_type structure --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0
python main.py --method gnnsafe --backbone gen --dataset coauthor-cs --ood_type structure --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01


### Coauthor with feature ood
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type feature --mode detect --use_bn --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type feature --mode detect --use_bn --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type feature --mode detect --use_bn --use_prop --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type feature --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01

python main.py --method gnnsafe --backbone gen --dataset coauthor-cs --ood_type feature --mode detect --use_bn --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gen --dataset coauthor-cs --ood_type feature --mode detect --use_bn --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gen --dataset coauthor-cs --ood_type feature --mode detect --use_bn --use_prop --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gen --dataset coauthor-cs --ood_type feature --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0
python main.py --method gnnsafe --backbone gen --dataset coauthor-cs --ood_type feature --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01


### Coauthor with label ood


python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type label --mode detect --use_bn --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type label --mode detect --use_bn --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type label --mode detect --use_bn --use_prop --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type label --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01

python main.py --method gnnsafe --backbone gen --dataset coauthor-cs --ood_type label --mode detect --use_bn --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gen --dataset coauthor-cs --ood_type label --mode detect --use_bn --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gen --dataset coauthor-cs --ood_type label --mode detect --use_bn --use_prop --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01
python main.py --method gnnsafe --backbone gen --dataset coauthor-cs --ood_type label --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0
python main.py --method gnnsafe --backbone gen --dataset coauthor-cs --ood_type label --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda1 0.01 --device 0 --use_oc True --lamda2 0.01  --nu 0.1 --eps 0.01


#### Twitch

#python main.py --method msp --backbone gcn --dataset twitch --mode detect --use_bn --device 0
#python main.py --method gnnsafe --backbone gcn --dataset twitch --mode detect --use_bn --device 0
#python main.py --method gnnsafe --backbone gcn --dataset twitch --mode detect --use_bn --use_reg --m_in -7 --m_out -2 --lamda 0.1 --device 0
#python main.py --method gnnsafe --backbone gcn --dataset twitch --mode detect --use_bn --use_prop --device 0
#python main.py --method gnnsafe --backbone gcn --dataset twitch --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda 0.1 --device 0

#### Arxiv

#python main.py --method msp --backbone gcn --dataset arxiv --mode detect --use_bn --device 0
#python main.py --method gnnsafe --backbone gcn --dataset arxiv --mode detect --use_bn --device 0
#python main.py --method gnnsafe --backbone gcn --dataset arxiv --mode detect --use_bn --use_reg --m_in -9 --m_out -4 --lamda 0.01 --device 0
#python main.py --method gnnsafe --backbone gcn --dataset arxiv --mode detect --use_bn --use_prop --device 0
#python main.py --method gnnsafe --backbone gcn --dataset arxiv --mode detect --use_bn --use_prop --use_reg --m_in -9 --m_out -2 --lamda 0.01 --device 0
