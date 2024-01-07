#  Out-of-Distribution Detection for Graph Data

## Dependence

- Ubuntu 16.04.6
- Cuda 10.2
- Pytorch 1.9.0
- Pytorch Geometric 2.0.3

More information about required packages is listed in `requirements.txt`.

## Problem Settings

Out-of-distribution (OOD) detection on graph-structured data: Given a set of training nodes (inter-connected as a graph), 
one needs to train a robust classifier that can effectively identify OOD nodes (that have disparate distributions than training nodes) in the test set.
In the meanwhile, the classifier should maintain decent classification performance on in-distribution testing nodes. Different from image data, OOD detection on graph data needs to handle data inter-dependence as compared below.

<img width="600" alt="image" src="https://user-images.githubusercontent.com/22075007/219937529-f7d57dbc-ca9d-445f-ae27-f8c244cf9158.png">

OOD detection often has two specific problem settings, which we introduce in the following figures in comparison with standard supervised learning.

<img width="800" alt="image" src="https://user-images.githubusercontent.com/22075007/219937584-6627f89e-803f-49e6-b3ce-553db7529806.png">

- ***Supervised learning:*** the training and testing are based on data from the same distribution, i.e., in-distribution (IND) data. We use IND-Tr/IND-Val/IND-Te to denote the train/valid/test sets of in-distribution data.

- ***OOD detection w/o exposure***: the training is based on pure IND-Tr, and the model is evaluated by the performance of discriminating IND-Te and out-of-distribution (OOD) data in test set (short as OOD-Te).

- ***OOD detection w/ OOD exposure***: besides IND-Tr, the training stage is exposed to extra OOD data (short as OOD-Tr),
and the model is evaluated on OOD-Te and IND-Te.


## Data Splits and Protocols

For comprehensive evaluation, we introduce new benchmarks for OOD detection on graphs, with regard to distribution shifts of real-world and synthetic settings. Generally, graph datasets can be divided into single-graph and multi-graph datasets, and we follow the principles in [1] for data splits as shown below.

<img width="700" alt="image" src="https://user-images.githubusercontent.com/22075007/219937890-d0739791-8e5b-4dda-b4ea-8f5653728b10.png">


- **Cora/Amazon/Coauthor** (single-graph dataset w/o context info): Each of these datasets contain one single graph and no explicit domain label is given. We use the original data as IND, and follow the public splits for train/valid/test partition.
As for OOD data, we modified the original dataset to obtain OODTr and OODTe, with three different ways:

    - Structure manipulation: adopt stochastic block model to randomly generate a graph for OOD data.
    - Feature interpolation: use random interpolation to create node features for OOD data. 
    - Label leave-out: use nodes with partial classes as IND and leave out others for OODTr and OODTe.

***Evalution Metrics***: the OOD detection performance is measured by AUROC, AUPR, FPR95 for discriminating IND-Te and OOD-Te.




## Reference

 
```bibtex
  @inproceedings{wu2023gnnsafe,
  title = {Energy-based Out-of-Distribution Detection for Graph Neural Networks},
  author = {Qitian Wu and Yiting Chen and Chenxiao Yang and Junchi Yan},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2023}
  }
```

[1] Qitian Wu et al., Handling Distribution Shifts on Graphs: An Invariance Perspective. In ICLR2022.


