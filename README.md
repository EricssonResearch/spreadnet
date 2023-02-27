# SpreadNet

Uppsala Project CS 2022

## Introduction

Graph Neural Networks(GNN) have made waves in the recent years. In this project we have compared the performance of different models against that of Dijkstra.

Applying GNN to problems where good algorithms already exist is not going to lead to a breakthrough. However, using synthetic data generated by those algorithms gives us a good idea of what patterns they learn and what are the possible use cases. Because the models we have used approach the shortest path problem in a graph from a classification perspective, they predict nodes and edges part of the path. As a result looking at the prediction accuracy can be misleading as even a 99% accuracy can be an incomplete path. There are scenarios where the models predict alternative paths that are similar in length. If we want to consider those solutions we need different methods to calculate the accuracy. We have wrote a number of variations titled "Maximum Probability Walk" where we try to construct a path by talking the maximum probability choice available for each node starting from the source node. This was seen as a good algorithm as the source and target nodes are easy to classify given their unique attributes. Additionally, we have observed that the model propagates the probability from the source and target nodes.

## Installation

```
pip install torch==1.13 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
grep -v "spreadnet" /requirements.txt > tmpfile && mv tmpfile /requirements.txt
pip install -r /requirements.txt
pip install -e .
```

## Functionalities

- Generate synthetic data using Dijkstra, `experiments/generate_dataset.py`
- Train a variety of Graph Neural Networks models, `experiments/train.py`
- Evaluate them using Ecology related metrics
- More accurate path prediction metrics
- Memory consumption metrics
- Dijkstra comparison metrics
- Different plotting metrics that highlight different aspects of the Predictions, `experiments/predict.py`
- Query Processor, `experiments/qp.py`

## Key Points

The models approach the shortest path problem in a classification manner. They predict the nodes part of the path. This approach while easy to implement is suboptimal unless the loss function will consider sets of nodes that represent actual paths.

If we view the shortest path problem as a classification one we need to take into account the inherent class imbalance. There will always be fewer nodes part of the path than outside of the path.

## Future Work

- Deal with the inherent class imbalance.
- Constructing a path is not the same as predicting nodes and edges part of the path. Reusing the models with a loss function that takes the validity of a path into account is necessary.
- Create a loss function that takes into account alternative paths.
- Explore GNN models that predict the next node until they reach the end node. These models would guarantee that a path exists.
- Further explore the hybrid Dijkstra and GNN models.

## Contributors
ChanVuth Chea
Boli Gao
Jennifer Gross
Ishita Jaju
Haouyuan Li
Akanksha Makkar
Mishkuat Sabir
Paarth Sanhotra
Gaurav Singh
George Stoian
Sofia Afnaan Syed
Haodong Zhao

## License
[Apache License 2.0](LICENSE)<br />