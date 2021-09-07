# LinkTeller

This repository contains the official implementation of the S&P 22 paper

"[LinkTeller: Recovering Private Edges from Graph Neural Networks via Influence Analysis](https://arxiv.org/abs/2108.06504)"

Fan Wu, Yunhui Long, Ce Zhang, Bo Li

## Download and Installation

1. The code requires Python >=3.6 and is built on PyTorch. Note that PyTorch may need to be [installed manually](https://pytorch.org/get-started/locally/) depending on different platforms and CUDA drivers.
2. The graph datasets can be obtained from the [Google Drive link](https://drive.google.com/file/d/1_TV_XNy0ljy_KWli30k_n4Kcj3pgwzPx/view?usp=sharing). Please download the file ``data.zip`` and uncompress it under the root path.

## Usage

We will use the twitch datasets as an example in the following.

### Evaluation of LinkTeller

In this part, we introduce how to use LinkTeller attack to reveal the private edges given access to a blackbox API.

1. Train a vanilla GCN

```bash
python main.py --mode vanilla-clean  --dataset twitch/ES/RU --hidden 256 \
--display --num-epochs 200 --dropout 0.5 --lr 0.01 --norm FirstOrderGCN
```

The concrete explanations for each **option**, the **choices** of each option, and the **optimal hyper-parameters** or configurations for each dataset can all be referred to in the Appendix F of the paper.

Additionally, the trained model can be tested via the following script:

```bash
python main.py --mode vanilla-clean  --dataset twitch/ES/RU --hidden 256 \
--display --num-epochs 200 --dropout 0.5 --lr 0.01 --norm FirstOrderGCN \
--test --model-path [model_path]
```

where the ``[model_path]`` can be obtained from the training log in the previous run.

2. Attack the trained GCN model

```bash
python main.py --mode vanilla-clean  --dataset twitch/ES/RU --hidden 256 \
--display --num-epochs 200 --dropout 0.5 --lr 0.01 --norm FirstOrderGCN \
--test --model-path [model_path] \
--attack --approx --sample-type unbalanced --n-test 500 --influence 0.0001 \
--sample-seed 42 --attack-mode efficient
```

Basically, given a trained model, we configure the following options of the attack

* **sample-type**: { 'unbalanced', 'unbalanced-lo', 'unbalanced-hi' }, where 'unbalanced' means random node sampling among all nodes in the inference graph, 'unbalanced-lo' means sampling from low degree nodes, and 'unbalanced-hi' means sampling from high degree nodes.
* **n-test**: the size of the node set of interest ($V^{(C)}$ In the paper)
* **sample-seed:** the preset random seed for node sampling
* **attack-mode**: { 'efficient', 'baseline', 'baseline-feat' }, where 'efficient' represents our LinkTeller attack, 'baseline' represents the baseline method LSA2-post, and 'baseline-feat' represents the baseline method LSA2-attr.

For ease of experimentation, training and attack can be merged into one stage. To do so, simply remove the options ``--test`` and ``--model-path`` in the attack script.

The attack results will be saved to the local path as indicated in the log.

### Evaluation of Differentially Private GCNs

In this part, we describe the evaluation of Differentially Private (DP) GCNs from two perspectives: utility and privacy.

1. Train a DP GCN model

```bash
python main.py --mode vanilla  --dataset twitch/ES/RU --hidden 256 \
--num-epochs 200 --dropout 0.5 --lr 0.01 --norm FirstOrderGCN \
--perturb-type continuous --eps 5 --noise-seed 42 \
```

We briefly introduce the privacy parameters:

* **perturb-type**: { 'continuous', 'discrete' }, where 'continuous' refers to the method Lapgraph and 'discrete' refers to the method EdgeRand
* **eps**: the privacy budget in differential privacy
* **noise-seed**: the random seed for noise generation

2. Measure the **utility** of the trained DP GCN model (i.e., test the trained model)

```bash
python main.py --mode vanilla  --dataset twitch/ES/RU --hidden 256 \
--num-epochs 200 --dropout 0.5 --lr 0.01 --norm FirstOrderGCN \
--perturb-type continuous --eps 5 --noise-seed 42 \
--test --model-path [model_path]
```

3. Measure the **privacy** of the trained DP GCN model via LinkTeller

```bash
python main.py --mode vanilla  --dataset twitch/ES/RU --hidden 256 \
--num-epochs 200 --dropout 0.5 --lr 0.01 --norm FirstOrderGCN \
--perturb-type continuous --eps 5 --noise-seed 42 \
--test --model-path [model_path] \
--attack --approx --sample-type unbalanced --n-test 500 --influence 0.0001 \
--sample-seed 42 --attack-mode efficient
```

The options for attack are the same as previously introduced in the evaluation of LinkTeller.

## Citation

To be updated