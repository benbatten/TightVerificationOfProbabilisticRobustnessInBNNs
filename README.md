# Tight Verification of Probabilistic Robustness in Bayesian Neural Networks

#### This file explain how to set up the environment and run the experiments in our paper on the verification of Beysian Neural Networks (BNNs), accepted in AISTATS 2024. The paper can be be accessed at [arxiv.org/abs/2401.11627](https://arxiv.org/abs/2401.11627).

## Setting Up

1. install the requirements from the pipfile with `pipenv install`
2. Activate pipenv with `pipenv shell`
3. navigate to `verifier`



## Examples

Below are some examples on how to run the experiments.

#### To run the Vanilla Iterative Expansion algorithm on MLP 1x128 on MNIST run:

```bash
python3 combined_verifier.py --net_name wick_MNIST_1_128 --implementation ours --dynamic_grad_ratio 0
```

#### To run the Gradient-guided Iterative Expansion algorithm on MLP 1x128 on MNIST run:

```bash
python3 combined_verifier.py --net_name wick_MNIST_1_128 --implementation ours
```

#### To run the Wicker et al. algorithm on MLP 1x128 on MNIST run:
```bash
python3 combined_verifier.py --net_name wick_MNIST_1_128 --implementation wicker --grad_stepsize 10
```

#### To run our method on CIFAR10 CNN run:
```bash
python3 combined_verifier.py --net_name CIFAR10 --epsilon 0.0
```

#### For citing the paper or the tool, please use
```
@inproceedings{BattenHL24,
  title     = {Tight Verification of Probabilistic Robustness in Bayesian Neural Networks},
  author    = {Batten, Ben and Hosseini, Mehran and Lomuscio, Alessio},
  booktitle = {The 27th International Conference on Artificial Intelligence and Statistics, {AISTATS}},
  year      = {2024}
}
```
