# Self-Supervised Learning with Kernel Dependence Maximization

This is the code for SSL-HSIC, a self-supervised learning loss proposed
in the paper Self-Supervised Learning with Kernel Dependence Maximization
(https://arxiv.org/abs/2106.08320).

Using this implementation should achieve a top-1 accuracy on Imagenet around
74.8% using 128 Cloud TPU v2/3.

## Installation

To set up a Python3 virtual environment with the required dependencies, run:

```bash
python3 -m venv ssl_hsic_env
source ssl_hsic_env/bin/activate
pip install --upgrade pip
pip install -r ssl_hsic/requirements.txt
```

## Usage

### Pre-training

For pre-training on ImageNet with SSL-HSIC loss:

```bash
mkdir /tmp/ssl_hsic
python3 -m ssl_hsic.experiment \
--config=ssl_hsic/config.py:default \
--jaxline_mode=train
```

This is going to pre-train for 1000 epochs. Change config to `config.py:test`
for testing purpose. See
[jaxline documentation](https://github.com/deepmind/jaxline) for more
information on jaxline_mode.

If `save_dir` is provided in `config.py`, the last checkpoint is saved and can
be used for evaluation.

### Linear Evaluation

For linear evaluation with the saved checkpoint:

```bash
mkdir /tmp/ssl_hsic
python3 -m ssl_hsic.eval_experiment \
--config=ssl_hsic/eval_config.py:default \
--jaxline_mode=train
```

This is going to train a linear layer for 90 epochs. Change config to
`eval_config.py:test` for testing.

## Citing this work

If you use this code in your work, please consider referencing our work:

```
@inproceedings{
  li2021selfsupervised,
  title={Self-Supervised Learning with Kernel Dependence Maximization},
  author={Yazhe Li and Roman Pogodin and Danica J. Sutherland and Arthur Gretton},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021},
  url={https://openreview.net/forum?id=0HW7A5YZjq7}
}
```

## Disclaimer

This is not an official Google product.
