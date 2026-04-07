# phasegrad

**Phase-Gradient Duality: backpropagation-free learning in coupled oscillator networks.**

[Paper (PDF)](paper/main.pdf)

In a Kuramoto oscillator network at equilibrium, the physical phase response to weak output clamping equals the gradient of the loss with respect to natural frequencies. This enables a learning rule that requires no separate backward pass — two forward passes of the same physics compute both the prediction and the gradient.

## Quick start: verify the theorem

```python
from phasegrad import run_verification

results = run_verification(sizes=[6, 10, 15, 20])
# All cosine similarities = 1.000000
```

## Quick start: train a vowel classifier

```python
from phasegrad import KuramotoNetwork, train
from phasegrad.kuramoto import make_network
from phasegrad.data import load_hillenbrand

train_data, test_data, info = load_hillenbrand(vowels=['a', 'i'])
net = make_network(n_input=2, n_hidden=5, n_output=2)
history = train(net, train_data, test_data, epochs=200)
```

## Installation

```bash
pip install -e .
```

Requires Python 3.10+, NumPy, SciPy, scikit-learn, matplotlib.

## The theorem

For a network of coupled Kuramoto oscillators with dynamics

$$\frac{d\theta_i}{dt} = \omega_i + \sum_j K_{ij} \sin(\theta_j - \theta_i)$$

at a stable equilibrium $\theta^\*$, the gradient of a quadratic output loss with respect to natural frequencies can be computed physically:

$$\lim_{\beta \to 0} \frac{\theta_k^\beta - \theta_k^\*}{\beta} = -\frac{\partial L}{\partial \omega_k}$$

where $\theta^\beta$ is the equilibrium under weak output clamping with strength $\beta$.

The proof uses the implicit function theorem: the equilibrium defines $\theta^\*(\omega)$ implicitly, and the Jacobian of the equilibrium equations (a coupling-weighted graph Laplacian) mediates the gradient flow.

## Key results

| Result | Value |
|---|---|
| Gradient identity (cosine similarity) | 1.000000 at all N tested (6–200) |
| Equilibrium residuals | At or below machine epsilon |
| Natural freq vs coupling (converged, matched params) | 96.0% vs 83.3% (p < 10⁻¹²) |
| Spectral seeding success rate | 100/100 (vs 46/100 random init) |

## Repository structure

```
phasegrad/       Library: kuramoto, gradient, training, verification
tests/           7 test files (gradient identity, training, robustness, scaling)
experiments/     All experiment scripts + reproducible JSON results
paper/           LaTeX source, compiled PDF, figures
examples/        Quick-start scripts
data/            Hillenbrand vowel formant dataset
```

## Running the tests

```bash
pip install -e ".[dev]"
pytest tests/
```

## Generating paper figures

```bash
python paper/generate_figures.py
# Outputs to paper/figures/
```

## Citation

```
Rashahmadi, M. (2026). The Gradient Was Already There: Exact Frequency
Gradients in Coupled Oscillator Networks. Calibur Labs.
```

## License

MIT
