#!/usr/bin/env python3.12
"""Oscillator as activation function in a digital neural network.

The architecture:
  Input → W1 (digital matmul) → Oscillator (physics settling) → W2 (digital matmul) → Loss

The oscillator IS the activation function. No lookup table. No ROM.
No piecewise approximation. The nonlinearity is the physics (sin coupling).
The gradient is exact (EP/analytical, not straight-through estimator).

Compared against:
  - ReLU activation (standard)
  - tanh activation (saturating)
  - Linear (no activation — the failure case)

Task: sklearn digits (8x8 images, 10 classes, PCA to 8 dimensions)
"""
import json, time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from phasegrad.kuramoto import KuramotoNetwork, kuramoto_jacobian


# ── Oscillator activation ────────────────────────────────────────

def make_oscillator_layer(N, K_coupling=1.5, topology='chain', seed=42):
    """Create a fixed coupling matrix for the oscillator activation layer."""
    rng = np.random.default_rng(seed)
    K = np.zeros((N, N))
    if topology == 'chain':
        for i in range(N - 1):
            K[i, i + 1] = K_coupling
            K[i + 1, i] = K_coupling
    elif topology == 'random_sparse':
        for i in range(N):
            for j in range(i + 1, N):
                if rng.random() < 0.4:
                    v = K_coupling * rng.uniform(0.5, 1.5)
                    K[i, j] = K[j, i] = v
    elif topology == 'alltoall':
        for i in range(N):
            for j in range(i + 1, N):
                K[i, j] = K[j, i] = K_coupling
    return K


def oscillator_forward(z, K_fixed, scale=1.0):
    """Forward pass through oscillator activation.

    Input: z (N,) — pre-activations from digital matmul
    Output: theta_star (N,) — equilibrium phases (the activated values)

    The oscillator frequencies are set to z * scale. The coupling K is fixed.
    The equilibrium phases are the activated values.

    The activation is tanh-like:
    - Small z: θ ∝ z (linear, like identity)
    - Large z: θ saturates at ±π/2 (like tanh saturation)
    - The coupling K controls the saturation threshold
    """
    N = len(z)
    omega = z.copy() * scale
    net = KuramotoNetwork(
        omega=omega, K=K_fixed.copy(),
        input_ids=[], output_ids=list(range(N)),
    )
    theta, res = net.equilibrium()
    return theta, res, net


def oscillator_backward(net, theta_star, dL_dtheta):
    """Backward pass through oscillator activation via analytical gradient.

    Given ∂L/∂θ* (the upstream gradient), compute ∂L/∂ω (gradient
    w.r.t. frequencies = gradient w.r.t. pre-activations z).

    Uses: ∂L/∂ω = -J^{-T} × ∂L/∂θ* (implicit function theorem)
    """
    N = net.N
    J_full = kuramoto_jacobian(theta_star, net.K)
    J_red = J_full[1:, 1:]

    # ∂L/∂θ reduced (skip pinned node)
    dL_dtheta_red = dL_dtheta[1:]

    try:
        x = np.linalg.solve(J_red.T, dL_dtheta_red)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(J_red.T, dL_dtheta_red, rcond=None)[0]

    dL_domega = np.zeros(N)
    dL_domega[1:] = -x
    return dL_domega


# ── Standard activations ─────────────────────────────────────────

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

def tanh_act(z):
    return np.tanh(z)

def tanh_grad(z):
    return 1 - np.tanh(z)**2

def linear_act(z):
    return z

def linear_grad(z):
    return np.ones_like(z)


# ── 2-layer network ──────────────────────────────────────────────

class TwoLayerNet:
    """Input → W1 → activation → W2 → softmax → loss."""

    def __init__(self, n_in, n_hidden, n_out, activation='relu',
                 K_coupling=3.0, topology='chain', seed=42):
        rng = np.random.default_rng(seed)
        # Smaller init for oscillator to stay within lock range
        scale1 = 0.3 if activation == 'oscillator' else np.sqrt(2.0 / n_in)
        scale2 = np.sqrt(2.0 / n_hidden)
        self.W1 = rng.standard_normal((n_hidden, n_in)) * scale1
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.standard_normal((n_out, n_hidden)) * scale2
        self.b2 = np.zeros(n_out)
        self.activation = activation
        self.K_fixed = None
        self.osc_scale = 1.0
        if activation == 'oscillator':
            self.K_fixed = make_oscillator_layer(n_hidden, K_coupling,
                                                  topology, seed)
            # Scale factor: keep z within ±0.8 * K * min_degree
            min_degree = 1 if topology == 'chain' else 2  # end nodes have degree 1
            self.osc_scale = 0.8 * K_coupling * (min_degree + 1)

    def forward(self, x):
        """Forward pass. Returns (logits, cache)."""
        z1 = self.W1 @ x + self.b1

        if self.activation == 'oscillator':
            a1, res, net = oscillator_forward(z1, self.K_fixed)
            cache = {'x': x, 'z1': z1, 'a1': a1, 'net': net, 'theta': a1,
                     'res': res}
        elif self.activation == 'relu':
            a1 = relu(z1)
            cache = {'x': x, 'z1': z1, 'a1': a1}
        elif self.activation == 'tanh':
            a1 = tanh_act(z1)
            cache = {'x': x, 'z1': z1, 'a1': a1}
        else:  # linear
            a1 = z1
            cache = {'x': x, 'z1': z1, 'a1': a1}

        logits = self.W2 @ a1 + self.b2
        cache['logits'] = logits
        return logits, cache

    def backward(self, cache, y_true):
        """Backward pass. Returns gradients and loss."""
        logits = cache['logits']
        a1 = cache['a1']
        x = cache['x']
        z1 = cache['z1']

        # Softmax + cross-entropy loss
        logits_shifted = logits - logits.max()
        probs = np.exp(logits_shifted) / np.exp(logits_shifted).sum()
        loss = -np.log(probs[y_true] + 1e-10)

        # ∂L/∂logits
        dL_dlogits = probs.copy()
        dL_dlogits[y_true] -= 1.0

        # ∂L/∂W2, ∂L/∂b2
        dL_dW2 = np.outer(dL_dlogits, a1)
        dL_db2 = dL_dlogits

        # ∂L/∂a1
        dL_da1 = self.W2.T @ dL_dlogits

        # ∂L/∂z1 (through activation)
        if self.activation == 'oscillator':
            dL_dz1 = oscillator_backward(cache['net'], cache['theta'], dL_da1)
        elif self.activation == 'relu':
            dL_dz1 = dL_da1 * relu_grad(z1)
        elif self.activation == 'tanh':
            dL_dz1 = dL_da1 * tanh_grad(z1)
        else:
            dL_dz1 = dL_da1

        # ∂L/∂W1, ∂L/∂b1
        dL_dW1 = np.outer(dL_dz1, x)
        dL_db1 = dL_dz1

        return loss, {'W1': dL_dW1, 'b1': dL_db1, 'W2': dL_dW2, 'b2': dL_db2}

    def update(self, grads, lr):
        self.W1 -= lr * grads['W1']
        self.b1 -= lr * grads['b1']
        self.W2 -= lr * grads['W2']
        self.b2 -= lr * grads['b2']

    def predict(self, X):
        preds = []
        for x in X:
            logits, _ = self.forward(x)
            preds.append(np.argmax(logits))
        return np.array(preds)


# ── Training loop ────────────────────────────────────────────────

def train_and_eval(activation, n_hidden, X_train, y_train, X_test, y_test,
                   n_epochs=200, lr=0.01, K_coupling=1.5, topology='chain',
                   seed=42):
    n_in = X_train.shape[1]
    n_out = len(np.unique(y_train))

    net = TwoLayerNet(n_in, n_hidden, n_out, activation,
                       K_coupling, topology, seed)

    history = []
    rng = np.random.default_rng(seed)
    n_skip = 0

    for epoch in range(n_epochs):
        indices = rng.permutation(len(X_train))
        total_loss = 0
        n_samples = 0

        for idx in indices:
            x, y = X_train[idx], y_train[idx]
            logits, cache = net.forward(x)

            # Skip if oscillator didn't converge
            if activation == 'oscillator' and cache.get('res', 0) > 0.1:
                n_skip += 1
                continue

            loss, grads = net.backward(cache, y)

            # Clip gradients
            for key in grads:
                grads[key] = np.clip(grads[key], -5.0, 5.0)

            net.update(grads, lr)
            total_loss += loss
            n_samples += 1

        if (epoch + 1) % 50 == 0 or epoch == 0:
            train_preds = net.predict(X_train)
            test_preds = net.predict(X_test)
            train_acc = np.mean(train_preds == y_train)
            test_acc = np.mean(test_preds == y_test)
            avg_loss = total_loss / max(n_samples, 1)
            history.append({
                'epoch': epoch + 1, 'loss': avg_loss,
                'train_acc': train_acc, 'test_acc': test_acc,
            })
            print(f"    Ep {epoch+1:3d}: loss={avg_loss:.3f} "
                  f"train={train_acc:.1%} test={test_acc:.1%}"
                  f"{f' skip={n_skip}' if n_skip else ''}", flush=True)

    final_test = np.mean(net.predict(X_test) == y_test)
    return final_test, history


# ── Main ─────────────────────────────────────────────────────────

def main():
    print("Oscillator as Activation Function")
    print("=" * 65)

    # Load data
    digits = load_digits()
    X, y = digits.data, digits.target
    X = X / 16.0  # normalize to [0, 1]

    # PCA to 8 dimensions
    n_components = 8
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    # Standardize
    X_pca = (X_pca - X_pca.mean(axis=0)) / (X_pca.std(axis=0) + 1e-8)

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y)

    print(f"  Data: {X_train.shape[0]} train, {X_test.shape[0]} test, "
          f"{n_components} features, 10 classes")
    print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.1%}")

    n_hidden = 8
    results = {}

    # Sweep activations
    configs = [
        ('linear', {'activation': 'linear'}),
        ('relu', {'activation': 'relu'}),
        ('tanh', {'activation': 'tanh'}),
        ('osc_chain_K3', {'activation': 'oscillator', 'K_coupling': 3.0,
                           'topology': 'chain'}),
        ('osc_chain_K5', {'activation': 'oscillator', 'K_coupling': 5.0,
                           'topology': 'chain'}),
        ('osc_sparse_K3', {'activation': 'oscillator', 'K_coupling': 3.0,
                            'topology': 'random_sparse'}),
        ('osc_alltoall_K5', {'activation': 'oscillator', 'K_coupling': 5.0,
                              'topology': 'alltoall'}),
    ]

    for name, kwargs in configs:
        print(f"\n{'─'*65}")
        print(f"  Activation: {name}")
        t0 = time.time()
        acc, hist = train_and_eval(
            n_hidden=n_hidden, X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test, n_epochs=200, lr=0.01,
            seed=42, **kwargs)
        dt = time.time() - t0
        results[name] = {
            'test_acc': acc, 'history': hist, 'time_s': dt,
        }
        print(f"  Final test acc: {acc:.1%} ({dt:.0f}s)")

    # Summary
    print(f"\n{'='*65}")
    print("RESULTS")
    print(f"{'='*65}")
    print(f"\n{'Activation':<30} {'Test Acc':>10} {'Time':>8}")
    print("─" * 50)
    for name, r in sorted(results.items(), key=lambda x: -x[1]['test_acc']):
        print(f"{name:<30} {r['test_acc']:>9.1%} {r['time_s']:>7.0f}s")

    # Key comparison
    linear_acc = results.get('linear', {}).get('test_acc', 0)
    relu_acc = results.get('relu', {}).get('test_acc', 0)
    best_osc_name = max([n for n in results if 'osc' in n],
                        key=lambda n: results[n]['test_acc'])
    best_osc_acc = results[best_osc_name]['test_acc']

    print(f"\n  Linear (no activation): {linear_acc:.1%}")
    print(f"  ReLU:                   {relu_acc:.1%}")
    print(f"  Best oscillator:        {best_osc_acc:.1%} ({best_osc_name})")

    if best_osc_acc > linear_acc + 0.01:
        print(f"\n  Oscillator BEATS linear by {best_osc_acc - linear_acc:.1%} → "
              f"the nonlinearity helps!")
    else:
        print(f"\n  Oscillator ≈ linear → nonlinearity not significant")

    if best_osc_acc >= relu_acc - 0.05:
        print(f"  Oscillator matches ReLU (within 5%) → viable activation!")
    else:
        print(f"  Oscillator {relu_acc - best_osc_acc:.1%} behind ReLU")

    results['summary'] = {
        'linear_acc': linear_acc, 'relu_acc': relu_acc,
        'best_osc_acc': best_osc_acc, 'best_osc_name': best_osc_name,
        'beats_linear': best_osc_acc > linear_acc + 0.01,
        'matches_relu': best_osc_acc >= relu_acc - 0.05,
    }

    with open('experiments/oscillator_activation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to experiments/oscillator_activation_results.json")

    return results


if __name__ == '__main__':
    main()
