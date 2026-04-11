"""Training loop for Kuramoto networks using equilibrium propagation."""

from __future__ import annotations

import time

import numpy as np

from phasegrad.kuramoto import KuramotoNetwork
from phasegrad.gradient import two_phase_gradient
from phasegrad.losses import mse_loss, mse_target


def train(net: KuramotoNetwork,
          train_data: list[tuple[np.ndarray, int]],
          test_data: list[tuple[np.ndarray, int]],
          lr_omega: float = 0.001,
          lr_K: float = 0.001,
          beta: float = 0.1,
          epochs: int = 200,
          margin: float = 0.2,
          grad_clip: float = 2.0,
          omega_bounds: tuple[float, float] = (-3.0, 3.0),
          K_bounds: tuple[float, float] = (0.01, 8.0),
          seed: int = 42,
          verbose: bool = True,
          eval_every: int = 10,
          ) -> list[dict]:
    """Train a Kuramoto network on classification using two-phase gradients.

    For each sample:
      1. Set input frequencies from features.
      2. Find free equilibrium θ*.
      3. Find clamped equilibrium θ^β (outputs nudged toward target).
      4. Update ω and K using the two-phase learning rule.

    Args:
        net: the oscillator network (modified in place).
        train_data: list of (features, class_idx).
        test_data: list of (features, class_idx).
        lr_omega: learning rate for natural frequencies.
        lr_K: learning rate for coupling strengths.
        beta: clamping strength for equilibrium propagation.
        epochs: number of passes through the training data.
        margin: phase margin for MSE targets (correct at -margin, wrong at +margin).
        grad_clip: maximum gradient magnitude (per-parameter clipping).
        omega_bounds: (min, max) clipping bounds for natural frequencies.
        K_bounds: (min, max) clipping bounds for coupling strengths.
        seed: random seed for shuffling.
        verbose: print progress.
        eval_every: evaluate on test set every N epochs.

    Returns:
        List of dicts with training history (one per evaluation point).
    """
    rng = np.random.default_rng(seed)
    history = []

    # Initial evaluation
    ev = _evaluate(net, test_data, margin)
    if verbose:
        print(f"  Init: loss={ev['loss']:.4f} acc={ev['acc']:.1%} sep={ev['separation']:.4f}")
    history.append({"epoch": 0, **ev, "train_loss": None, "train_acc": None,
                    "grad_cos": None, "skip": 0})

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, n_skip = _train_epoch(
            net, train_data, beta, lr_omega, lr_K, margin, grad_clip,
            omega_bounds, K_bounds, rng)

        if epoch % eval_every == 0 or epoch == 1 or epoch == epochs:
            ev = _evaluate(net, test_data, margin)

            # Gradient verification on a random training sample
            sample = train_data[rng.integers(len(train_data))]
            gs = _verify_one(net, sample, beta, margin)

            elapsed = time.time() - t0
            if verbose:
                print(f"  Ep {epoch:3d}: loss={tr_loss:.4f} tr={tr_acc:.1%} "
                      f"test={ev['acc']:.1%} sep={ev['separation']:.4f} "
                      f"gcos={gs:+.4f} skip={n_skip} ({elapsed:.0f}s)")

            history.append({"epoch": epoch, **ev, "train_loss": tr_loss,
                           "train_acc": tr_acc, "grad_cos": gs, "skip": n_skip})

    return history


def _train_epoch(net, data, beta, lr_omega, lr_K, margin, grad_clip,
                 omega_bounds, K_bounds, rng):
    """One pass through the training data with two-phase updates."""
    indices = rng.permutation(len(data))
    total_loss = 0.0
    correct = n = n_skip = 0
    theta_warm = None

    for idx in indices:
        x, cls = data[idx]
        net.set_input(x)
        target = mse_target(net.N, net.output_ids, cls, margin)

        theta_free, res_free = net.equilibrium(theta_warm)
        if res_free > 0.01:
            n_skip += 1
            continue

        theta_clamp, res_clamp = net.clamped_equilibrium(
            beta, target, theta_init=theta_free.copy())

        loss = mse_loss(theta_free, target, net.output_ids)
        pred = net.classify(theta_free)
        total_loss += loss
        correct += (pred == cls)
        n += 1

        if res_clamp < 0.01:
            _apply_update(net, theta_free, theta_clamp, beta, lr_omega, lr_K,
                          grad_clip, omega_bounds, K_bounds)
        else:
            n_skip += 1

        theta_warm = theta_free

    return (total_loss / max(n, 1), correct / max(n, 1), n_skip)


def _apply_update(net, theta_free, theta_clamp, beta, lr_omega, lr_K,
                  grad_clip, omega_bounds, K_bounds):
    """Apply the two-phase learning rule."""
    # Frequency updates
    for i in net.learnable_ids:
        g = -(theta_clamp[i] - theta_free[i]) / beta
        g = np.clip(g, -grad_clip, grad_clip)
        net.omega[i] -= lr_omega * g

    # Coupling updates
    for (i, j) in net.edges:
        cos_free = np.cos(theta_free[j] - theta_free[i])
        cos_clamp = np.cos(theta_clamp[j] - theta_clamp[i])
        g = (cos_free - cos_clamp) / beta
        g = np.clip(g, -grad_clip, grad_clip)
        net.K[i, j] -= lr_K * g
        net.K[j, i] = net.K[i, j]

    # Physical bounds: clip existing edges only, don't create new ones
    for i in net.learnable_ids:
        net.omega[i] = np.clip(net.omega[i], omega_bounds[0], omega_bounds[1])
    existing = net.K > 0
    net.K = np.where(existing, np.clip(net.K, K_bounds[0], K_bounds[1]), 0.0)


def _evaluate(net, data, margin):
    """Evaluate network on a dataset."""
    total_loss = 0.0
    correct = n = 0
    scores_by_class: dict[int, list[float]] = {}
    theta_warm = None

    for x, cls in data:
        net.set_input(x)
        theta, res = net.equilibrium(theta_warm)
        if res > 0.1:
            continue

        target = mse_target(net.N, net.output_ids, cls, margin)
        loss = mse_loss(theta, target, net.output_ids)
        pred = net.classify(theta)

        total_loss += loss
        correct += (pred == cls)
        n += 1

        # Track output phase difference as score
        out_phases = [theta[o] for o in net.output_ids]
        score = out_phases[0] - out_phases[-1] if len(out_phases) >= 2 else 0.0
        scores_by_class.setdefault(cls, []).append(score)
        theta_warm = theta

    # Compute class separation
    separation = 0.0
    class_keys = sorted(scores_by_class.keys())
    if len(class_keys) >= 2:
        means = [np.mean(scores_by_class[c]) for c in class_keys]
        stds = [np.std(scores_by_class[c]) + 1e-10 for c in class_keys]
        separation = abs(means[0] - means[-1]) / np.mean(stds)

    return {
        "loss": total_loss / max(n, 1),
        "acc": correct / max(n, 1),
        "separation": separation,
        "n_eval": n,
    }


def _verify_one(net, sample, beta, margin):
    """Gradient cosine on a single sample."""
    x, cls = sample
    net.set_input(x)
    target = mse_target(net.N, net.output_ids, cls, margin)

    theta_free, _ = net.equilibrium()
    theta_clamp, _ = net.clamped_equilibrium(beta, target, theta_free.copy())

    grad_tp = np.array([-(theta_clamp[i] - theta_free[i]) / beta
                         for i in net.learnable_ids])

    eps = 1e-5
    omega_c = net.omega_centered.copy()
    grad_fd = np.zeros(len(net.learnable_ids))
    for k, i in enumerate(net.learnable_ids):
        oc_plus = omega_c.copy()
        oc_plus[i] += eps
        th_p, _ = net.equilibrium(theta_free.copy(), omega_c=oc_plus)
        L_p = mse_loss(th_p, target, net.output_ids)

        oc_minus = omega_c.copy()
        oc_minus[i] -= eps
        th_m, _ = net.equilibrium(theta_free.copy(), omega_c=oc_minus)
        L_m = mse_loss(th_m, target, net.output_ids)
        grad_fd[k] = (L_p - L_m) / (2 * eps)

    ntp, nfd = np.linalg.norm(grad_tp), np.linalg.norm(grad_fd)
    if ntp < 1e-12 or nfd < 1e-12:
        return 0.0
    return float(np.dot(grad_tp, grad_fd) / (ntp * nfd))
