#!/usr/bin/env python3.12
"""C1: Convergence failure diagnosis.

Track Jacobian condition number, equilibrium residual, and skip rate
per epoch for 40 seeds. Correlate with convergence/failure.
"""
import json, time, numpy as np
from phasegrad.kuramoto import make_network, kuramoto_jacobian
from phasegrad.data import load_hillenbrand
from phasegrad.losses import mse_loss, mse_target

N_SEEDS = 40
EPOCHS = 150
EVAL_EVERY = 10

def diagnose_training(seed):
    """Train one seed, recording diagnostics at each eval point."""
    from phasegrad.training import _train_epoch, _evaluate
    tr, te, _ = load_hillenbrand(vowels=['a', 'i'], seed=seed)
    net = make_network(n_input=2, n_hidden=5, n_output=2,
                       K_scale=2.0, input_scale=1.5, seed=seed)
    rng = np.random.default_rng(seed)

    records = []
    # Initial state
    sample_x, sample_cls = tr[0]
    net.set_input(sample_x)
    theta0, res0 = net.equilibrium()
    J0 = kuramoto_jacobian(theta0, net.K)
    J0_red = J0[1:, 1:]
    cond0 = float(np.linalg.cond(J0_red))
    ev0 = _evaluate(net, te, margin=0.2)
    records.append({
        'epoch': 0, 'cond': cond0, 'residual': float(res0),
        'test_acc': ev0['acc'], 'test_loss': ev0['loss'],
        'skip': 0, 'min_K': float(net.K[net.K > 0].min()),
        'max_omega_spread': float(np.ptp(net.omega)),
    })

    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc, n_skip = _train_epoch(
            net, tr, beta=0.1, lr_omega=0.001, lr_K=0.001,
            margin=0.2, grad_clip=2.0, rng=rng)

        if ep % EVAL_EVERY == 0 or ep == EPOCHS:
            net.set_input(sample_x)
            theta, res = net.equilibrium()
            J = kuramoto_jacobian(theta, net.K)
            J_red = J[1:, 1:]
            try:
                cond = float(np.linalg.cond(J_red))
            except:
                cond = 1e15
            ev = _evaluate(net, te, margin=0.2)
            records.append({
                'epoch': ep, 'cond': cond, 'residual': float(res),
                'test_acc': ev['acc'], 'test_loss': ev['loss'],
                'train_acc': tr_acc, 'train_loss': tr_loss,
                'skip': n_skip, 'min_K': float(net.K[net.K > 0].min()),
                'max_omega_spread': float(np.ptp(net.omega)),
            })

    final_acc = records[-1]['test_acc']
    converged = final_acc > 0.60
    return {'seed': seed, 'converged': converged, 'final_acc': final_acc,
            'records': records}

if __name__ == '__main__':
    print(f"Convergence Diagnosis: {N_SEEDS} seeds, {EPOCHS} epochs")
    all_results = []
    t0 = time.time()

    for seed in range(N_SEEDS):
        r = diagnose_training(seed)
        all_results.append(r)
        status = 'CONV' if r['converged'] else 'FAIL'
        final_cond = r['records'][-1]['cond']
        final_skip = r['records'][-1]['skip']
        print(f"  seed {seed:2d}: {status} acc={r['final_acc']:.1%} "
              f"cond={final_cond:.1e} skip={final_skip} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # Analysis
    conv = [r for r in all_results if r['converged']]
    fail = [r for r in all_results if not r['converged']]
    print(f"\nConverged: {len(conv)}/{N_SEEDS}")

    # Condition number at init for converged vs failed
    conv_cond0 = [r['records'][0]['cond'] for r in conv]
    fail_cond0 = [r['records'][0]['cond'] for r in fail]
    if conv_cond0 and fail_cond0:
        print(f"\nInit condition number:")
        print(f"  Converged: {np.mean(conv_cond0):.1f} ± {np.std(conv_cond0):.1f}")
        print(f"  Failed:    {np.mean(fail_cond0):.1f} ± {np.std(fail_cond0):.1f}")

    # Final condition number
    conv_cond_final = [r['records'][-1]['cond'] for r in conv]
    fail_cond_final = [r['records'][-1]['cond'] for r in fail]
    if conv_cond_final and fail_cond_final:
        print(f"\nFinal condition number:")
        print(f"  Converged: {np.mean(conv_cond_final):.1f} ± {np.std(conv_cond_final):.1f}")
        print(f"  Failed:    {np.mean(fail_cond_final):.1f} ± {np.std(fail_cond_final):.1f}")

    # Skip rate at final epoch
    conv_skip = [r['records'][-1]['skip'] for r in conv]
    fail_skip = [r['records'][-1]['skip'] for r in fail]
    if conv_skip and fail_skip:
        print(f"\nFinal skip rate:")
        print(f"  Converged: {np.mean(conv_skip):.1f}")
        print(f"  Failed:    {np.mean(fail_skip):.1f}")

    # Min K during training
    conv_minK = [min(rec['min_K'] for rec in r['records']) for r in conv]
    fail_minK = [min(rec['min_K'] for rec in r['records']) for r in fail]
    if conv_minK and fail_minK:
        print(f"\nMin coupling during training:")
        print(f"  Converged: {np.mean(conv_minK):.3f}")
        print(f"  Failed:    {np.mean(fail_minK):.3f}")

    out_path = 'experiments/convergence_diagnosis.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
