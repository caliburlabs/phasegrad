#!/usr/bin/env python3.12
"""E4: Training dynamics — per-epoch loss/accuracy for converged vs failed seeds."""
import json, numpy as np
from phasegrad.kuramoto import make_network
from phasegrad.data import load_hillenbrand
from phasegrad.training import train

# Use seeds we know converge/fail from prior experiments
CONV_SEEDS = [0, 2, 3, 6, 7, 9, 11, 13, 17, 19]  # from stabilization data
FAIL_SEEDS = [1, 4, 5, 8, 10, 12, 14, 15, 16]      # first 9 failed seeds
EPOCHS = 200

if __name__ == '__main__':
    print(f"E4: Training curves ({len(CONV_SEEDS)} converged, {len(FAIL_SEEDS)} failed)")
    all_curves = []

    for label, seeds in [('converged', CONV_SEEDS), ('failed', FAIL_SEEDS)]:
        for seed in seeds:
            tr, te, _ = load_hillenbrand(vowels=['a', 'i'], seed=seed)
            net = make_network(seed=seed, K_scale=2.0, input_scale=1.5)
            hist = train(net, tr, te, lr_omega=0.001, lr_K=0.001, beta=0.1,
                         epochs=EPOCHS, verbose=False, eval_every=5, seed=seed)

            curve = {
                'seed': seed, 'group': label,
                'epochs': [h['epoch'] for h in hist],
                'test_loss': [h['loss'] for h in hist],
                'test_acc': [h['acc'] for h in hist],
                'train_loss': [h.get('train_loss') for h in hist],
                'train_acc': [h.get('train_acc') for h in hist],
            }
            all_curves.append(curve)
            final = hist[-1]
            print(f"  [{label[:4]}] seed {seed:2d}: "
                  f"loss={final['loss']:.4f} acc={final['acc']:.1%}",
                  flush=True)

    # Summary: do failed seeds plateau early or descend to wrong minimum?
    conv_final_loss = [c['test_loss'][-1] for c in all_curves if c['group'] == 'converged']
    fail_final_loss = [c['test_loss'][-1] for c in all_curves if c['group'] == 'failed']
    conv_init_loss = [c['test_loss'][0] for c in all_curves if c['group'] == 'converged']
    fail_init_loss = [c['test_loss'][0] for c in all_curves if c['group'] == 'failed']

    print(f"\n  Converged: init_loss={np.mean(conv_init_loss):.4f} → final_loss={np.mean(conv_final_loss):.4f}")
    print(f"  Failed:    init_loss={np.mean(fail_init_loss):.4f} → final_loss={np.mean(fail_final_loss):.4f}")

    loss_drop_conv = np.mean(conv_init_loss) - np.mean(conv_final_loss)
    loss_drop_fail = np.mean(fail_init_loss) - np.mean(fail_final_loss)
    print(f"  Converged loss drop: {loss_drop_conv:.4f}")
    print(f"  Failed loss drop:    {loss_drop_fail:.4f}")
    if loss_drop_fail > 0.001:
        print(f"  → Failed seeds DO decrease loss (descend to wrong minimum)")
    else:
        print(f"  → Failed seeds DON'T decrease loss (stuck from the start)")

    with open('experiments/training_curves.json', 'w') as f:
        json.dump(all_curves, f, indent=2, default=str)
    print(f"\nSaved to experiments/training_curves.json")
