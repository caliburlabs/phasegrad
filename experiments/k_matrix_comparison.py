"""K-matrix structural comparison: lucky vs unlucky XOR basins.

Pure inspection of on-disk outcomes — no training, no new experiments.
Regenerates K deterministically for all 23 seeds with known XOR accuracy,
then computes block-level features the prior analysis missed.

Block structure of the 2+5+2 network:
  K_ih: input→hidden (2×5, all-to-all)
  K_hh: hidden↔hidden (5×5, chain only — tridiagonal)
  K_ho: hidden→output (5×2, all-to-all)

Prior analysis covered: Laplacian spectrum, seeding weights, effective resistance.
This script covers: raw K block statistics, per-hidden-node through-flow,
path products, coupling asymmetries, bottleneck analysis.
"""

import numpy as np
import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phasegrad.kuramoto import make_network


# ── Seeds with known training outcomes ──────────────────────────────

TRAINED_SEEDS = {
    # (seed, xor_acc)
    1: 0.935, 24: 0.4925, 84: 0.500,
    63: 0.5225, 94: 0.5125, 18: 0.475, 89: 0.5075,
    82: 0.6675, 71: 0.540, 13: 0.5025, 7: 0.4925,
    54: 0.5325, 50: 0.5975, 83: 0.475, 8: 0.4925,
    45: 0.5125, 73: 0.4825, 9: 0.5025, 68: 0.545,
    75: 0.820, 37: 0.4825, 61: 0.5125, 48: 0.540,
}

LUCKY_SEEDS = {s for s, a in TRAINED_SEEDS.items() if a > 0.75}
UNLUCKY_SEEDS = {s for s, a in TRAINED_SEEDS.items() if a <= 0.75}


def extract_blocks(K, input_ids, hidden_ids, output_ids):
    """Extract the three coupling blocks from the full K matrix."""
    K_ih = K[np.ix_(input_ids, hidden_ids)]   # (2, 5)
    K_hh_full = K[np.ix_(hidden_ids, hidden_ids)]  # (5, 5) — sparse chain
    K_ho = K[np.ix_(hidden_ids, output_ids)]   # (5, 2)
    return K_ih, K_hh_full, K_ho


def per_hidden_features(K_ih, K_hh, K_ho, hidden_ids):
    """Compute per-hidden-node structural features.

    For each hidden node h:
      - in_coupling[h]: sum of input→h couplings (how strongly driven by inputs)
      - out_coupling[h]: sum of h→output couplings (how strongly drives outputs)
      - through_flow[h]: in_coupling * out_coupling (proxy for signal throughput)
      - hh_degree[h]: sum of hidden↔hidden couplings (lateral connectivity)
      - in_asymmetry[h]: |K[in0,h] - K[in1,h]| / (K[in0,h] + K[in1,h])
        (how asymmetrically the two inputs couple to this hidden node)
      - out_asymmetry[h]: |K[h,out0] - K[h,out1]| / (K[h,out0] + K[h,out1])
    """
    n_hidden = len(hidden_ids)
    features = []
    for h in range(n_hidden):
        in_coup = K_ih[:, h].sum()          # total input coupling
        out_coup = K_ho[h, :].sum()          # total output coupling
        hh_deg = K_hh[h, :].sum()            # hidden-hidden degree
        through = in_coup * out_coup         # throughput proxy

        # Input asymmetry: how differently the two inputs couple to this node
        in_vals = K_ih[:, h]
        in_asym = abs(in_vals[0] - in_vals[1]) / (in_vals.sum() + 1e-12)

        # Output asymmetry: how differently this node couples to the two outputs
        out_vals = K_ho[h, :]
        out_asym = abs(out_vals[0] - out_vals[1]) / (out_vals.sum() + 1e-12)

        features.append({
            'hidden_idx': h,
            'in_coupling': float(in_coup),
            'out_coupling': float(out_coup),
            'through_flow': float(through),
            'hh_degree': float(hh_deg),
            'in_asymmetry': float(in_asym),
            'out_asymmetry': float(out_asym),
            'in_vals': in_vals.tolist(),
            'out_vals': out_vals.tolist(),
        })
    return features


def path_products(K_ih, K_ho):
    """Compute all input→hidden→output path products.

    For each (input_i, hidden_h, output_o) triple:
      path_weight = K[input_i, hidden_h] * K[hidden_h, output_o]

    Returns (2, 5, 2) array of path weights.
    Also computes per-path "XOR discriminability":
      For XOR, we need output_0 ≠ output_1 when input_0 ≠ input_1.
      The differential path: sum over h of (K[in0,h] - K[in1,h]) * (K[h,out0] - K[h,out1])
      captures how much the coupling structure supports input-dependent output separation.
    """
    # (2, 5) @ (5, 2) doesn't capture per-hidden contribution
    # Compute element-wise
    n_in, n_hid = K_ih.shape
    _, n_out = K_ho.shape

    paths = np.zeros((n_in, n_hid, n_out))
    for i in range(n_in):
        for h in range(n_hid):
            for o in range(n_out):
                paths[i, h, o] = K_ih[i, h] * K_ho[h, o]

    # XOR discriminability: does the graph's coupling structure naturally
    # create different paths for "same input" vs "different input" patterns?
    # For each hidden node h:
    #   input_diff[h] = K[in0,h] - K[in1,h]  (sensitivity to input difference)
    #   output_diff[h] = K[h,out0] - K[h,out1]  (ability to split outputs)
    #   xor_score[h] = input_diff * output_diff

    input_diff = K_ih[0, :] - K_ih[1, :]     # (5,)
    output_diff = K_ho[:, 0] - K_ho[:, 1]     # (5,)

    per_h_xor_score = input_diff * output_diff  # (5,)
    total_xor_score = per_h_xor_score.sum()

    return paths, per_h_xor_score, total_xor_score


def graph_features(seed):
    """Full structural feature set for a single seed."""
    net = make_network(n_input=2, n_hidden=5, n_output=2,
                       K_scale=2.0, seed=42 + seed)
    K = net.K
    input_ids = net.input_ids
    hidden_ids = list(range(2, 7))
    output_ids = net.output_ids

    K_ih, K_hh, K_ho = extract_blocks(K, input_ids, hidden_ids, output_ids)

    # ── Block-level statistics ──
    block_stats = {
        'K_ih_mean': float(K_ih.mean()),
        'K_ih_std': float(K_ih.std()),
        'K_ih_min': float(K_ih.min()),
        'K_ih_max': float(K_ih.max()),
        'K_ih_sum': float(K_ih.sum()),
        'K_ho_mean': float(K_ho.mean()),
        'K_ho_std': float(K_ho.std()),
        'K_ho_min': float(K_ho.min()),
        'K_ho_max': float(K_ho.max()),
        'K_ho_sum': float(K_ho.sum()),
        'K_hh_sum': float(K_hh.sum()),  # total hidden-hidden coupling
        'K_hh_max': float(K_hh.max()),
        # Ratios
        'ih_ho_ratio': float(K_ih.sum() / (K_ho.sum() + 1e-12)),
        'hh_fraction': float(K_hh.sum() / (K.sum() + 1e-12)),
    }

    # ── Per-hidden-node features ──
    hidden_feats = per_hidden_features(K_ih, K_hh, K_ho, hidden_ids)

    # ── Aggregate hidden-node statistics ──
    through_flows = np.array([f['through_flow'] for f in hidden_feats])
    in_asyms = np.array([f['in_asymmetry'] for f in hidden_feats])
    out_asyms = np.array([f['out_asymmetry'] for f in hidden_feats])
    in_coups = np.array([f['in_coupling'] for f in hidden_feats])
    out_coups = np.array([f['out_coupling'] for f in hidden_feats])

    aggregate = {
        # Through-flow distribution
        'through_flow_mean': float(through_flows.mean()),
        'through_flow_std': float(through_flows.std()),
        'through_flow_max': float(through_flows.max()),
        'through_flow_min': float(through_flows.min()),
        'through_flow_range': float(through_flows.max() - through_flows.min()),
        'through_flow_cv': float(through_flows.std() / (through_flows.mean() + 1e-12)),
        'through_flow_gini': float(_gini(through_flows)),
        # Input coupling distribution
        'in_coupling_std': float(in_coups.std()),
        'in_coupling_cv': float(in_coups.std() / (in_coups.mean() + 1e-12)),
        # Output coupling distribution
        'out_coupling_std': float(out_coups.std()),
        'out_coupling_cv': float(out_coups.std() / (out_coups.mean() + 1e-12)),
        # Asymmetry statistics
        'in_asymmetry_mean': float(in_asyms.mean()),
        'in_asymmetry_max': float(in_asyms.max()),
        'out_asymmetry_mean': float(out_asyms.mean()),
        'out_asymmetry_max': float(out_asyms.max()),
        # Cross-asymmetry: correlation between input and output asymmetry
        'in_out_asym_corr': float(np.corrcoef(in_asyms, out_asyms)[0, 1])
            if in_asyms.std() > 1e-10 and out_asyms.std() > 1e-10 else 0.0,
    }

    # ── Path products and XOR discriminability ──
    paths, per_h_xor, total_xor = path_products(K_ih, K_ho)

    path_feats = {
        'xor_discriminability': float(total_xor),
        'xor_discriminability_abs': float(np.abs(per_h_xor).sum()),
        'per_h_xor_scores': per_h_xor.tolist(),
        'xor_score_std': float(per_h_xor.std()),
        'xor_score_max_abs': float(np.abs(per_h_xor).max()),
        'n_positive_xor_paths': int((per_h_xor > 0).sum()),
        'n_negative_xor_paths': int((per_h_xor < 0).sum()),
        # Path strength statistics
        'path_mean': float(paths.mean()),
        'path_std': float(paths.std()),
        'path_max': float(paths.max()),
        'path_min': float(paths.min()),
        # Max single path vs rest
        'path_dominance': float(paths.max() / (paths.mean() + 1e-12)),
    }

    # ── Effective rank / condition of coupling blocks ──
    # SVD of K_ih and K_ho to check rank/condition
    s_ih = np.linalg.svd(K_ih, compute_uv=False)
    s_ho = np.linalg.svd(K_ho, compute_uv=False)

    svd_feats = {
        'K_ih_sv': s_ih.tolist(),
        'K_ih_condition': float(s_ih[0] / (s_ih[-1] + 1e-12)),
        'K_ih_rank_ratio': float(s_ih[-1] / (s_ih[0] + 1e-12)),
        'K_ho_sv': s_ho.tolist(),
        'K_ho_condition': float(s_ho[0] / (s_ho[-1] + 1e-12)),
        'K_ho_rank_ratio': float(s_ho[-1] / (s_ho[0] + 1e-12)),
    }

    # ── Compound metric: combined input-output path diversity ──
    # The product K_ih.T @ K_ih captures input-side correlations among hidden nodes
    # The product K_ho @ K_ho.T captures output-side correlations
    # Their interaction tells us about end-to-end path diversity
    input_gram = K_ih.T @ K_ih   # (5, 5) — how similar hidden nodes look from input side
    output_gram = K_ho @ K_ho.T  # (5, 5) — how similar hidden nodes look from output side

    # Frobenius inner product of the two grams: high means hidden nodes
    # that are similar on input side are also similar on output side (bad for XOR)
    gram_alignment = float(np.sum(input_gram * output_gram))
    # Normalize
    gram_align_norm = gram_alignment / (
        np.linalg.norm(input_gram, 'fro') * np.linalg.norm(output_gram, 'fro') + 1e-12)

    compound = {
        'gram_alignment': gram_alignment,
        'gram_alignment_normalized': float(gram_align_norm),
    }

    # ── Hidden chain topology ──
    # Chain coupling values (4 edges for 5 hidden nodes)
    chain_couplings = []
    for i in range(4):
        chain_couplings.append(float(K_hh[i, i+1]))

    chain_feats = {
        'chain_couplings': chain_couplings,
        'chain_mean': float(np.mean(chain_couplings)),
        'chain_std': float(np.std(chain_couplings)),
        'chain_min': float(np.min(chain_couplings)),
        'chain_max': float(np.max(chain_couplings)),
        'chain_range': float(np.max(chain_couplings) - np.min(chain_couplings)),
    }

    return {
        'seed': seed,
        'xor_acc': TRAINED_SEEDS[seed],
        'lucky': seed in LUCKY_SEEDS,
        'block_stats': block_stats,
        'hidden_feats': hidden_feats,
        'aggregate': aggregate,
        'path_feats': path_feats,
        'svd_feats': svd_feats,
        'compound': compound,
        'chain_feats': chain_feats,
        'K_ih': K_ih.tolist(),
        'K_ho': K_ho.tolist(),
        'K_hh': K_hh.tolist(),
    }


def _gini(x):
    """Gini coefficient of array x."""
    x = np.abs(x)
    if x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * x) - (n + 1) * np.sum(x)) / (n * np.sum(x)))


def main():
    print("=" * 78)
    print("K-MATRIX STRUCTURAL COMPARISON: lucky vs unlucky XOR basins")
    print("=" * 78)
    print(f"\nLucky seeds: {sorted(LUCKY_SEEDS)} (acc > 0.75)")
    print(f"Unlucky seeds: {sorted(UNLUCKY_SEEDS)} ({len(UNLUCKY_SEEDS)} seeds)")

    # Compute features for all seeds
    all_feats = []
    for seed in sorted(TRAINED_SEEDS.keys()):
        all_feats.append(graph_features(seed))

    lucky = [f for f in all_feats if f['lucky']]
    unlucky = [f for f in all_feats if not f['lucky']]

    # ══════════════════════════════════════════════════════════
    # Feature-by-feature comparison
    # ══════════════════════════════════════════════════════════

    def flat_scalar_features(feat_dict):
        """Extract all scalar features into a flat dict."""
        flat = {'seed': feat_dict['seed'], 'xor_acc': feat_dict['xor_acc']}
        for section in ['block_stats', 'aggregate', 'path_feats', 'svd_feats',
                        'compound', 'chain_feats']:
            for k, v in feat_dict[section].items():
                if isinstance(v, (int, float)):
                    flat[f"{section}.{k}"] = v
        return flat

    all_flat = [flat_scalar_features(f) for f in all_feats]
    lucky_flat = [f for f in all_flat if f['seed'] in LUCKY_SEEDS]
    unlucky_flat = [f for f in all_flat if f['seed'] not in LUCKY_SEEDS]

    # Get all feature names
    feature_names = [k for k in all_flat[0].keys()
                     if k not in ('seed', 'xor_acc')]

    print("\n" + "=" * 78)
    print("FEATURE COMPARISON: lucky mean vs unlucky mean")
    print("=" * 78)
    print(f"\n{'feature':>50} {'lucky':>10} {'unlucky':>10} {'diff':>10} {'z-score':>10}")
    print("-" * 95)

    separations = []
    for fname in feature_names:
        lvals = np.array([f[fname] for f in lucky_flat])
        uvals = np.array([f[fname] for f in unlucky_flat])
        lmean = lvals.mean()
        umean = uvals.mean()
        ustd = uvals.std()
        diff = lmean - umean
        z = diff / (ustd + 1e-12) if ustd > 1e-10 else 0.0
        separations.append((abs(z), fname, lmean, umean, diff, z))

    # Sort by absolute z-score descending
    separations.sort(key=lambda x: -x[0])

    for absz, fname, lm, um, diff, z in separations:
        flag = ''
        if absz > 2.0:
            flag = ' ***'
        elif absz > 1.5:
            flag = ' **'
        elif absz > 1.0:
            flag = ' *'
        print(f"{fname:>50} {lm:>10.4f} {um:>10.4f} {diff:>+10.4f} {z:>+10.2f}{flag}")

    # ══════════════════════════════════════════════════════════
    # Per-hidden-node comparison for lucky seeds
    # ══════════════════════════════════════════════════════════

    print("\n" + "=" * 78)
    print("PER-HIDDEN-NODE ANALYSIS FOR LUCKY SEEDS")
    print("=" * 78)

    for f in lucky:
        print(f"\n--- Seed {f['seed']} (acc={f['xor_acc']:.3f}) ---")
        print(f"{'node':>6} {'in_coup':>10} {'out_coup':>10} {'through':>10} "
              f"{'hh_deg':>10} {'in_asym':>10} {'out_asym':>10} {'xor_score':>10}")
        for i, hf in enumerate(f['hidden_feats']):
            xor_s = f['path_feats']['per_h_xor_scores'][i]
            print(f"{hf['hidden_idx']:>6} {hf['in_coupling']:>10.3f} "
                  f"{hf['out_coupling']:>10.3f} {hf['through_flow']:>10.3f} "
                  f"{hf['hh_degree']:>10.3f} {hf['in_asymmetry']:>10.3f} "
                  f"{hf['out_asymmetry']:>10.3f} {xor_s:>10.3f}")
        print(f"  Total XOR discriminability: {f['path_feats']['xor_discriminability']:.4f}")
        print(f"  Chain couplings: {f['chain_feats']['chain_couplings']}")
        print(f"  K_ih:\n    {np.array(f['K_ih'])}")
        print(f"  K_ho:\n    {np.array(f['K_ho'])}")

    # ══════════════════════════════════════════════════════════
    # Top discriminating features — detailed breakdown
    # ══════════════════════════════════════════════════════════

    print("\n" + "=" * 78)
    print("TOP 10 DISCRIMINATING FEATURES — seed-level values")
    print("=" * 78)

    for rank, (absz, fname, lm, um, diff, z) in enumerate(separations[:10], 1):
        print(f"\n#{rank}: {fname} (z={z:+.2f})")
        vals_with_seed = [(f['seed'], f[fname], f['xor_acc'])
                          for f in all_flat]
        vals_with_seed.sort(key=lambda x: x[1])
        for seed, val, acc in vals_with_seed:
            tag = " << LUCKY" if seed in LUCKY_SEEDS else ""
            print(f"  seed {seed:3d}: {val:>10.4f}  (acc={acc:.3f}){tag}")

    # ══════════════════════════════════════════════════════════
    # XOR discriminability deep-dive
    # ══════════════════════════════════════════════════════════

    print("\n" + "=" * 78)
    print("XOR DISCRIMINABILITY: per-hidden-node scores for ALL seeds")
    print("=" * 78)

    print(f"\n{'seed':>5} {'acc':>7} {'total':>10} "
          + " ".join(f"{'h'+str(i):>8}" for i in range(5))
          + f" {'#pos':>5} {'#neg':>5}")
    for f in sorted(all_feats, key=lambda x: -x['xor_acc']):
        scores = f['path_feats']['per_h_xor_scores']
        total = f['path_feats']['xor_discriminability']
        npos = sum(1 for s in scores if s > 0)
        nneg = sum(1 for s in scores if s < 0)
        tag = " *" if f['lucky'] else "  "
        score_str = " ".join(f"{s:>8.3f}" for s in scores)
        print(f"{f['seed']:>5}{tag} {f['xor_acc']:>6.3f} {total:>10.4f} "
              f"{score_str} {npos:>5} {nneg:>5}")

    # ══════════════════════════════════════════════════════════
    # Correlation of all features with XOR accuracy
    # ══════════════════════════════════════════════════════════

    print("\n" + "=" * 78)
    print("CORRELATION WITH XOR ACCURACY (all 23 seeds)")
    print("=" * 78)

    accs = np.array([f['xor_acc'] for f in all_flat])
    correlations = []
    for fname in feature_names:
        vals = np.array([f[fname] for f in all_flat])
        if vals.std() > 1e-10:
            r = np.corrcoef(vals, accs)[0, 1]
            correlations.append((abs(r), fname, r))

    correlations.sort(key=lambda x: -x[0])
    print(f"\n{'feature':>50} {'r':>10}")
    print("-" * 65)
    for absr, fname, r in correlations:
        flag = ''
        if absr > 0.5:
            flag = ' ***'
        elif absr > 0.3:
            flag = ' **'
        elif absr > 0.2:
            flag = ' *'
        print(f"{fname:>50} {r:>+10.3f}{flag}")

    # Save results
    outpath = os.path.join(os.path.dirname(__file__),
                           'k_matrix_comparison_results.json')
    with open(outpath, 'w') as f:
        json.dump({
            'features': all_feats,
            'separations': [
                {'feature': fname, 'lucky_mean': lm, 'unlucky_mean': um,
                 'diff': diff, 'z_score': z}
                for absz, fname, lm, um, diff, z in separations
            ],
            'correlations': [
                {'feature': fname, 'r': r}
                for absr, fname, r in correlations
            ],
        }, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")


if __name__ == '__main__':
    main()
