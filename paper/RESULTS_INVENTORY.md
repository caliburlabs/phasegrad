# Results Inventory — Complete

Everything we built, tested, and learned. Organized by verdict.
Updated 2026-03-30 after all audits (including training energy and power gating).

---

## PROVEN (paper-ready, survived all audits)

### P1. Phase-gradient identity theorem
- **What:** Phase displacement under weak clamping = exact gradient of loss w.r.t. natural frequency
- **Evidence:** cos = 1.000000 for N = 6, 10, 15, 20, 30, 50, 100, 200
- **Extends to:** Forced Kuramoto (65/65 converged points, cos > 0.99999998, Jacobian asymmetry = 0.0)
- **Audit:** PASS (never challenged)
- **Files:** `phasegrad/experiments/scale_results.json`, `forced_gradient_verification_extended.json`

### P2. ω outperforms K (sparse architectures)
- **What:** Learning natural frequency beats learning coupling weights
- **Evidence:**
  - Vowels a/i: ω 91.9% vs K 82.2% (20 seeds, p=0.004)
  - Vowels o/u: ω 75.1% vs K 71.9% (20 seeds)
  - 100-seed replication: ω 93.5% vs K 83.0% (p < 10⁻⁹)
  - Symmetric lr sweep: ω dominates K at ALL 5 learning rates tested
  - Best ω: 96.6% at lr=0.01, 15/20 converge
  - Best K: 82.2% at lr=0.001, 9/20 converge
  - FM task: ω 73.4% (10/20 converge), K 52.2% (1/20 converge, dead)
- **Scope:** Sparse layered architectures (2+5+2). NOT universal — K works on all-to-all (Wang 94.1%, Rageau 97.8%)
- **Audit:** PASS (ablation bit-exact reproducible, lr sweep closes fairness concern)
- **Files:** `ablation_results.json`, `ablation_ou_results.json`, `fm_raw_results.json`, `ADVERSARIAL_FIXES.md` §6

### P3. Convergence failure = multistability
- **What:** ~50% of seeds fail, but failure is loss-landscape topology, not gradient breakdown
- **Evidence:**
  - cond(J) identical in converged vs failed seeds (15.2 vs 15.4)
  - Skip rate = 0 for all seeds (equilibrium always found)
  - Same 11/20 seeds converge across 5 training variants (Cochran's Q = 0)
  - Training curves: both groups decrease loss identically, failed seeds converge to degenerate minimum
  - No standard diagnostic (connectivity, spectral gap, gradient norm) predicts failure at init (best p=0.11)
- **Audit:** PASS
- **Files:** `convergence_diagnosis.json`, `stabilization_results.json`, `convergence_matrix_results.json`

### P4. Asymmetry robustness
- **What:** Gradient identity degrades gracefully under coupling asymmetry
- **Evidence:** cos > 0.995 at 20% asymmetry, cos > 0.999 at 5% (N=15, 10 seeds/level)
- **Implication:** CMOS parasitic asymmetries (<5%) are negligible
- **Audit:** PASS
- **Files:** `asymmetry_results.json`

### P5. Two-phase = FD training equivalence
- **What:** Two-phase gradient produces identical training dynamics to finite-difference
- **Evidence:** Mean accuracy gap 0.7% ± 0.9%, max 1.8%, 6/10 seeds exactly 0.0% gap
- **Audit:** PASS
- **Files:** `gradient_validation_clean_results.json`

---

## QUALIFIED (real but needs careful framing)

### Q1. SPICE hardware bridge
- **What:** Kuramoto model matches SPICE ring oscillator physics
- **Evidence:**
  - VDD-to-frequency calibration: R² = 0.999994 (17 points, 1.4–2.2V → 1.7–4.0 GHz, cubic fit)
  - Effective coupling: K = 1.357 rad/ns at coupling strength 0.9
  - Gradient validation: 5 stable configs, Kuramoto cos = 1.0, SPICE-vs-FD cos = 0.9957, phase correlation = 0.9011
- **Caveats:**
  - Only 5 of many configs converge — prefiltered to stable subset
  - gradient_validation.py drops weak cases (line 122) and prunes for phase correlation (line 154)
  - Paper must say "five stable SPICE configurations" not "broad hardware validation"
- **Audit:** QUALIFIED (filtering is honest if disclosed)
- **Files:** `paper/hardware/freq_calibration.json`, `k_calibration.json`, `spice_gradient_validation.json`

### Q2. One-step SPICE learning
- **What:** Single gradient step in SPICE improves classification
- **Evidence:** Loss 1.485 → 1.095 (−26%), accuracy 55% → 70% (+15%) after one VDD update
- **Caveats:**
  - Trivial 2-band dataset
  - Only one hidden VDD updated
  - Peaks at step 3, regresses at step 4 (not a stable learning loop)
  - one_step_learning.py line 38: toy task, line 124: single parameter
- **Honest framing:** "proof-of-concept for hardware-guided one-step improvement"
- **Audit:** QUALIFIED
- **Files:** `paper/hardware/spice_one_step_learning.json`

### Q3. Concurrent layer settling
- **What:** Multiple oscillator layers settle simultaneously, zero latency cost for depth
- **Evidence:**
  - 2-layer: 72 transistors, 86.5 pJ, both layers settled at 40 ns
  - 3-layer: 14 oscillators, L3 phase drift < 0.02° between 30-60 ns
  - Different inputs produce different L2/L3 outputs
- **Caveats:**
  - Verified for ≤3 layers only
  - Requires direct inter-layer coupling (not chain between layers)
  - Small phase regime (<0.7 rad) — linearity of sin coupling
  - Scaling beyond 3 layers is untested
- **Audit:** CONFIRMED with scope limits
- **Files:** `kb/exploration/substrate/matvec_2layer_results.json`, `matvec_3layer_results.json`

### Q4. Density advantage (10-20× per layer, grows with depth)
- **What:** Oscillator layers use fewer transistors than digital equivalent
- **Raw numbers:** 48 oscillator transistors vs 39,630 digital = 825×
- **After audit:** 448 (with DACs) vs 4,500 (hand-optimized 8-bit digital) = 10×
- **Depth scaling:** 24 transistors per oscillator layer vs ~1,800 per digital layer → converges to 75× asymptotically
- **Caveats:**
  - Capacitor area is 66% of total silicon area (not counted in transistor number)
  - DACs needed for input encoding (240-1,600 transistors)
  - Precision is ~7 bits (oscillator) vs 8-12 bits (digital)
  - The "matrix" is fixed by topology — no reprogramming
- **Honest claim:** "10× fewer transistors at 1 layer, growing to 31× at 10 layers, including DAC overhead"
- **Files:** `paper/MATVEC_COMPARISON.md`, `MATVEC_AUDIT.md`, `MATVEC_2LAYER.md`, `THREE_CLAIMS_AUDIT.md`

### Q5. Training energy scaling curves cross (with caveats)
- **What:** At sufficient N with all-to-all coupling, oscillator training energy (O(N)) beats digital (O(N²))
- **Measured at N=8:** Oscillator 120 pJ training vs digital 28.54 pJ — digital wins 4.2×
- **Power gating tested:** 60.08 pJ gated vs 57.2 pJ ungated — gating is 5% WORSE (PMOS overhead). The energy IS the oscillation; cannot be gated during computation.
- **Scaling projections (from SPICE + synthesis measurements):**
  - Chain coupling: oscillator O(N³) — never competitive
  - All-to-all 10 fF: crossover ~N≈150 (marginal)
  - All-to-all 1 fF: crossover ~N≈25 (4× win at N=128) — **1 fF coupling unvalidated in SPICE**
  - Matched VDD (0.9V) + 1 fF: crossover ~N≈10 — **0.9V oscillation unvalidated, likely won't start**
- **Critical caveat (from TRAINING_ENERGY_AUDIT.md Attack 7):** The comparison measures energy per equilibrium computation, NOT energy per usable gradient. TDC noise requires 100+ averages per gradient, potentially making true training energy 100× higher (12,000+ pJ).
- **Additional missing costs:** DAC for weight update, clamping circuit, loss computation — true oscillator training step is 135-250 pJ, not 120 pJ.
- **Honest claim:** "Energy per equilibrium computation scales as O(N) for oscillators vs O(N²) for digital. The curves cross at N≈25-150 depending on coupling architecture. Per-usable-gradient comparison requires TDC noise resolution."
- **Significance:** First measured, same-process-node comparison of physical vs digital equilibrium energy. Prior field claims (Rain AI 10,000×, Kendall 100×, Momeni 10,000×) are projections, not measurements.
- **Files:** `paper/TRAINING_ENERGY_COMPARISON.md`, `paper/TRAINING_ENERGY_AUDIT.md`, `kb/exploration/substrate/power_gated_results.json`, `kb/exploration/substrate/power_gated_oscillator.scs`, `rtl/training_step_digital.v`

### Q6. TDC noise limitation
- **What:** Single-shot phase measurement at GHz is impractical with current TDC
- **Evidence:** 1 ps TDC noise at 3 GHz → gradient destroyed (cos ≈ 0)
- **Implication:** Requires averaging O(100+) measurements, or larger β, or lower frequency
- **Value:** This is an honest hardware constraint that strengthens the paper's credibility
- **Interaction with Q5:** TDC noise is the reason energy-per-equilibrium ≠ energy-per-usable-gradient
- **Files:** `tdc_noise_results.json`

### Q7. Preliminary SPICE-Kuramoto agreement
- **What:** Kuramoto model predicts lock/beat boundaries in SPICE
- **Evidence:** 73% agreement (22/30) on lock/beat, phase correlation 0.05 (poor)
- **Caveats:** K calibration wrong by ~10×. Qualitative agreement, not quantitative.
- **Superseded by:** Q1 (new hardware bridge with proper calibration)
- **Files:** `kb/exploration/substrate/spice_vs_kuramoto.json`

---

## FAILED (killed by audits — do not repeat)

### F1. Energy efficiency claim
- **What we claimed:** 120 pJ oscillator bank is energy-efficient
- **What audit found:** Digital frequency counter does the same job at 2.9 pJ (40-58× better)
- **Why it failed:** 94% of oscillator power is bias current (oscillators running idle). Digital is active only during computation.
- **Lesson:** Oscillators burn power by existing. Energy claims require accounting for the full duty cycle. Never compare against MCU FFT (absurdly unfair) — compare against the simplest digital circuit that does the same job.
- **Files:** `paper/ENERGY_AUDIT.md`

### F2. 825× transistor density claim
- **What we claimed:** 48 transistors vs 39,630 = 825×
- **What audit found:** 448 (with DACs) vs 4,500 (fair digital) = 10×
- **Why it failed:** (a) Ignored DAC cost for input encoding, (b) compared against general-purpose digital instead of fixed-coefficient, (c) capacitor area dominates but wasn't counted, (d) precision mismatch (7-bit vs 12-bit)
- **Lesson:** Always compare like-for-like. Include the full analog signal chain. Count area, not just transistors.
- **Files:** `paper/MATVEC_AUDIT.md`, `THREE_CLAIMS_AUDIT.md`

### F3. "K can't learn / every other group is wrong" (universal claim)
- **What we claimed:** Natural frequency is THE parameter to learn, coupling is wrong
- **What audit found:** Wang et al. get 94.1% MNIST with K-only. Rageau gets 97.8%. K works fine on all-to-all architectures.
- **Why it failed:** Our finding is architecture-specific (sparse layered). We overclaimed universality.
- **Lesson:** Scope empirical claims to the conditions tested. "ω outperforms K on sparse architectures" is defensible. "Everyone else is wrong" is not.
- **Files:** `THREE_CLAIMS_AUDIT.md` Claim 3

### F4. Softmax equivalence
- **What we claimed:** All-to-all Kuramoto equilibrium ≈ softmax
- **What we found:** θ* = ω^c/(KN) — it's linear. The sin() nonlinearity only matters near the sync boundary where equilibrium is unstable.
- **Why it failed:** Uniform all-to-all coupling with strong K linearizes the dynamics. No operating regime is both stable AND nonlinear.
- **Lesson:** Strong coupling = linear. Weak coupling = unstable. The stable-nonlinear regime requires heterogeneous coupling (untested).
- **Files:** `paper/SOFTMAX_COMPARISON.md`

### F5. Oscillator as neural network activation function
- **What we claimed:** Oscillator can replace ReLU as a drop-in activation
- **What we found:** 88.5% ± 0.7% (oscillator) vs 88.4% ± 0.6% (linear/no activation). p = 0.87.
- **Why it failed:** (a) In the stable regime (K=5-7), the oscillator is linear, (b) gradient through W1 has cos=0.618 (centering bug), (c) ReLU at matched capacity gets 95% (gap is 6.1%, not 2%), (d) sin(θ*) readout adds only +0.5%
- **Lesson:** The stability-linearity trap is fundamental for uniform all-to-all coupling. The activation function needs heterogeneous coupling or a different oscillator model (Van der Pol, amplitude dynamics). Also: verify gradients end-to-end before claiming "exact."
- **Files:** `paper/ACTIVATION_AUDIT.md`, `OSCILLATOR_ACTIVATION.md`

### F6. 100% oscillator bank on FM (trivially easy task)
- **What we claimed:** Oscillator bank achieves 100% spectral decomposition
- **What audit found:** A raw frequency scalar also gets 100%. FFT gets 100%. Uniform sensor spacing with no learning gets 100%. At 5 classes, trivial baseline (93%) beats oscillator (83%).
- **Why it failed:** The 3-class task is trivially separable by any method. Classes too far apart.
- **Lesson:** Always run the trivial baseline FIRST. If a single feature solves the task, the oscillator adds nothing.
- **Files:** `phasegrad/experiments/audit/AUDIT_REPORT.md` Checks 1 and 6

### F7. Adaptive oscillator bank v1 (units bug)
- **What happened:** omega was in Hz, input_phase was in rad/s. No sensor ever injection-locked.
- **Impact:** All v1 results (76.3% mean, 54% ± 0.0% anomaly) were artifacts of the bug
- **Lesson:** Check units at the physics interface. If coherence features are uniformly blunt, suspect a scale mismatch.
- **Files:** `paper/ADAPTIVE_RESULTS.md`

### F8. FD gradient on transient coherence
- **What we tried:** Finite-difference gradient of LogReg log-loss w.r.t. sensor frequencies
- **What happened:** |∇| ≈ 0.02, zero accuracy change, across all experiments (v1, v2, sensor bank training)
- **Why it failed:** Coherence is a step function (locked ≈ 1, beating ≈ 0). The transition is sharp. FD straddles the boundary only if the sensor is exactly at the lock edge. Otherwise gradient is zero almost everywhere.
- **Lesson:** Don't differentiate through a threshold function with FD. Use EP on the equilibrium (where it's smooth) or a different training signal.
- **Files:** `train_sensor_bank_results.json`, `adaptive_bank_results.json`

### F9. Forced equilibrium for beating oscillators
- **What we tried:** Solving the forced Kuramoto equilibrium for sensors far from the input frequency
- **What happened:** 85% of configurations fail to converge. Beating oscillators have no fixed point in the rotating frame.
- **Lesson:** The forced equilibrium only exists for locked oscillators (small detuning). The EP gradient identity works beautifully where the equilibrium exists, but the equilibrium doesn't exist for the oscillators that are "computing" by NOT locking.
- **Files:** `forced_gradient_verification_extended.json` (65/443 converge = 14.7%)

### F10. Power gating reduces per-inference energy
- **What we claimed:** Gating oscillators off between inferences eliminates idle bias, reducing energy
- **What we found:** Gated 60.08 pJ vs ungated 57.2 pJ — 5% WORSE due to PMOS switch overhead
- **Why it failed:** The energy IS the oscillation. Switching power during active computation cannot be eliminated. Gating only saves leakage between inferences (0.0015 μW vs 178.8 μW).
- **What it's good for:** Duty-cycled operation (24× average power reduction at 4% duty cycle). Not per-computation energy.
- **Lesson:** Analog oscillators burn power by oscillating. There is no equivalent to clock gating. The only way to reduce per-computation energy is lower VDD or smaller transistors.
- **Files:** `kb/exploration/substrate/power_gated_results.json`, `power_gated_oscillator.scs`

### F11. Matched VDD (0.9V) oscillator operation
- **What we projected:** Running oscillators at 0.9V (same as digital) would cut energy ~4×
- **What audit found:** Lowest validated SPICE is 1.4V. GPDK045 NMOS Vt ≈ 0.4-0.5V → gate overdrive at 0.9V is marginal. Oscillators likely won't start.
- **Lesson:** VDD matching is a physics constraint, not a design knob. Ring oscillators need VDD >> 2×Vt for reliable oscillation.
- **Files:** `paper/TRAINING_ENERGY_AUDIT.md` Attack 2, `paper/hardware/freq_calibration.json` (starts at 1.4V)

### F12. Training energy advantage at small N
- **What we hoped:** Physical training (2× forward) beats digital training (forward + backward) on energy
- **What we measured:** At N=8, oscillator training 120 pJ vs digital 28.54 pJ — digital wins 4.2×
- **Why it failed:** (a) VDD mismatch (1.8V vs 0.9V = 4× penalty), (b) oscillator forward pass is 20× more expensive than digital, (c) incomplete oscillator accounting (no DAC, TDC, clamping — adds 15-130 pJ), (d) TDC noise requires 100+ averages per usable gradient
- **Lesson:** The "free backward pass" doesn't help if the forward pass is 20× more expensive. Training energy wins require either much lower forward pass energy (lower VDD, smaller transistors) or much larger N (where O(N) vs O(N²) dominates).
- **Files:** `paper/TRAINING_ENERGY_COMPARISON.md`, `paper/TRAINING_ENERGY_AUDIT.md`

---

## OPEN QUESTIONS (not tested, not failed — future work)

### O1. Heterogeneous coupling for nonlinearity
- Non-uniform K_ij might provide both stability AND nonlinearity
- The softmax and activation failures both trace to uniform coupling → linear
- Untested. Could break the stability-linearity trap.

### O2. ω-gradient training of sensor frequencies via EP
- The Hebbian rule works for adaptation. EP gradient on forced equilibrium works for locked sensors.
- Combining them: use EP for locked sensors, Hebbian for unlocked?
- The centering bug in the hybrid layer gradient (cos=0.618) needs fixing first.

### O3. Scaling beyond N=8 / 3 layers
- All SPICE results are N≤14, ≤3 layers
- The scaling argument (O(N) oscillator vs O(N²) digital) is theoretical
- Needs SPICE verification at N=16, 32 to confirm energy scaling

### O4. ~~Power-gated oscillators~~ TESTED — moved to F10
- Power gating doesn't help per-inference energy. See F10.
- Still useful for duty-cycled average power reduction.

### O5. Multi-dimensional classification (beyond frequency)
- The vowel classifier does 2D (F1 + F2), 120 transistors, 87%
- The density argument gets stronger with dimensionality (oscillator: linear, digital: quadratic)
- No systematic study of how the advantage scales with input dimension

### O6. Different oscillator models
- Van der Pol: amplitude dynamics provide a second variable (saturation nonlinearity for free)
- Stuart-Landau: complex amplitude, natural for both phase and amplitude encoding
- Could break the stability-linearity trap that kills uniform Kuramoto

---

## CALIBRATION DATA (reference, not paper claims)

### VDD-to-frequency (GPDK045 ring oscillator)
- Cubic fit, R² = 0.999994
- Range: 1.4V → 1.696 GHz, 2.2V → 3.979 GHz
- Source: `paper/hardware/freq_calibration.json`

### Effective coupling strength
- At coupling_strength=0.9: K_eff = 1.357 rad/ns
- Source: `paper/hardware/k_calibration.json`
- **Bug:** JSON has field `k_effective_ghz` that actually stores rad/ns value. Do not expose this label.

### Digital synthesis references (GPDK045)
- Frequency counter (8-bin): 91 cells, 72.3 μW, 2.9 pJ/classification
- 7×7 mat-vec (Q4.12): 5,284 cells, 2.92 mW, 2.9 pJ/multiply
- 2-layer mat-vec: 5,602 cells, 1.22 mW, 6.1 pJ
- 7×3 sparse (8-bit): 72 cells, ~540 transistors
- 4×4 dense (8-bit): 263 cells, ~1,973 transistors
- Sources: various `kb/exploration/digital-flow-runs/` directories

### SPICE oscillator power
- Per oscillator (W=4μm L=1μm): ~180-220 μW
- Per oscillator gated energy: 7.51 pJ (startup 0.63 + settling 3.26 + computation 3.62)
- 8-oscillator bank: 1.43-2.11 mW total, 57-60 pJ per inference
- Settling time: ~20-40 ns
- Leakage when OFF: 0.0015 μW
- Sources: `sensor_bank_measurements.json`, `matvec_results.json`, `power_gated_results.json`

### Digital training step (GPDK045)
- Full training circuit (2× matvec + subtractor + FSM): 9,590 cells, 2.854 mW, 28.54 pJ/step
- Achievable clock: ~333 MHz (timing violated at 400 MHz)
- Source: `rtl/training_step_digital.v`, synthesis in `kb/exploration/digital-flow-runs/`

---

## AUDIT TRAIL (all hostile reviews and their findings)

| Audit | File | Key findings |
|-------|------|-------------|
| Original adversarial review | `paper/ADVERSARIAL_FIXES.md` | 22 issues: sign error, notation mismatch, missing citations |
| Hostile audit (6 checks) | `phasegrad/experiments/audit/AUDIT_REPORT.md` | 2 PASS, 2 FAIL, 2 QUALIFIED |
| Three claims audit | `paper/THREE_CLAIMS_AUDIT.md` | Density WEAKENED 2.5-9×, stacking CONFIRMED, wrong-parameter DESTROYED |
| Matvec audit (7 attacks) | `paper/MATVEC_AUDIT.md` | 825× → 10-20× after DACs + fair comparison |
| Energy audit | `paper/ENERGY_AUDIT.md` | Digital counter wins 40-58× on energy |
| Activation audit (6 attacks) | `paper/ACTIVATION_AUDIT.md` | Oscillator = linear (p=0.87), gradient broken (cos=0.618) |
| Adaptive v2 audit (Check 7) | `phasegrad/experiments/audit/check7_v2_audit.py` | Uniform spacing gets 100%, Hebbian not needed |
| Softmax comparison | `paper/SOFTMAX_COMPARISON.md` | Linear in stable regime, not softmax |
| Training energy audit (7 attacks) | `paper/TRAINING_ENERGY_AUDIT.md` | 3 CONFIRMED, 4 QUALIFIED. Kill shot: TDC noise makes energy-per-gradient 100× worse |
