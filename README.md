# Thermal- and Energy-Aware CPU Scheduling with Preference-Conditioned Deep Q-Learning

A reinforcement-learning scheduler that decides, every time a CPU core frees up,
**which process runs, on which kind of core, and at what clock frequency**, trading
off execution time, energy, fairness and heat. It is compared against Round Robin,
Priority Scheduling and Shortest Job First on six different workloads.

What goes beyond a plain RL scheduler:

| Feature | Why it matters |
|---|---|
| **Heterogeneous big.LITTLE platform + DVFS** | 2 fast/power-hungry and 2 slow/efficient cores, 3 frequency levels each. The agent jointly chooses the job, the core type and the frequency. |
| **Thermal model with hardware throttling** | Each core heats up (first-order RC model), leakage power grows with temperature, and a big core above 85 °C is forced down to its lowest frequency. Running everything at f_max (the "performance governor" used by the baselines) hits this thermal wall, and the agent learns to avoid it. |
| **Energy-preference input** | The energy weight `w_E` is part of the state and is randomized during training, so a single network can be queried at any energy/latency preference. In our runs the learned policy responds only weakly to it (see Limitations). |
| **Permutation-equivariant Q-network** | One small network scores every waiting job the same way, using the system context, the job's own features and how it compares with the other visible jobs. Picking the best job becomes an argmax over jobs instead of something a flat network must discover. In our first version a plain MLP over all slots could not beat SJF under load. |
| **Potential-based reward shaping with a scheduling-theory potential** | Without it the agent was myopic: the energy saved by a slow core shows up in the reward at once, while the latency it causes only arrives when jobs finally finish. We add `γΦ(s') − Φ(s)` to each reward, where `Φ` is minus a lower bound on the remaining priority-weighted flow time (Smith's WSPT rule on one machine with the CPU's full capacity). Progress then pays off immediately, in proportion to the queue it unblocks. Shaping of this form provably leaves the optimal policy unchanged (Ng et al., 1999), and it is used only during training. |
| **DQN written from scratch in NumPy** | Forward pass, backpropagation and Adam are hand-written (verified by a finite-difference gradient check). No deep-learning framework is needed. |
| **Noisy burst estimates, varied workloads, unseen overload test** | Schedulers only see noisy burst-time estimates. Training randomizes load, arrival process and burst distribution. Testing includes an overload level never seen in training. |

## Repository layout

```
env.py          simulator: cores, DVFS, power, thermal model, workloads, MDP (state/action/reward), metrics
baselines.py    Round Robin, Priority (non-preemptive), SJF (non-preemptive)
agent.py        Double DQN (shared per-job NumPy MLP, masked actions, replay buffer, soft target network)
train.py        training loop with domain randomisation + validation-based checkpointing
evaluate.py     test on 6 workloads x unseen seeds, Pareto sweep, policy analysis, figures
tests.py        sanity checks (timing, energy accounting, throttling, gradient check)
models/         final agent (dqn_seed0.npz) + checkpoints of the ablation variants
results/        results.csv, summary.md, pareto.csv, ablation.md, figures, training log
results/ablation_logs/   validation curves of every ablation run
report/         2-page report (LaTeX source; compile on Overleaf)
```

## Installation

Python 3.9+ is required (tested with 3.13). The only dependencies are NumPy and Matplotlib.

```bash
cd rl-energy-scheduler
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Running it (all from the terminal)

```bash
python tests.py                      # sanity checks, ~5 s

python evaluate.py                   # evaluate the included trained agent, ~5-10 min
python evaluate.py --seeds 3         # faster, fewer test workloads

python train.py --seed 0             # retrain the final agent (200 episodes, ~15-30 min on a laptop CPU)
```

`evaluate.py` prints the comparison table to the terminal and writes
`results/summary.md`, `results/results.csv`, `results/pareto.csv`, `results/ablation.md`
and the figures `results/fig_*.png`.

### Reproduction

```bash
python train.py --seed 0             # -> models/dqn_seed0.npz (single-threaded and fully seeded, so deterministic)
python evaluate.py
```

Every random number comes from a seeded generator. Training workloads use seeds
`seed*1e6 + episode`, checkpoint selection uses 8 validation workloads (`10000-10031`),
the ablation table uses 16 further validation workloads (`10100-10133`), and testing uses
`20000-20009` for every scenario, so test workloads are never seen during training or model selection.

### Ablation runs (seed 0)

```bash
python train.py --tag abl_flat_g99      --nstep 1 --shaping 0
python train.py --tag abl_flat_g95      --nstep 1 --shaping 0 --gamma 0.95
python train.py --tag abl_duel_g99      --nstep 1 --shaping 0 --dueling
python train.py --tag abl_flat_g99_al   --nstep 1 --shaping 0 --al-alpha 0.9
python train.py --tag abl_flat_g99_n3   --nstep 3 --shaping 0
python train.py --tag abl_shape_flat_n3 --nstep 3 --potential work
python train.py --tag wspt_n1           --nstep 1
python train.py --tag wspt_n3_duel      --nstep 3 --dueling
```

The early ablation runs used a 4-workload validation set during training. To compare them fairly,
`evaluate.py` re-scores every saved checkpoint on the same 16 extra validation workloads (`results/ablation.md`).
Some runs were stopped before 200 episodes once their trend was clear; the table lists the episodes actually run.

## RL formulation (summary; details in the report)

- **State (65-d vector):** for each core: busy, big/LITTLE, current frequency, remaining work of its job, temperature; global: queue length, CPU-utilisation EMA, queued work (backlog), arrival-load EMA, energy preference `w_E`; for up to 8 visible waiting jobs: present, priority, estimated remaining burst (log), waiting/size ratio, waiting time. The 8 visible jobs are the 3 oldest, 3 shortest and 2 most important ones, so a long queue never hides a starving or a tiny job.
- **Action (48 discrete):** (which of the 8 visible jobs) × (big or LITTLE core) × (frequency 0.5 / 0.75 / 1.0 f_max). Illegal actions (empty slot, no free core of that type) are masked. The job runs for one 4-tick time slice, then the agent decides again, so preemption and migration are possible.
- **Reward:** `r_t = -(1/10) * sum over ticks [ sum_{j in system} rho_j * (1 + 0.25 * wait_j / (est_j + 4)) + w_E * P(tick) ]`, where `rho_j` is the priority weight and `P` the chip power (dynamic `k f^3`, temperature-dependent leakage, idle power). The first term is a priority-weighted holding cost (its integral is weighted turnaround time). The waiting/size ratio grows the longer a job starves relative to its size, which pushes for fairness. The second term prices energy.
- **Algorithm:** Double DQN with 3-step returns and potential-based reward shaping (WSPT potential). The Q-network is one MLP (33→64→64→6, ReLU) applied to each of the 8 visible jobs. Its input is the 25 shared context features, the job's 5 features, and 3 relative features (size above the shortest visible job, priority below the highest, waiting ratio below the most-starved). Its 6 outputs are the Q-values of (big/LITTLE) × (3 frequencies) for that job. Training uses Huber loss, Adam (lr 3e-4), γ = 0.99, a 200k replay buffer, batch 64, soft target update τ = 0.005, ε-greedy 1.0→0.05 over the first 40 % of episodes, and one gradient step every 2 decisions. Each training episode samples a random workload and a random `w_E ~ U[0, 12]`. Every 10 episodes the greedy policy is scored on 8 held-out validation workloads at `w_E = 4`, and the best checkpoint is kept.

## Results

Test set: 10 unseen workloads per scenario, energy weight `w_E = 4`. Cost is the negative episode reward (lower is better).

| Scenario | RR | Priority | SJF | **DQN** | DQN energy vs SJF | DQN turnaround vs SJF | Jain fairness SJF → DQN |
|---|---|---|---|---|---|---|---|
| light | 1602 | 1604 | 1585 | **1368** | −38.5 % | +64.6 % | 0.70 → 0.88 |
| medium | 1696 | 1856 | 1654 | **1527** | −27.1 % | +29.9 % | 0.56 → 0.85 |
| heavy | 2367 | 3101 | 1996 | **1956** | −20.0 % | +12.4 % | 0.57 → 0.72 |
| bursty | 1726 | 1874 | 1670 | **1572** | −26.2 % | +29.1 % | 0.63 → 0.86 |
| heavytail | 592 | 607 | 580 | **533** | −29.0 % | +38.0 % | 0.71 → 0.84 |
| overload (unseen load) | 3608 | 5452 | **2312** | 2415 | −17.8 % | +10.1 % | 0.61 → 0.62 |

- **Lowest cost in 5 of 6 scenarios** (−2 % to −14 % vs. the best baseline). It loses by 4.5 % on overload, a load level never seen in training.
- **18–39 % less energy than SJF**, and **almost no thermal throttling** (≤ 0.2 % of busy time vs. 8.5–21 % for the baselines).
- **Fairer:** best or second-best Jain index in every scenario, and the lowest 95th-percentile slowdown in 4 of 6.
- **The price is a higher mean turnaround** (+10 % to +65 % vs. SJF at f_max): with `w_E = 4` the agent trades latency for energy.
- **At equal energy it beats every fixed-frequency baseline** (`results/fig_pareto.png`). Against SJF at 0.75 f_max it gets 9 % lower turnaround, half the p95 slowdown and Jain 0.86 vs. 0.62.
- **Learned behaviour** (`results/fig_policy.png`):
  - The frequency follows the queue length.
  - The efficient LITTLE cores are preferred when both core types are free.
  - Big cores are kept below the 85 °C throttle point.
- **Ablation** (`results/ablation.md`):
  - Dueling, advantage learning and γ = 0.95 did not help.
  - Potential-based reward shaping with the WSPT potential plus 3-step returns is what beats SJF.

**Limitations (honest):**
- DQN training was unstable. The best validation checkpoint came at episode 10, and later checkpoints got worse, so the run was stopped at episode 80. The model is chosen by validation cost from a single seed. A full 200-episode `python train.py --seed 0` reproduces the same run (seeded, single-threaded) and replaces the checkpoint only if a later one scores better on validation.
- The energy-preference input `w_E` has only a weak effect on the learned policy (turnaround 29–33, energy 1986–2113 for `w_E` = 0…12).
- All results come from simulation.

Figures: `results/fig_comparison.png`, `results/fig_pareto.png`, `results/fig_policy.png`, `results/fig_training.png`.
Full numbers: `results/summary.md`, `results/results.csv`, `results/pareto.csv`.
