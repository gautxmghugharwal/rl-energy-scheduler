"""Train the preference-conditioned DQN scheduler on randomised workloads.

    python train.py --seed 0          # final configuration -> models/dqn_seed0.npz
Flags expose the ablation variants (see README).
"""
import argparse
import csv
import os
import time
from collections import deque

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')  # small matrices: threading only adds overhead
import numpy as np  # noqa: E402

from agent import DQNAgent  # noqa: E402
from env import (CONTEXT_DIM, K_SLOTS, N_FREQ, N_TYPES, SCENARIOS, SLOT_DIM, W_ENERGY, W_MAX,  # noqa: E402
                 SchedulingEnv)

VAL = [(name, 10_000 + 10 * i + k) for i, name in enumerate(['light', 'medium', 'heavy', 'bursty'])
       for k in range(2)]  # 8 held-out validation workloads, never used for test


def make_agent(seed=0, **kw):
    return DQNAgent(CONTEXT_DIM, K_SLOTS, SLOT_DIM, N_TYPES * N_FREQ, seed=seed, **kw)


def random_workload(rng):
    """Domain randomisation: load, arrival process, burst law and the energy preference."""
    return dict(load=rng.uniform(0.3, 0.95),
                arrivals='bursty' if rng.random() < 0.3 else 'poisson',
                bursts='pareto' if rng.random() < 0.3 else 'mixed',
                w_energy=rng.uniform(0, W_MAX))


def run_greedy(agent, env, seed, record=None, **workload):
    env.reset(seed, **workload)
    obs, mask = env.observe()
    while not env.done:
        a = agent.act(obs, mask)
        if record is not None:
            record.append((obs, a))
        obs, mask, _, _ = env.step(a)
    return env.metrics()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--nstep', type=int, default=3, help='n-step return length')
    p.add_argument('--al-alpha', type=float, default=0.0, help='advantage-learning gap coefficient')
    p.add_argument('--shaping', type=float, default=1.0, help='weight of the potential-based shaping term (0 = off)')
    p.add_argument('--potential', default='wspt', choices=['wspt', 'work'], help='shaping potential')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--eps-min', type=float, default=0.05, help='exploration floor')
    p.add_argument('--dueling', action='store_true', help='ablation: dueling (V + A) Q-network')
    p.add_argument('--tag', default='dqn', help='name prefix for the model and log files')
    args = p.parse_args()
    args.out = f'models/{args.tag}_seed{args.seed}.npz'
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    rng = np.random.default_rng(args.seed)
    env = SchedulingEnv()
    agent = make_agent(args.seed, dueling=args.dueling, al_alpha=args.al_alpha, lr=args.lr)
    log_path = f'results/train_log_{args.tag}_seed{args.seed}.csv'
    progress = open(f'results/train_stdout_{args.tag}_seed{args.seed}.txt', 'w')  # read by evaluate.py

    def report(line):
        print(line, flush=True)
        progress.write(line + '\n')
        progress.flush()

    with open(log_path, 'w', newline='') as fh:
        log = csv.writer(fh)
        log.writerow(['episode', 'steps', 'epsilon', 'return', 'loss', 'val_cost', 'minutes'])
        steps, best, t0 = 0, np.inf, time.time()
        for ep in range(args.episodes):
            eps = max(args.eps_min, 1 - ep / (0.4 * args.episodes))
            env.reset(seed=args.seed * 1_000_000 + ep, **random_workload(rng))
            obs, mask = env.observe()
            ret, losses, window = 0.0, [], deque()  # window: last n (obs, mask, action, reward)
            phi = env.potential(args.potential)
            while not env.done:
                a = agent.act(obs, mask, eps)
                obs2, mask2, r, done = env.step(a)
                ret += r
                # potential-based shaping (Ng et al. 1999): leaves the optimal policy unchanged but pays
                # for progress immediately instead of only when jobs finally complete
                phi2 = env.potential(args.potential)
                r += args.shaping * (args.gamma * phi2 - phi)
                phi = phi2
                window.append((obs, mask, a, r))
                while window and (len(window) == args.nstep or done):  # n-step return; flush at episode end
                    ret_n = sum(args.gamma ** k * x[3] for k, x in enumerate(window))
                    agent.remember(*window[0][:3], ret_n, obs2, mask2, 0.0 if done else args.gamma ** len(window))
                    window.popleft()
                if agent.size >= 5_000 and steps % 2 == 0:
                    losses.append(agent.learn())
                obs, mask = obs2, mask2
                steps += 1
            val = ''
            if (ep + 1) % 10 == 0:
                val = np.mean([run_greedy(agent, env, s, w_energy=W_ENERGY, **SCENARIOS[n])['cost'] for n, s in VAL])
                if val < best:
                    best = val
                    agent.save(args.out)
                report(f'ep {ep + 1:4d}  steps {steps:7d}  eps {eps:.2f}  return {ret:8.1f}  '
                       f'val_cost {val:7.1f}  best {best:7.1f}  {(time.time() - t0) / 60:.1f} min')
            log.writerow([ep + 1, steps, round(eps, 3), round(ret, 2),
                          round(np.mean(losses), 4) if losses else '', val, round((time.time() - t0) / 60, 2)])
            fh.flush()
    report(f'saved best model (val cost {best:.1f}) to {args.out}')
    progress.close()


if __name__ == '__main__':
    main()
