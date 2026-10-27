"""Evaluate the trained DQN agents against Round Robin, Priority and SJF.

    python evaluate.py              # every models/dqn_seed*.npz, 10 unseen test seeds per workload
    python evaluate.py --seeds 3    # quicker

Prints a summary table and writes results/results.csv, results/summary.md and figures.
"""
import argparse
import csv
import glob
import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

from baselines import BASELINES, run_baseline  # noqa: E402
from env import FREQS, IS_BIG, N_CORES, N_FREQ, N_TYPES, SCENARIOS, SchedulingEnv  # noqa: E402
from train import make_agent, run_greedy  # noqa: E402

POLICIES = ['RR', 'Priority', 'SJF', 'DQN']
COLORS = dict(RR='#eb6834', Priority='#1baf7a', SJF='#eda100', DQN='#2a78d6')  # validated categorical slots
INK, MUTED, GRID = '#0b0b0b', '#898781', '#e1e0d9'
METRICS = ['turnaround', 'waiting', 'weighted_turnaround', 'energy', 'fairness', 'p95_slowdown', 'throttle', 'cost']
W_SWEEP = [0, 1, 2, 4, 8, 12]
G = 5 * N_CORES  # start of the global features in the observation

plt.rcParams.update({'font.size': 7, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.axisbelow': True,
                     'legend.frameon': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})


def load_agents():
    paths = sorted(glob.glob('models/dqn_seed*.npz'))
    if not paths:
        raise SystemExit('no trained model in models/ - run train.py first')
    agents = []
    for p in paths:
        ag = make_agent()
        ag.load(p)
        agents.append(ag)
    print(f'loaded {len(agents)} agent(s): {", ".join(paths)}')
    return agents


def summarise(rows):
    out = {}
    for scen in SCENARIOS:
        for pol in POLICIES:
            sel = [r for r in rows if r['scenario'] == scen and r['policy'] == pol]
            out[scen, pol] = {m: (np.mean([r[m] for r in sel]), np.std([r[m] for r in sel])) for m in METRICS}
    return out


def print_and_save_table(summary, path):
    cols = [('turnaround', 'Turnaround', 1), ('weighted_turnaround', 'Prio-wtd TA', 1), ('energy', 'Energy', 0),
            ('fairness', 'Jain', 3), ('p95_slowdown', 'p95 slowdown', 1), ('throttle', 'Throttle %', 1),
            ('cost', 'Cost', 0)]
    lines = ['| Workload | Policy | ' + ' | '.join(c[1] for c in cols) + ' |', '|' + '---|' * (len(cols) + 2)]
    for scen in SCENARIOS:
        for pol in POLICIES:
            s = summary[scen, pol]
            cells = [f'{s[m][0] * (100 if m == "throttle" else 1):.{d}f}' for m, _, d in cols]
            lines.append(f'| {scen} | {pol} | ' + ' | '.join(cells) + ' |')
    text = '\n'.join(lines)
    print(text)
    with open(path, 'w') as fh:
        fh.write(text + '\n')


def fig_comparison(summary):
    panels = [('turnaround', 'Avg turnaround (ticks)'), ('energy', 'Energy (units)'),
              ('fairness', "Jain's fairness on slowdown"), ('throttle', 'Thermally throttled time (%)')]
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.0))
    x, w = np.arange(len(SCENARIOS)), 0.2
    for ax, (m, title) in zip(axes, panels):
        k = 100 if m == 'throttle' else 1
        for i, pol in enumerate(POLICIES):
            mean = [summary[s, pol][m][0] * k for s in SCENARIOS]
            std = [summary[s, pol][m][1] * k for s in SCENARIOS]
            ax.bar(x + (i - 1.5) * w, mean, w, yerr=std, color=COLORS[pol], edgecolor='white', linewidth=0.6,
                   error_kw=dict(elinewidth=0.5, ecolor=INK, capsize=0), label=pol)
        ax.set_title(title, fontsize=7, color=INK)
        ax.set_xticks(x, list(SCENARIOS), rotation=35, ha='right')
        ax.grid(axis='x', visible=False)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.2)
    fig.savefig('results/fig_comparison.png')
    plt.close(fig)


def fig_pareto(pareto):
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    markers = dict(RR='o', Priority='s', SJF='^')
    for pol, mk in markers.items():
        pts = sorted((p['energy'], p['turnaround'], p['freq']) for p in pareto if p['policy'] == pol)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=MUTED, lw=1, marker=mk, ms=4,
                markeredgecolor='white', markeredgewidth=0.6, label=f'{pol} (fixed f)')
        ax.annotate(pol, pts[0][:2], xytext=(5, 0), textcoords='offset points', va='center', color=INK,
                    fontsize=6.5)  # label the low-frequency end, where the three lines are far apart
    dqn = [p for p in pareto if p['policy'] == 'DQN']
    ax.scatter([p['energy'] for p in dqn], [p['turnaround'] for p in dqn], s=22, color=COLORS['DQN'],
               edgecolor='white', linewidth=0.6, zorder=3, label=f"DQN, $w_E$ = {dqn[0]['w']:g}-{dqn[-1]['w']:g}")
    ax.set_xlabel('Energy (units)')
    ax.set_ylabel('Avg turnaround (ticks)')
    ax.set_yscale('log')
    ax.set_yticks([20, 30, 50, 100, 200])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(fontsize=6.5, loc='upper right')
    fig.savefig('results/fig_pareto.png')
    plt.close(fig)


def val_curve(path):
    """(episode, validation cost) pairs from a train.py stdout log."""
    with open(path) as fh:
        return [(int(line.split()[1]), float(line.split()[9])) for line in fh if line.startswith('ep ')]


def fig_training():
    logs = [val_curve(p) for p in sorted(glob.glob('results/train_stdout_dqn_seed*.txt'))]
    n = min(len(lg) for lg in logs)
    ep = np.array([e for e, _ in logs[0][:n]])
    vals = np.array([[v for _, v in lg[:n]] for lg in logs])
    env = SchedulingEnv()
    from train import VAL
    best = min(np.mean([run_baseline(env, rule, s, **SCENARIOS[sc])['cost'] for sc, s in VAL])
               for rule in BASELINES.values())
    fig, ax = plt.subplots(figsize=(3.3, 1.8))
    ax.fill_between(ep, vals.min(0), vals.max(0), color=COLORS['DQN'], alpha=0.18, linewidth=0)
    ax.plot(ep, vals.mean(0), color=COLORS['DQN'], lw=2, marker='o', ms=3,
            label='DQN (final config.)' if len(logs) == 1 else f'DQN, mean of {len(logs)} seeds')
    ax.axhline(best, color=MUTED, lw=1)
    ax.annotate('best baseline (SJF @ $f_{max}$)', (ep[-1], best), xytext=(0, 3), textcoords='offset points',
                ha='right', color=INK, fontsize=6.5)
    ax.set_xlabel('Training episode')
    ax.set_ylabel('Validation cost (lower is better)')
    ax.set_ylim(top=min(vals.max(), 3 * best))
    ax.legend(fontsize=6.5, loc='upper right')
    fig.savefig('results/fig_training.png')
    plt.close(fig)


def fig_policy(decisions, temps):
    obs = np.array([o for o, _ in decisions])
    act = np.array([a for _, a in decisions])
    slot, typ, f = act // (N_TYPES * N_FREQ), (act // N_FREQ) % N_TYPES, act % N_FREQ
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.75))

    # (a) DVFS level vs. number of waiting jobs: ordinal blue ramp, light = slow
    n_wait = np.rint(obs[:, G] * 20)
    edges, labels = [1, 2, 4, 8, 16, 1e9], ['1', '2-3', '4-7', '8-15', '16+']
    ramp = ['#9ec5f4', '#3987e5', '#184f95']
    bucket = np.digitize(n_wait, edges) - 1
    share = np.array([[np.mean(f[bucket == b] == k) if (bucket == b).any() else 0 for b in range(5)]
                      for k in range(N_FREQ)])
    bottom = np.zeros(5)
    for k in range(N_FREQ):
        axes[0].bar(labels, share[k], 0.7, bottom=bottom, color=ramp[k], edgecolor='white', linewidth=0.8,
                    label=f'{FREQS[k]:g} $f_{{max}}$')
        bottom += share[k]
    axes[0].set_xlabel('Jobs waiting')
    axes[0].set_ylabel('Share of dispatches')
    axes[0].set_title('(a) Chosen frequency vs. queue length', fontsize=7)
    axes[0].legend(fontsize=6, loc='upper left', bbox_to_anchor=(1, 1))
    axes[0].grid(axis='x', visible=False)

    # (b) LITTLE-core share vs. estimated job size, only when both core types were free
    both = ((obs[:, 0] == 0) | (obs[:, 5] == 0)) & ((obs[:, 10] == 0) | (obs[:, 15] == 0))
    size = np.expm1(obs[np.arange(len(obs)), G + 5 + 5 * slot + 2] * 5)
    s_edges, s_labels = [0, 2, 5, 15, 40, 1e9], ['<2', '2-5', '5-15', '15-40', '40+']
    sb = np.digitize(size, s_edges) - 1
    little = [np.mean(typ[both & (sb == b)] == 1) if (both & (sb == b)).any() else 0 for b in range(5)]
    axes[1].bar(s_labels, little, 0.6, color=COLORS['DQN'], edgecolor='white')
    axes[1].set_xlabel('Estimated remaining work of the chosen job')
    axes[1].set_ylabel('Sent to a LITTLE core')
    axes[1].set_title('(b) Core choice when both types are free', fontsize=7)
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis='x', visible=False)

    # (c) hottest big-core temperature over time, heavy load
    for pol, (t, temp) in temps.items():
        axes[2].plot(t, temp, color=COLORS[pol], lw=1.2, label=pol)
    axes[2].axhline(85, color=MUTED, lw=0.8)
    axes[2].set_xlabel('Time (ticks), heavy workload')
    axes[2].set_ylabel('Hottest big core (\u00b0C)')
    axes[2].set_title('(c) Temperature (throttle at 85 \u00b0C)', fontsize=7)
    axes[2].legend(fontsize=6, loc='lower right')
    fig.tight_layout(w_pad=1.0)
    fig.savefig('results/fig_policy.png')
    plt.close(fig)


ABLATION = [  # (model tag, description); seed 0, up to 200 episodes, commands in README
    ('abl_flat_g99', 'shared Q-net, 1-step, no shaping'),
    ('abl_flat_g95', '  same with gamma = 0.95'),
    ('abl_duel_g99', '  + dueling'),
    ('abl_flat_g99_al', '  + advantage learning'),
    ('abl_flat_g99_n3', '  + 3-step returns'),
    ('abl_shape_flat_n3', '3-step + shaping, remaining-work potential'),
    ('wspt_n1', '1-step + shaping, WSPT potential'),
    ('wspt_n3_duel', '3-step + WSPT shaping + dueling'),
    ('dqn', '3-step + WSPT shaping (final)'),
]


def ablation(env):
    """Score every ablation checkpoint on 16 extra held-out validation workloads (not test)."""
    extra = [(n, 10_100 + 10 * i + k) for i, n in enumerate(['light', 'medium', 'heavy', 'bursty']) for k in range(4)]
    sjf = np.mean([run_baseline(env, BASELINES['SJF'], s, **SCENARIOS[n])['cost'] for n, s in extra])
    lines = ['| Variant | Episodes run | Validation cost |', '|---|---|---|',
             f'| SJF @ f_max (best baseline) | - | {sjf:.0f} |']
    for tag, desc in ABLATION:
        path = f'models/{tag}_seed0.npz'
        if not os.path.exists(path):
            continue
        with np.load(path) as f:
            ag = make_agent(dueling=len(f.files) == 12)
        ag.load(path)
        cost = np.mean([run_greedy(ag, env, s, **SCENARIOS[n])['cost'] for n, s in extra])
        log = glob.glob(f'results/train_stdout_{tag}_seed0.txt') + glob.glob(f'results/ablation_logs/train_stdout_{tag}.txt')
        episodes = val_curve(log[0])[-1][0] if log else '-'
        lines.append(f'| {desc.strip()} | {episodes} | {cost:.0f} |')
    text = '\n'.join(lines)
    print(text)
    with open('results/ablation.md', 'w') as fh:
        fh.write(text + '\n')


def temperature_trace(env, agent, seed, horizon=600):
    """Hottest big-core temperature at every decision point: SJF @ f_max vs. the DQN agent."""
    traces = {}
    for pol in ('SJF', 'DQN'):
        env.reset(seed, **SCENARIOS['heavy'])
        t, temp = [], []
        obs, mask = env.observe()
        while not env.done and env.t < horizon:
            t.append(env.t)
            temp.append(env.temp[IS_BIG].max())
            if pol == 'SJF':
                j, sl = BASELINES['SJF'](env)
                env.dispatch(j, env.core_job.index(-1), N_FREQ - 1, sl)
                env.advance()
            else:
                obs, mask, _, _ = env.step(agent.act(obs, mask))
        traces[pol] = (t, temp)
    return traces


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=10, help='test workloads per scenario')
    args = p.parse_args()
    os.makedirs('results', exist_ok=True)
    agents = load_agents()
    env = SchedulingEnv()
    seeds = range(20_000, 20_000 + args.seeds)  # disjoint from training and validation seeds

    rows, decisions = [], []
    for scen, wl in SCENARIOS.items():
        print(f'evaluating {scen} ...', flush=True)
        for s in seeds:
            for name, rule in BASELINES.items():
                rows.append(dict(scenario=scen, policy=name, seed=s, model='', **run_baseline(env, rule, s, **wl)))
            for k, ag in enumerate(agents):
                rec = decisions if k == 0 and scen != 'overload' else None
                rows.append(dict(scenario=scen, policy='DQN', seed=s, model=k, **run_greedy(ag, env, s, rec, **wl)))
    with open('results/results.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    print('energy/latency trade-off sweep (medium workload) ...', flush=True)
    pareto = []
    for name, rule in BASELINES.items():
        for f in range(N_FREQ):
            ms = [run_baseline(env, rule, s, freq=f, **SCENARIOS['medium']) for s in seeds]
            pareto.append(dict(policy=name, freq=FREQS[f], w='', **{m: np.mean([x[m] for x in ms]) for m in METRICS}))
    for w in W_SWEEP:
        ms = [run_greedy(ag, env, s, w_energy=w, **SCENARIOS['medium']) for ag in agents for s in seeds]
        pareto.append(dict(policy='DQN', freq='', w=w, **{m: np.mean([x[m] for x in ms]) for m in METRICS}))
    with open('results/pareto.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(pareto[0]))
        wr.writeheader()
        wr.writerows(pareto)

    summary = summarise(rows)
    print_and_save_table(summary, 'results/summary.md')
    fig_comparison(summary)
    fig_pareto(pareto)
    fig_policy(decisions, temperature_trace(env, agents[0], seeds[0]))
    if glob.glob('results/train_stdout_dqn_seed*.txt'):
        fig_training()
    ablation(env)
    print('wrote results/results.csv, results/pareto.csv, results/summary.md and results/fig_*.png')


if __name__ == '__main__':
    main()
