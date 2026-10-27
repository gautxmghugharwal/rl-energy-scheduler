"""Sanity checks for the simulator and the hand-written DQN. Run: python tests.py"""
import numpy as np

from agent import DQNAgent
from baselines import BASELINES, run_baseline
from env import FREQS, K_DYN, N_ACTIONS, OBS_DIM, P_IDLE, P_STATIC, SCENARIOS, T_MAX, SchedulingEnv


def run_one_job(work):
    env = SchedulingEnv(n_jobs=1)
    env.reset(seed=0)
    env.burst[:] = env.rem[:] = env.est[:] = work
    env.dispatch(env.ready[0], 0, len(FREQS) - 1, None)  # big core 0 at f_max, run to completion
    env.advance()
    return env


def test_single_job_timing_and_energy():
    env = run_one_job(5.0)
    t0 = env.arr[0]
    assert env.done and env.finish[0] == t0 + 5 and env.throttle_ticks == 0
    nominal = t0 * P_IDLE.sum() + 5 * (P_STATIC[0] + K_DYN[0] + P_IDLE[1:].sum())
    assert nominal < env.energy < 1.03 * nominal  # small extra leakage as the core warms up


def test_sustained_f_max_throttles():
    env = run_one_job(300.0)
    assert env.throttle_ticks > 0 and env.peak_temp < T_MAX + 1
    assert env.finish[0] - env.arr[0] > 300  # throttling cost real time


def test_every_policy_finishes_every_job():
    env = SchedulingEnv(n_jobs=60)
    for name, wl in SCENARIOS.items():
        for rule in BASELINES.values():
            m = run_baseline(env, rule, seed=1, **wl)
            assert (env.finish >= env.arr + 1).all() and (env.rem <= 1e-9).all(), name
            assert 0 < m['fairness'] <= 1 and m['waiting'] >= 0
        env.reset(seed=1, **wl)  # random legal actions also terminate
        obs, mask = env.observe()
        rng = np.random.default_rng(0)
        while not env.done:
            assert obs.shape == (OBS_DIM,) and mask.shape == (N_ACTIONS,) and mask.any()
            obs, mask, r, done = env.step(rng.choice(np.flatnonzero(mask)))
            assert r <= 0
        assert env.n_done == env.n and env.potential() == 0


def test_backprop_matches_finite_differences():
    rng = np.random.default_rng(0)
    for dueling in (True, False):
        ag = DQNAgent(n_context=3, n_slots=3, slot_dim=4, per_slot=2, hidden=8, dueling=dueling, seed=3)
        nets = [ag.q.adv] + ([ag.q.val] if dueling else [])
        for net in nets:  # float64 so finite differences are exact enough
            net.params = [p.astype(np.float64) for p in net.params]
        ag.q.params = [p for net in nets for p in net.params]
        s, a, y = rng.standard_normal((5, 15)), rng.integers(0, 6, 5), rng.standard_normal(5)
        s[:, 3::4] = rng.random((5, 3)) < 0.7  # slot 'present' flags
        _, grads = ag.loss_and_grads(s, a, y)
        for p, g in zip(ag.q.params, grads):
            for i in rng.choice(p.size, min(3, p.size), replace=False):
                old = p.flat[i]
                p.flat[i] = old + 1e-6
                up = ag.loss_and_grads(s, a, y)[0]
                p.flat[i] = old - 1e-6
                down = ag.loss_and_grads(s, a, y)[0]
                p.flat[i] = old
                assert abs((up - down) / 2e-6 - g.flat[i]) < 1e-6 + 1e-4 * abs(g.flat[i]), dueling


if __name__ == '__main__':
    test_single_job_timing_and_energy()
    test_sustained_f_max_throttles()
    test_every_policy_finishes_every_job()
    test_backprop_matches_finite_differences()
    print('all checks passed')
