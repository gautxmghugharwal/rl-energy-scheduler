"""CPU scheduling simulator: 4 heterogeneous cores (2 big + 2 LITTLE) with DVFS and a
thermal model (temperature-dependent leakage + hardware throttling).

Time moves in integer ticks. Whenever a core is free and jobs are waiting, the
scheduler picks (job, core, frequency). The job then runs for one time slice
(QUANTUM ticks) or, for the non-preemptive baselines, until it finishes.

The RL agent (observe/step) and the classical baselines (dispatch/advance) drive
the same simulator, so every policy sees identical workloads and identical metrics.
"""
import math

import numpy as np

# --- platform ---------------------------------------------------------------
SPEED = np.array([1.0, 1.0, 0.55, 0.55])       # work/tick at f_max (big, big, LITTLE, LITTLE)
K_DYN = np.array([1.0, 1.0, 0.30, 0.30])       # dynamic power = K_DYN * f^3  (P ~ C V^2 f, V ~ f)
P_STATIC = np.array([0.15, 0.15, 0.05, 0.05])  # leakage while busy, at ambient temperature
P_IDLE = np.array([0.05, 0.05, 0.02, 0.02])    # clock-gated idle core
FREQS = np.array([0.5, 0.75, 1.0])             # DVFS levels, fraction of f_max
IS_BIG = SPEED == SPEED.max()
N_CORES, N_FREQ = len(SPEED), len(FREQS)
CAPACITY = SPEED.sum()                         # work/tick with every core at f_max

# --- thermal (first-order RC model per core) ----------------------------------
T_AMB, T_MAX = 45.0, 85.0  # deg C: chassis ambient, throttle threshold
R_TH = 45.0                # steady-state rise, deg C per power unit
TAU_TH = 60.0              # thermal time constant, ticks
LEAK_TEMP = 0.02           # leakage grows 2% per deg C above ambient
# A big core held at f_max heads for ~105 C and throttles after ~65 ticks; 0.75 f_max
# settles near 75 C. LITTLE cores never throttle.

# --- MDP --------------------------------------------------------------------
QUANTUM = 4                                    # time slice for RR and the agent
K_SLOTS = 8                                    # waiting jobs visible to the agent
N_TYPES = 2                                    # agent picks a core type: big / LITTLE
N_ACTIONS = K_SLOTS * N_TYPES * N_FREQ         # 48
CONTEXT_DIM, SLOT_DIM = 5 * N_CORES + 5, 5     # shared features, features per visible job
OBS_DIM = CONTEXT_DIM + SLOT_DIM * K_SLOTS     # 65
W_ENERGY = 4.0                                 # default price of one energy unit (in waiting job-ticks)
W_MAX = 12.0                                   # training range of the energy-preference knob [0, W_MAX]
ETA = 0.25                                     # weight of the slowdown (fairness) pressure
REWARD_SCALE = 10.0

SCENARIOS = {
    'light':     dict(load=0.45, arrivals='poisson', bursts='mixed'),
    'medium':    dict(load=0.70, arrivals='poisson', bursts='mixed'),
    'heavy':     dict(load=0.90, arrivals='poisson', bursts='mixed'),
    'bursty':    dict(load=0.70, arrivals='bursty', bursts='mixed'),
    'heavytail': dict(load=0.70, arrivals='poisson', bursts='pareto'),
    'overload':  dict(load=1.05, arrivals='poisson', bursts='mixed'),  # outside the training range
}


def make_workload(rng, n_jobs, load, arrivals, bursts):
    """Sample n_jobs as (arrival tick, true burst, noisy burst estimate, priority 1..5)."""
    if bursts == 'mixed':  # 70% short interactive, 30% long batch
        short = rng.random(n_jobs) < 0.7
        burst = np.where(short, 1 + rng.exponential(3, n_jobs), 10 + rng.exponential(25, n_jobs))
    else:                  # heavy-tailed: Pareto(alpha=1.5), capped
        burst = np.minimum(1.5 * (1 + rng.pareto(1.5, n_jobs)), 150)
    rate = load * CAPACITY / burst.mean()  # arrival rate that gives the requested utilisation
    # bursty = on/off phases of 50 ticks at 1.8x / 0.2x the mean rate, sampled by thinning
    peak = rate * (1.8 if arrivals == 'bursty' else 1.0)
    arr, t = [], 0.0
    while len(arr) < n_jobs:
        t += rng.exponential(1 / peak)
        if arrivals == 'poisson' or (t // 50) % 2 == 0 or rng.random() < 0.2 / 1.8:
            arr.append(t)
    est = burst * rng.lognormal(0, 0.3, n_jobs)  # schedulers only see a noisy estimate
    prio = rng.integers(1, 6, n_jobs)            # 5 = most important
    return np.floor(arr).astype(int), burst, est, prio


class SchedulingEnv:
    def __init__(self, n_jobs=200):
        self.n = n_jobs

    def reset(self, seed, load=0.7, arrivals='poisson', bursts='mixed', w_energy=W_ENERGY):
        rng = np.random.default_rng(seed)
        self.arr, self.burst, self.est, self.prio = make_workload(rng, self.n, load, arrivals, bursts)
        self.w_energy = w_energy
        self.rho = self.prio / 3.0  # holding-cost weight, mean 1
        self.rem = self.burst.copy()
        self.wait = np.zeros(self.n)
        self.start = np.full(self.n, -1)
        self.finish = np.full(self.n, -1)
        self.t = self.nxt = self.n_done = self.busy_ticks = self.throttle_ticks = 0
        self.ready = []
        self.core_job = [-1] * N_CORES
        self.core_freq = [0] * N_CORES
        self.slice_left = [0] * N_CORES
        self.temp = np.full(N_CORES, T_AMB)
        self.peak_temp = T_AMB
        self.energy = self.total_cost = self.util = self.load_ema = self.arrived = 0.0
        self.done = False
        self.advance()

    def est_rem(self, j):
        return max(self.est[j] - (self.burst[j] - self.rem[j]), 0.5)

    def potential(self, kind='wspt'):
        """Shaping potential Phi(s) (training only; 0 once every job is done).
        'wspt': minus a lower bound on the remaining priority-weighted flow time - all jobs in
                the system run in Smith's WSPT order (smallest est_rem/rho first) on one machine
                with the CPU's full capacity. Progress on a job then pays for every job queued
                behind it, which is the real value of finishing work early.
        'work': minus the priority-weighted estimated remaining work (ignores queueing)."""
        jobs = [j for j in self.ready + self.core_job if j >= 0]
        if kind == 'work':
            return -sum(self.rho[j] * self.est_rem(j) for j in jobs) / REWARD_SCALE
        t = total = 0.0
        for j in sorted(jobs, key=lambda j: self.est_rem(j) / self.rho[j]):
            t += self.est_rem(j) / CAPACITY
            total += self.rho[j] * t
        return -total / REWARD_SCALE

    def dispatch(self, j, core, f, slice_len=QUANTUM):
        """Start job j on a free core at frequency level f; slice_len=None runs it to completion."""
        self.ready.remove(j)
        self.core_job[core], self.core_freq[core] = j, f
        self.slice_left[core] = slice_len or math.inf
        if self.start[j] < 0:
            self.start[j] = self.t

    def advance(self):
        """Run the clock until a core is free while jobs wait, or every job is done.
        Returns the cost accumulated on the way (the negative reward)."""
        cost = 0.0
        while True:
            while self.nxt < self.n and self.arr[self.nxt] <= self.t:
                self.ready.append(self.nxt)
                self.arrived += self.est[self.nxt]
                self.nxt += 1
            if self.ready and -1 in self.core_job:
                return cost
            if self.n_done == self.n:
                self.done = True
                return cost
            cost += self._tick()

    def _tick(self):
        for j in self.ready:
            self.wait[j] += 1
        running = [j for j in self.core_job if j >= 0]
        # holding cost: every job in the system costs rho_j per tick, more the longer it has
        # waited relative to its size (HRRN-style pressure against starvation)
        hold = sum(self.rho[j] * (1 + ETA * self.wait[j] / (self.est[j] + QUANTUM))
                   for j in self.ready + running)
        power = 0.0
        for c in range(N_CORES):
            j = self.core_job[c]
            if j < 0:
                p = P_IDLE[c]
            else:
                throttled = self.temp[c] >= T_MAX  # hardware forces the lowest DVFS level
                f = FREQS[0 if throttled else self.core_freq[c]]
                p = P_STATIC[c] * (1 + LEAK_TEMP * (self.temp[c] - T_AMB)) + K_DYN[c] * f ** 3
                self.busy_ticks += 1
                self.throttle_ticks += throttled
                self.rem[j] -= SPEED[c] * f
                self.slice_left[c] -= 1
                if self.rem[j] <= 1e-9:
                    self.finish[j] = self.t + 1
                    self.n_done += 1
                    self.core_job[c] = -1
                elif self.slice_left[c] <= 0:  # slice expired: back of the queue
                    self.ready.append(j)
                    self.core_job[c] = -1
            self.temp[c] += (T_AMB + R_TH * p - self.temp[c]) / TAU_TH
            power += p
        self.peak_temp = max(self.peak_temp, self.temp.max())
        self.util = 0.95 * self.util + 0.05 * len(running) / N_CORES
        self.load_ema = 0.98 * self.load_ema + 0.02 * self.arrived / CAPACITY
        self.arrived = 0.0
        self.energy += power
        self.t += 1
        cost = (hold + self.w_energy * power) / REWARD_SCALE
        self.total_cost += cost
        return cost

    # --- RL interface -------------------------------------------------------
    def _candidates(self):
        """Up to K_SLOTS waiting jobs shown to the agent: the oldest, shortest and most
        important ones, so a long queue never hides a starving or a tiny job."""
        q = sorted(self.ready, key=lambda j: self.arr[j])
        if len(q) > K_SLOTS:
            pick = (q[:3] + sorted(q, key=self.est_rem)[:3]
                    + sorted(q, key=lambda j: -self.prio[j])[:2] + q)
            q = sorted(list(dict.fromkeys(pick))[:K_SLOTS], key=lambda j: self.arr[j])
        return q

    def observe(self):
        """Return (state vector, boolean mask of legal actions)."""
        o = np.zeros(OBS_DIM, np.float32)
        for c in range(N_CORES):
            j = self.core_job[c]
            heat = (self.temp[c] - T_AMB) / (T_MAX - T_AMB)
            if j >= 0:
                o[5 * c:5 * c + 5] = 1, IS_BIG[c], FREQS[self.core_freq[c]], min(self.est_rem(j), 40) / 40, heat
            else:
                o[5 * c + 1], o[5 * c + 4] = IS_BIG[c], heat
        g = 5 * N_CORES
        backlog = sum(self.est_rem(j) for j in self.ready)
        o[g:g + 5] = (min(len(self.ready) / 20, 3), self.util, min(backlog / CAPACITY / 50, 3),
                      min(self.load_ema, 2), self.w_energy / W_MAX)
        free = [any(self.core_job[c] < 0 and IS_BIG[c] == big for c in range(N_CORES)) for big in (True, False)]
        mask = np.zeros(N_ACTIONS, bool)
        self.cands = self._candidates()
        for i, j in enumerate(self.cands):
            b = g + 5 + 5 * i
            o[b:b + 5] = (1, self.prio[j] / 5, math.log1p(self.est_rem(j)) / 5,
                          min(self.wait[j] / (self.est[j] + QUANTUM) / 10, 1), min(self.wait[j] / 100, 2))
            for typ in range(N_TYPES):
                if free[typ]:
                    a = (i * N_TYPES + typ) * N_FREQ
                    mask[a:a + N_FREQ] = True
        return o, mask

    def step(self, a):
        """Action a = (slot, core type, frequency level). Returns obs, mask, reward, done."""
        slot, rest = divmod(int(a), N_TYPES * N_FREQ)
        typ, f = divmod(rest, N_FREQ)
        core = next(c for c in range(N_CORES) if self.core_job[c] < 0 and IS_BIG[c] == (typ == 0))
        self.dispatch(self.cands[slot], core, f)
        r = -self.advance()
        return *self.observe(), r, self.done

    def metrics(self):
        ta = self.finish - self.arr
        sd = ta / np.ceil(self.burst)  # slowdown vs. running alone on a big core at f_max
        return dict(turnaround=ta.mean(), waiting=self.wait.mean(), response=(self.start - self.arr).mean(),
                    weighted_turnaround=(self.prio * ta).sum() / self.prio.sum(),
                    energy=self.energy, fairness=sd.sum() ** 2 / (len(sd) * (sd ** 2).sum()),
                    p95_slowdown=np.percentile(sd, 95), throttle=self.throttle_ticks / max(self.busy_ticks, 1),
                    peak_temp=self.peak_temp, cost=self.total_cost)
