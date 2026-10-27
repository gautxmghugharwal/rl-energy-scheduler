"""Double DQN with a dueling, permutation-equivariant Q-network, action masking, uniform
replay and a soft-updated target network.

Q(s, a) = V(context) + A(context, job_i, job_i vs. other jobs)[core type, freq] - mean_a A
  * A is ONE small MLP shared by all job slots: a job is scored the same way wherever it
    sits in the queue window (permutation equivariance), so "pick the best job" is an
    argmax over slots instead of something a flat MLP must discover.
  * V is a separate MLP on the shared context. Under load V is ~-100 while good and bad
    dispatches differ by <1; the dueling split lets A learn those small gaps directly.

Everything (forward pass, backprop, Adam) is written directly in NumPy: for networks
this small that is several times faster on a CPU than a deep-learning framework.
"""
import copy

import numpy as np


class MLP:
    def __init__(self, sizes, rng):
        self.params = []  # [W1, b1, W2, b2, ...]
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            scale = np.sqrt(2 / n_in) * (0.1 if i == len(sizes) - 2 else 1.0)  # He init, small output layer
            self.params += [(rng.standard_normal((n_in, n_out)) * scale).astype(np.float32),
                            np.zeros(n_out, np.float32)]

    def forward(self, x):
        """Return the output and the layer inputs that backward() needs."""
        acts = [x]
        for i in range(0, len(self.params), 2):
            x = x @ self.params[i] + self.params[i + 1]
            if i < len(self.params) - 2:
                x = np.maximum(x, 0)
            acts.append(x)
        return x, acts

    def backward(self, acts, dout):
        grads = [None] * len(self.params)
        for i in range(len(self.params) - 2, -1, -2):
            grads[i] = acts[i // 2].T @ dout
            grads[i + 1] = dout.sum(0)
            if i:
                dout = (dout @ self.params[i].T) * (acts[i // 2] > 0)  # through the ReLU
        return grads


class QNet:
    def __init__(self, n_context, n_slots, slot_dim, per_slot, hidden, dueling, rng):
        """Observation = n_context shared features + n_slots blocks of slot_dim job features
        (slot feature 0 = present, 1 = priority, 2 = size, 3 = waiting/size ratio);
        per_slot actions per job; actions are numbered slot-major."""
        self.split = n_context, n_slots, slot_dim
        self.adv = MLP([n_context + slot_dim + 3, hidden, hidden, per_slot], rng)
        self.val = MLP([n_context, hidden, hidden, 1], rng) if dueling else None
        self.params = self.adv.params + (self.val.params if dueling else [])

    def slot_inputs(self, obs):
        """(B, obs_dim) -> (B * n_slots, input): context + job features + job vs. the best
        visible job (size above the shortest, priority below the highest, waiting ratio below
        the most-starved); all three are 0 for the job that is best on that criterion."""
        c, k, d = self.split
        ctx, slots = obs[:, :c], obs[:, c:].reshape(len(obs), k, d)
        present = slots[:, :, :1] > 0
        size, prio, starve = slots[:, :, 2:3], slots[:, :, 1:2], slots[:, :, 3:4]
        rel = [size - np.where(present, size, np.inf).min(1, keepdims=True),
               prio - np.where(present, prio, -np.inf).max(1, keepdims=True),
               starve - np.where(present, starve, -np.inf).max(1, keepdims=True)]
        rel = [np.where(present, r, 0) for r in rel]
        x = np.concatenate([np.repeat(ctx[:, None], k, 1), slots, *rel], axis=2)
        return x.reshape(len(obs) * k, -1).astype(np.float32)

    def forward(self, obs):
        a, a_cache = self.adv.forward(self.slot_inputs(obs))
        q = a.reshape(len(obs), -1)
        if self.val is None:
            return q, (a_cache, None)
        v, v_cache = self.val.forward(obs[:, :self.split[0]].astype(np.float32))
        return v + q - q.mean(1, keepdims=True), (a_cache, v_cache)

    def backward(self, cache, dq):
        a_cache, v_cache = cache
        if self.val is None:
            return self.adv.backward(a_cache, dq.reshape(-1, self.adv.params[-1].size))
        da = dq - dq.mean(1, keepdims=True)  # d/dA_b of (A_a - mean A)
        return (self.adv.backward(a_cache, da.reshape(-1, self.adv.params[-1].size))
                + self.val.backward(v_cache, dq.sum(1, keepdims=True)))


class DQNAgent:
    def __init__(self, n_context, n_slots, slot_dim, per_slot, hidden=64, dueling=False, al_alpha=0.0,
                 lr=3e-4, batch=64, buffer=200_000, tau=0.005, seed=0):
        self.rng = np.random.default_rng(seed)
        self.q = QNet(n_context, n_slots, slot_dim, per_slot, hidden, dueling, self.rng)
        self.target = copy.deepcopy(self.q)
        self.lr, self.batch, self.tau, self.al_alpha = lr, batch, tau, al_alpha
        self.adam_m = [np.zeros_like(p) for p in self.q.params]
        self.adam_v = [np.zeros_like(p) for p in self.q.params]
        self.updates = 0
        obs_dim, n_actions = n_context + n_slots * slot_dim, n_slots * per_slot
        self.cap, self.size, self.ptr = buffer, 0, 0
        self.s = np.zeros((buffer, obs_dim), np.float32)
        self.s2 = np.zeros((buffer, obs_dim), np.float32)
        self.m1 = np.zeros((buffer, n_actions), bool)  # legal actions in s
        self.m2 = np.zeros((buffer, n_actions), bool)  # legal actions in s2
        self.a = np.zeros(buffer, np.int64)
        self.r = np.zeros(buffer, np.float32)
        self.g = np.zeros(buffer, np.float32)  # bootstrap discount: gamma^n, or 0 at episode end

    def act(self, obs, mask, eps=0.0):
        if self.rng.random() < eps:
            return int(self.rng.choice(np.flatnonzero(mask)))
        return int(np.where(mask, self.q.forward(obs[None])[0][0], -np.inf).argmax())

    def remember(self, s, m, a, r, s2, m2, discount):
        """Store a (possibly n-step) transition: r is the discounted reward sum, discount = gamma^n."""
        i = self.ptr
        self.s[i], self.m1[i], self.a[i], self.r[i] = s, m, a, r
        self.s2[i], self.m2[i], self.g[i] = s2, m2, discount
        self.ptr = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def loss_and_grads(self, s, a, y):
        """Huber loss between Q(s, a) and targets y, and its gradient w.r.t. the online net."""
        q, cache = self.q.forward(s)
        rows = np.arange(len(a))
        td = q[rows, a] - y
        dq = np.zeros_like(q)
        dq[rows, a] = np.clip(td, -1, 1) / len(a)
        loss = np.where(np.abs(td) < 1, 0.5 * td ** 2, np.abs(td) - 0.5).mean()
        return loss, self.q.backward(cache, dq)

    def learn(self):
        idx = self.rng.integers(0, self.size, self.batch)
        s, m, a, r, s2, m2, g = (x[idx] for x in (self.s, self.m1, self.a, self.r, self.s2, self.m2, self.g))
        rows = np.arange(len(a))
        a2 = np.where(m2, self.q.forward(s2)[0], -1e9).argmax(1)  # Double DQN: online net picks a'
        y = r + g * self.target.forward(s2)[0][rows, a2]  # target net scores it
        if self.al_alpha:  # advantage learning (Bellemare et al. 2016): widen the action gap,
            qt = self.target.forward(s)[0]  # same greedy policy, far less sensitive to Q noise
            y -= self.al_alpha * (np.where(m, qt, -np.inf).max(1) - qt[rows, a])
        loss, grads = self.loss_and_grads(s, a, y)
        norm = np.sqrt(sum((g * g).sum() for g in grads))
        if norm > 10:
            grads = [g * (10 / norm) for g in grads]
        self.updates += 1
        b1, b2 = 0.9, 0.999
        step = self.lr * (1 - b2 ** self.updates) ** 0.5 / (1 - b1 ** self.updates)
        for p, g, m, v, tp in zip(self.q.params, grads, self.adam_m, self.adam_v, self.target.params):
            m += (1 - b1) * (g - m)
            v += (1 - b2) * (g * g - v)
            p -= step * m / (np.sqrt(v) + 1e-8)
            tp += self.tau * (p - tp)  # soft target update
        return float(loss)

    def save(self, path):
        np.savez(path, *self.q.params)

    def load(self, path):
        with np.load(path) as f:
            assert len(f.files) == len(self.q.params), 'checkpoint does not match this network'
            for i, p in enumerate(self.q.params):
                p[...] = f[f'arr_{i}']
        self.target = copy.deepcopy(self.q)
