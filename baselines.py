"""Classical schedulers. Each rule picks (job, time slice) from env.ready; the job goes to
the first free core (big cores first) at a fixed frequency (f_max by default)."""
from env import N_FREQ, QUANTUM


def round_robin(env):
    return env.ready[0], QUANTUM  # FIFO queue, preempted jobs rejoin at the back


def priority(env):  # non-preemptive, highest priority first, ties by arrival
    return max(env.ready, key=lambda j: (env.prio[j], -env.arr[j])), None


def sjf(env):  # non-preemptive, shortest *estimated* burst first
    return min(env.ready, key=lambda j: (env.est[j], env.arr[j])), None


BASELINES = {'RR': round_robin, 'Priority': priority, 'SJF': sjf}


def run_baseline(env, rule, seed, freq=N_FREQ - 1, **workload):
    env.reset(seed, **workload)
    while not env.done:
        j, slice_len = rule(env)
        env.dispatch(j, env.core_job.index(-1), freq, slice_len)
        env.advance()
    return env.metrics()
