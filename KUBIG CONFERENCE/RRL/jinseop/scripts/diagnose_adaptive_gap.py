"""
Diagnoses why the Adaptive condition's adversarial-teammate reward (65.93) is
much lower than PWADV_advonly's own training eval_mean_reward (~180), even
though only 1/4 of the episode (100/400 steps) is spent on the naive SP policy.

Runs, for N forced-adversarial episodes each:
  (A) SPN_1ADV playing the full 400 steps with its own policy from t=0
      (native, well-adapted from the start)
  (B) PWADV_advonly playing the full 400 steps with its own policy from t=0
      ("fresh-start" baseline -- matches its own training eval condition)
  (C) Adaptive: SP for steps [0,100), then PWADV_advonly for [100,400)
      (the actual deployed condition)

For each, reward is split into phase1 (steps 0-99) and phase2 (steps 100-400)
so we can see whether PWADV_advonly's phase2 reward collapses specifically
when it takes over a state that was shaped by 100 steps of naive SP play
against the adversary (distribution-shift / accumulated-damage hypothesis),
versus a fresh start at t=100 with a clean board.
"""
import os
import sys
import random
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np

sys.path.insert(0, '/workspace')

from oai_agents.common.arguments import get_arguments
from oai_agents.agents.agent_utils import load_agent

LAYOUT = '3_chefs_counter_circuit'
BASE_DIR = Path('/workspace/agent_models/Classic/3')
HORIZON = 400
SWITCH_STEP = 100
N_EPISODES = 30
SEED = 123

SP_PATH = BASE_DIR / 'SP_s1010_h256_tr[SP]_ran' / 'best'
SPN_PATH = BASE_DIR / 'PWADV-N-1-SP_s1010_h256_tr[SP_SPADV]_ran_originaler_attack2' / 'best'
ADV_PATH = BASE_DIR / 'ADV_s68_h512_tr[H]_ran_selfisher_attack2' / 'best'
ADVONLY_PATH = BASE_DIR / 'PWADV-N-1-SP_s1010_h256_tr[SPADV]_ran_originaler_attack2_advonly' / 'best'


def load_best_agent(agent_dir, args):
    candidate = agent_dir / 'agents_dir' / 'agent_0'
    path = candidate if candidate.exists() else agent_dir
    return load_agent(path, args)


def make_env(args):
    from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv
    return OvercookedGymEnv(args=args, layout_name=LAYOUT, ret_completed_subtasks=False,
                             is_eval_env=True, horizon=HORIZON, learner_type='originaler')


class SwitchingWrapper:
    """Plays as `first` until `step_count==switch_at`, then plays as `second`. Used to make
    BOTH main and teammate1 switch simultaneously, exactly matching the real Adaptive setup
    (2 copies of the switched-to policy + 1 real adversary during phase2)."""
    def __init__(self, first, second, switch_at):
        self.first = first
        self.second = second
        self.switch_at = switch_at
        self.policy = first.policy
        self.args = first.args
        self.step_count = 0
        self.active = first

    @property
    def encoding_fn(self):
        return self.active.encoding_fn

    def get_start_position(self, *a, **kw):
        return None

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        action, st = self.active.predict(obs, state=state, episode_start=episode_start, deterministic=deterministic)
        self.step_count += 1
        if self.step_count == self.switch_at:
            self.active = self.second
        return action, st


def run_episode(env, main_agent, clone_tm, adv_agent, switch_step):
    """
    main_agent plays position 0, clone_tm plays position 1, adv_agent plays position 2
    (always the true adversary -- forced adversarial every episode, for this diagnostic).
    main_agent/clone_tm may be plain agents or SwitchingWrapper instances.
    """
    if hasattr(main_agent, 'step_count'):
        main_agent.step_count = 0
        main_agent.active = main_agent.first
    if hasattr(clone_tm, 'step_count'):
        clone_tm.step_count = 0
        clone_tm.active = clone_tm.first

    env.set_teammates([clone_tm, adv_agent])
    obs = env.reset(p_idx=0)
    done = False
    t = 0
    r1, r2 = 0.0, 0.0
    while not done:
        action = main_agent.predict(obs)[0]
        obs, r, done, _ = env.step(action)
        if t < switch_step:
            r1 += r
        else:
            r2 += r
        t += 1
    return r1, r2, r1 + r2, t


def summarize(name, r1s, r2s, totals):
    print(f"\n-- {name} (n={len(totals)}) --")
    print(f"  phase1 [0,{SWITCH_STEP}) mean reward : {np.mean(r1s):.2f}")
    print(f"  phase2 [{SWITCH_STEP},400) mean reward: {np.mean(r2s):.2f}")
    print(f"  total mean reward             : {np.mean(totals):.2f}")


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    sys.argv = ['diag', '--layout-names', LAYOUT, '--num-players', '3', '--teammates-len', '2',
                '--n-envs', '1', '--horizon', str(HORIZON)]
    args = get_arguments()
    args.base_dir = Path('/workspace')

    print("Loading agents...")
    spn_agent = load_best_agent(SPN_PATH, args)
    spn_clone = load_best_agent(SPN_PATH, args)
    advonly_agent = load_best_agent(ADVONLY_PATH, args)
    advonly_clone = load_best_agent(ADVONLY_PATH, args)
    sp_agent = load_best_agent(SP_PATH, args)
    sp_clone = load_best_agent(SP_PATH, args)
    adv_agent = load_best_agent(ADV_PATH, args)
    print("Agents loaded.\n")

    env = make_env(args)

    # (A) SPN_1ADV native, full 400 steps, forced adversarial
    a_r1, a_r2, a_tot = [], [], []
    for ep in range(N_EPISODES):
        r1, r2, tot, _ = run_episode(env, spn_agent, spn_clone, adv_agent, SWITCH_STEP)
        a_r1.append(r1); a_r2.append(r2); a_tot.append(tot)
    summarize('SPN_1ADV (native, full 400 steps)', a_r1, a_r2, a_tot)

    # (B) PWADV_advonly fresh-start, full 400 steps, forced adversarial (matches its own training-eval condition)
    b_r1, b_r2, b_tot = [], [], []
    for ep in range(N_EPISODES):
        r1, r2, tot, _ = run_episode(env, advonly_agent, advonly_clone, adv_agent, SWITCH_STEP)
        b_r1.append(r1); b_r2.append(r2); b_tot.append(tot)
    summarize('PWADV_advonly (fresh-start, full 400 steps)', b_r1, b_r2, b_tot)

    # (C) Adaptive actual: SP for [0,100), then advonly for [100,400) -- BOTH main and teammate1
    # switch together (matches the real Adaptive eval: 2x advonly + 1x ADV during phase2)
    main_sw = SwitchingWrapper(sp_agent, advonly_agent, SWITCH_STEP)
    clone_sw = SwitchingWrapper(sp_clone, advonly_clone, SWITCH_STEP)
    c_r1, c_r2, c_tot = [], [], []
    for ep in range(N_EPISODES):
        r1, r2, tot, _ = run_episode(env, main_sw, clone_sw, adv_agent, SWITCH_STEP)
        c_r1.append(r1); c_r2.append(r2); c_tot.append(tot)
    summarize('Adaptive (SP->advonly at t=100, both team members switch together)', c_r1, c_r2, c_tot)


if __name__ == '__main__':
    main()
