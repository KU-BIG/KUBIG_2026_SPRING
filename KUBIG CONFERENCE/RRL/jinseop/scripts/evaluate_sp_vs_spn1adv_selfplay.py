"""
Evaluation: SP vs SPN_1ADV under probabilistic adversary environment.
Each agent plays with clones of itself as teammates.

Test setup:
  - 3-player cooperative game (3_chefs_counter_circuit)
  - Main agent + 2 clones of itself as teammates:
      Teammate 1: always a clone (cooperative)
      Teammate 2: probabilistic -- acts as ADV with prob=ADV_PROB, else clone
  - SP  : SP_s1010_h256_tr[SP]_ran   (standalone SP, 80M steps)
  - SPN : PWADV-N-1-SP_..._attack2   (SPN_1ADV final agent)

Outputs:
  - Per-episode rewards split by normal / adversary condition
  - Summary comparison table
  - 10 GIF videos per agent → eval_results/gifs_selfplay_v2/
"""

import os
import sys
import random
import numpy as np
from pathlib import Path
from PIL import Image

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import pygame
pygame.init()
pygame.display.set_mode((1, 1))

sys.path.insert(0, '/workspace/multiHRI')

from oai_agents.common.arguments import get_arguments
from oai_agents.agents.agent_utils import load_agent
from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from scripts.evaluate_sp_vs_spn1adv import load_best_agent, make_env, render_frame, save_gif

# ─── Config ────────────────────────────────────────────────────────────────────
LAYOUT   = '3_chefs_counter_circuit'
BASE_DIR = Path('/workspace/multiHRI/agent_models/Classic/3')
GIFS_DIR = Path('/workspace/multiHRI/eval_results/gifs_selfplay_v2')
ADV_PROB = 0.5
N_EVAL   = 50
N_RENDER = 10
HORIZON  = 400
SEED     = 42

SP_PATH  = BASE_DIR / 'SP_s1010_h256_tr[SP]_ran' / 'best'
SPN_PATH = BASE_DIR / 'PWADV-N-1-SP_s1010_h256_tr[SP_SPADV]_ran_originaler_attack2' / 'best'
ADV_PATH = BASE_DIR / 'ADV_s68_h512_tr[H]_ran_selfisher_attack2' / 'best'
# ───────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)


class ProbAdversaryTeammate:
    """
    Teammate that switches between clone and adversary each episode.
    - With prob `adv_prob`: uses trained ADV agent (adversarial)
    - Otherwise: uses clone of the main agent (cooperative)
    """
    def __init__(self, clone_agent, adv_agent, adv_prob=0.5):
        self.clone    = clone_agent
        self.adv      = adv_agent
        self.adv_prob = adv_prob
        self.is_adversarial = False
        self.policy = clone_agent.policy
        self.args   = clone_agent.args
        self.name   = f"ProbAdv(p={adv_prob:.1f})"

    @property
    def encoding_fn(self):
        return self.adv.encoding_fn if self.is_adversarial else self.clone.encoding_fn

    def new_episode(self):
        self.is_adversarial = random.random() < self.adv_prob

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        agent = self.adv if self.is_adversarial else self.clone
        return agent.predict(obs, state=state, deterministic=deterministic)

    def set_encoding_params(self, p_idx, horizon, env=None, mdp=None, **kwargs):
        self.clone.set_encoding_params(p_idx, horizon, env=env, mdp=mdp, **kwargs)
        self.adv.set_encoding_params(p_idx, horizon, env=env, mdp=mdp, **kwargs)

    def get_start_position(self, *args, **kwargs):
        return None


def evaluate(label, main_agent, clone_tm, prob_adv, args):
    """Run N_EVAL + N_RENDER episodes, collect stats and save GIFs."""
    env = make_env(args)
    env.set_teammates([clone_tm, prob_adv])

    main_agent.set_encoding_params(0, HORIZON, env=env, is_haha=False, tune_subtasks=False)
    env.encoding_fn = main_agent.encoding_fn
    clone_tm.set_encoding_params(1, HORIZON, env=env, is_haha=False, tune_subtasks=False)
    prob_adv.set_encoding_params(2, HORIZON, env=env, is_haha=False, tune_subtasks=False)

    all_r, adv_r, nrm_r = [], [], []
    gif_n = 0

    print(f"\n{'='*57}")
    print(f"  Agent : {label}")
    print(f"  Team  : {label}_clone × 2  (1 may become ADV p={ADV_PROB})")
    print(f"{'='*57}")

    for ep in range(N_EVAL + N_RENDER):
        render = ep < N_RENDER
        prob_adv.new_episode()
        env.set_teammates([clone_tm, prob_adv])
        obs = env.reset(p_idx=0)

        done = False
        total = 0.0
        frames = [render_frame(env)] if render else []

        while not done:
            action = main_agent.predict(obs)[0]
            obs, r, done, _ = env.step(action)
            total += r
            if render:
                frames.append(render_frame(env))

        all_r.append(total)
        (adv_r if prob_adv.is_adversarial else nrm_r).append(total)

        tag  = "ADV" if prob_adv.is_adversarial else "NRM"
        mark = " 🎬" if render else ""
        print(f"  ep {ep+1:3d} [{tag}]  reward={total:6.1f}{mark}")

        if render and frames:
            path = GIFS_DIR / label / f"ep{ep+1:02d}_{tag}_rew{total:.0f}.gif"
            save_gif(frames, path)
            gif_n += 1

    nrm_mean = np.mean(nrm_r) if nrm_r else 0.0
    adv_mean = np.mean(adv_r) if adv_r else 0.0
    print(f"\n  ── {label} ──")
    print(f"  Overall        : {np.mean(all_r):.2f} ± {np.std(all_r):.2f}  (n={len(all_r)})")
    print(f"  Normal team    : {nrm_mean:.2f}  (n={len(nrm_r)})")
    print(f"  Adversary team : {adv_mean:.2f}  (n={len(adv_r)})")
    print(f"  Robustness     : {adv_mean/nrm_mean*100:.1f}%  |  zero-reward: {adv_r.count(0)} / {len(adv_r)}")
    print(f"  GIFs saved     : {gif_n}  →  {GIFS_DIR / label}/")

    return {
        'all': np.array(all_r),
        'nrm': np.array(nrm_r) if nrm_r else np.array([0.]),
        'adv': np.array(adv_r) if adv_r else np.array([0.]),
    }


def main():
    sys.argv = [
        'eval',
        '--layout-names', LAYOUT,
        '--num-players',  '3',
        '--teammates-len', '2',
        '--n-envs', '1',
        '--horizon', str(HORIZON),
    ]
    args = get_arguments()
    args.base_dir = Path('/workspace/multiHRI')

    GIFS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading agents...")
    sp_agent  = load_best_agent(SP_PATH,  args)
    spn_agent = load_best_agent(SPN_PATH, args)
    adv_agent = load_best_agent(ADV_PATH, args)

    def fresh(path):
        return load_best_agent(path, args)

    print("All agents loaded.")

    sp_res = evaluate(
        label      = 'SP',
        main_agent = sp_agent,
        clone_tm   = fresh(SP_PATH),
        prob_adv   = ProbAdversaryTeammate(fresh(SP_PATH), adv_agent, ADV_PROB),
        args       = args,
    )

    spn_res = evaluate(
        label      = 'SPN_1ADV',
        main_agent = spn_agent,
        clone_tm   = fresh(SPN_PATH),
        prob_adv   = ProbAdversaryTeammate(fresh(SPN_PATH), adv_agent, ADV_PROB),
        args       = args,
    )

    print("\n" + "="*62)
    print("  FINAL COMPARISON  (each agent plays with clones of itself)")
    print("="*62)
    fmt = "{:<40} {:>9} {:>11}"
    print(fmt.format("Metric", "SP", "SPN_1ADV"))
    print("-"*62)
    print(fmt.format(f"Overall  (n={N_EVAL+N_RENDER})",
        f"{np.mean(sp_res['all']):.2f}", f"{np.mean(spn_res['all']):.2f}"))
    print(fmt.format("  ↳ Normal teammate",
        f"{np.mean(sp_res['nrm']):.2f}", f"{np.mean(spn_res['nrm']):.2f}"))
    print(fmt.format("  ↳ Adversarial teammate",
        f"{np.mean(sp_res['adv']):.2f}", f"{np.mean(spn_res['adv']):.2f}"))
    print(fmt.format("Robustness (ADV/NRM)",
        f"{np.mean(sp_res['adv'])/np.mean(sp_res['nrm'])*100:.1f}%",
        f"{np.mean(spn_res['adv'])/np.mean(spn_res['nrm'])*100:.1f}%"))
    print(fmt.format("Zero-reward under ADV",
        f"{list(sp_res['adv']).count(0)}/{len(sp_res['adv'])}",
        f"{list(spn_res['adv']).count(0)}/{len(spn_res['adv'])}"))

    d_adv = np.mean(spn_res['adv']) - np.mean(sp_res['adv'])
    print(f"\n  SPN_1ADV advantage under adversary: {d_adv:+.2f}")
    print(f"  GIFs → {GIFS_DIR}")
    print("="*62)


if __name__ == '__main__':
    main()
