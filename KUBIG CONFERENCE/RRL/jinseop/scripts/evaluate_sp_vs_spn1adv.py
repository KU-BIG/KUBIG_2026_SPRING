"""
Evaluation: SP vs SPN_1ADV under probabilistic adversary environment.

Test setup:
  - 3-player cooperative game (3_chefs_counter_circuit)
  - 2 teammates per episode:
      Teammate 1: normal SP agent (always cooperative)
      Teammate 2: probabilistic -- acts as ADV with prob=ADV_PROB, else SP
  - Compare SP-trained vs SPN_1ADV-trained main agents

Outputs:
  - Per-episode rewards (normal vs adversary conditions)
  - Summary comparison table
  - 10 GIF videos per agent condition (eval_results/gifs/)
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

# ─── Config ────────────────────────────────────────────────────────────────────
LAYOUT       = '3_chefs_counter_circuit'
BASE_DIR     = Path('/workspace/multiHRI/agent_models/Classic/3')
OUTPUT_DIR   = Path('/workspace/multiHRI/eval_results')
GIFS_DIR     = OUTPUT_DIR / 'gifs'
ADV_PROB     = 0.5      # probability that teammate 2 is adversarial each episode
N_EVAL       = 50       # evaluation episodes (no rendering)
N_RENDER     = 10       # episodes to render as GIF
TILE_SIZE    = 80
HORIZON      = 400
SEED         = 42
# ───────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)


class ProbAdversaryTeammate:
    """
    Teammate that switches between normal and adversarial behavior each episode.
    - With prob `adv_prob`: uses ADV agent (adversarial)
    - Otherwise: uses normal SP agent (cooperative)
    Uses a property for encoding_fn so obs generation always matches the active agent.
    """
    def __init__(self, normal_agent, adv_agent, adv_prob=0.5):
        self.normal_agent = normal_agent
        self.adv_agent    = adv_agent
        self.adv_prob     = adv_prob
        self.is_adversarial = False
        self.policy = normal_agent.policy   # for env compatibility checks
        self.args   = normal_agent.args
        self.name   = f"ProbAdv(p={adv_prob:.1f})"

    @property
    def encoding_fn(self):
        return self.adv_agent.encoding_fn if self.is_adversarial else self.normal_agent.encoding_fn

    def new_episode(self):
        self.is_adversarial = random.random() < self.adv_prob

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        agent = self.adv_agent if self.is_adversarial else self.normal_agent
        return agent.predict(obs, state=state, deterministic=deterministic)

    def set_encoding_params(self, p_idx, horizon, env=None, mdp=None, **kwargs):
        self.normal_agent.set_encoding_params(p_idx, horizon, env=env, mdp=mdp, **kwargs)
        self.adv_agent.set_encoding_params(p_idx, horizon, env=env, mdp=mdp, **kwargs)

    def get_start_position(self, *args, **kwargs):
        return None


def load_best_agent(agent_dir: Path, args):
    """Load agent from best/ checkpoint directory."""
    candidate = agent_dir / 'agents_dir' / 'agent_0'
    path = candidate if candidate.exists() else agent_dir
    return load_agent(path, args)


def make_env(args):
    return OvercookedGymEnv(
        args=args,
        layout_name=LAYOUT,
        ret_completed_subtasks=False,
        is_eval_env=True,
        horizon=HORIZON,
        learner_type='originaler',
    )


def render_frame(env):
    surface = StateVisualizer(tile_size=TILE_SIZE).render_state(
        env.state, grid=env.env.mdp.terrain_mtx
    )
    arr = pygame.surfarray.array3d(surface)
    return np.transpose(arr, (1, 0, 2))   # (W,H,3) → (H,W,3)


def save_gif(frames, path: Path, duration=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        str(path), save_all=True, append_images=pil_frames[1:],
        optimize=False, duration=duration, loop=0
    )


def run_episode(env, main_agent, prob_adv, sp_teammate, render=False):
    """
    Run one episode.
    Returns: (total_reward, is_adversarial, frames)
    """
    prob_adv.new_episode()
    teammates = [sp_teammate, prob_adv]

    # set_teammates calls env.reset() internally
    env.set_teammates(teammates)
    obs = env.reset(p_idx=0)   # fix main agent to player 0 for consistency

    done = False
    total_reward = 0.0
    frames = [render_frame(env)] if render else []

    while not done:
        action = main_agent.predict(obs)[0]
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if render:
            frames.append(render_frame(env))

    return total_reward, prob_adv.is_adversarial, frames


def evaluate(label, main_agent, prob_adv, sp_teammate, args):
    """Run N_EVAL + N_RENDER episodes, collect stats and save GIFs."""
    env = make_env(args)
    env.set_teammates([sp_teammate, prob_adv])

    # Setup encoding params for all agents
    main_agent.set_encoding_params(0, HORIZON, env=env, is_haha=False, tune_subtasks=False)
    env.encoding_fn = main_agent.encoding_fn
    sp_teammate.set_encoding_params(1, HORIZON, env=env, is_haha=False, tune_subtasks=False)
    prob_adv.set_encoding_params(2, HORIZON, env=env, is_haha=False, tune_subtasks=False)

    all_rewards, adv_rewards, nrm_rewards = [], [], []
    gif_idx = 0

    print(f"\n{'='*55}")
    print(f"  Agent: {label}")
    print(f"  Episodes: {N_EVAL} eval  +  {N_RENDER} rendered")
    print(f"{'='*55}")

    total_episodes = N_EVAL + N_RENDER
    for ep in range(total_episodes):
        render = (ep < N_RENDER)
        reward, is_adv, frames = run_episode(env, main_agent, prob_adv, sp_teammate, render=render)

        all_rewards.append(reward)
        (adv_rewards if is_adv else nrm_rewards).append(reward)

        tag  = "ADV" if is_adv else "NRM"
        mark = " 🎬" if render else ""
        print(f"  ep {ep+1:3d} [{tag}]  reward={reward:6.1f}{mark}")

        if render and frames:
            gif_name = f"ep{ep+1:02d}_{tag}_rew{reward:.0f}.gif"
            gif_path = GIFS_DIR / label / gif_name
            save_gif(frames, gif_path)
            gif_idx += 1

    print(f"\n  ── {label} summary ──")
    print(f"  Overall        : {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}  (n={len(all_rewards)})")
    print(f"  Normal team    : {np.mean(nrm_rewards) if nrm_rewards else 0:.2f}  (n={len(nrm_rewards)})")
    print(f"  Adversary team : {np.mean(adv_rewards) if adv_rewards else 0:.2f}  (n={len(adv_rewards)})")
    print(f"  GIFs saved     : {gif_idx}  →  {GIFS_DIR / label}/")

    return {
        'all':  np.array(all_rewards),
        'nrm':  np.array(nrm_rewards) if nrm_rewards else np.array([0.0]),
        'adv':  np.array(adv_rewards) if adv_rewards else np.array([0.0]),
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GIFS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading agents...")

    sp_main   = load_best_agent(BASE_DIR / 'SP_hd256_seed1010' / 'best', args)
    spn_main  = load_best_agent(BASE_DIR / 'PWADV-N-1-SP_s1010_h256_tr[SP_SPADV]_ran_originaler_attack2' / 'best', args)
    adv_agent = load_best_agent(BASE_DIR / 'ADV_s68_h512_tr[H]_ran_selfisher_attack2' / 'best', args)

    # Load separate instances of SP teammate for each evaluation run
    def fresh_sp_tm():
        a = load_best_agent(BASE_DIR / 'SP_hd256_seed2020' / 'best', args)
        a.name = 'SP_tm'
        return a

    print("All agents loaded.\n")

    # ── Evaluate SP ────────────────────────────────────────────────────────────
    sp_results = evaluate(
        label       = 'SP',
        main_agent  = sp_main,
        prob_adv    = ProbAdversaryTeammate(fresh_sp_tm(), adv_agent, ADV_PROB),
        sp_teammate = fresh_sp_tm(),
        args        = args,
    )

    # ── Evaluate SPN_1ADV ──────────────────────────────────────────────────────
    spn_results = evaluate(
        label       = 'SPN_1ADV',
        main_agent  = spn_main,
        prob_adv    = ProbAdversaryTeammate(fresh_sp_tm(), adv_agent, ADV_PROB),
        sp_teammate = fresh_sp_tm(),
        args        = args,
    )

    # ── Final comparison ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL COMPARISON")
    print("="*60)
    fmt = "{:<35} {:>10} {:>12}"
    print(fmt.format("Metric", "SP", "SPN_1ADV"))
    print("-"*60)
    print(fmt.format(
        f"Overall mean reward (n={N_EVAL+N_RENDER})",
        f"{np.mean(sp_results['all']):.2f}",
        f"{np.mean(spn_results['all']):.2f}",
    ))
    print(fmt.format(
        "  ↳ Normal teammate",
        f"{np.mean(sp_results['nrm']):.2f}",
        f"{np.mean(spn_results['nrm']):.2f}",
    ))
    print(fmt.format(
        "  ↳ Adversarial teammate",
        f"{np.mean(sp_results['adv']):.2f}",
        f"{np.mean(spn_results['adv']):.2f}",
    ))

    delta_overall = np.mean(spn_results['all']) - np.mean(sp_results['all'])
    delta_adv     = np.mean(spn_results['adv']) - np.mean(sp_results['adv'])
    delta_nrm     = np.mean(spn_results['nrm']) - np.mean(sp_results['nrm'])

    print("\n  SPN_1ADV vs SP  (positive = SPN_1ADV better)")
    print(f"    Overall            : {delta_overall:+.2f}")
    print(f"    Under adversary    : {delta_adv:+.2f}")
    print(f"    Normal condition   : {delta_nrm:+.2f}")
    print(f"\n  GIFs → {GIFS_DIR}")
    print("="*60)


if __name__ == '__main__':
    main()
