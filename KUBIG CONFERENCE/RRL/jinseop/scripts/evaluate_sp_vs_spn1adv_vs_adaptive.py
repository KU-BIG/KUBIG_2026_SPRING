"""
Evaluation: SP vs SPN_1ADV vs Adaptive (SP -> LSTM team-type classifier -> {SP, PWADV_advonly}).
Each agent plays with clones of itself as teammates (extends evaluate_sp_vs_spn1adv_selfplay.py to 3 conditions).

Test setup:
  - 3-player cooperative game (3_chefs_counter_circuit)
  - Main agent + 2 clones of itself as teammates:
      Teammate 1: always a clone (cooperative)
      Teammate 2: probabilistic -- acts as ADV with prob=ADV_PROB, else clone
  - SP       : SP_s1010_h256_tr[SP]_ran
  - SPN_1ADV : PWADV-N-1-SP_s1010_h256_tr[SP_SPADV]_ran_originaler_attack2
  - Adaptive : first 100 steps uses SP_s1010_h256_tr[SP]_ran, then an LSTM
               (team_type_classifier attempt_1) predicts SP vs SPADV from the
               3 players' observations over those 100 steps, and the remaining
               steps use SP_s1010_h256_tr[SP]_ran (if predicted SP) or
               PWADV-N-1-SP_s1010_h256_tr[SPADV]_ran_originaler_attack2_advonly
               (if predicted SPADV).

Outputs:
  - Per-episode rewards split by normal / adversary condition
  - Summary comparison table (3 columns)
  - 10 GIF videos per condition -> eval_results/gifs_3way/
"""

import os
import sys
import random
import numpy as np
import torch
from pathlib import Path
from PIL import Image

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import pygame
pygame.init()
pygame.display.set_mode((1, 1))

sys.path.insert(0, '/workspace')

from oai_agents.common.arguments import get_arguments
from oai_agents.agents.agent_utils import load_agent
from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from scripts.team_type_classifier import TeamTypeLSTMClassifier

# --- Config -----------------------------------------------------------------
LAYOUT   = '3_chefs_counter_circuit'
BASE_DIR = Path('/workspace/agent_models/Classic/3')
GIFS_DIR = Path('/workspace/eval_results/gifs_3way_v2_switch50')
ADV_PROB = 0.5
N_EVAL   = 190
N_RENDER = 10
TILE_SIZE = 80
HORIZON  = 400
SEED     = 42
SWITCH_STEP = 50

SP_PATH      = BASE_DIR / 'SP_s1010_h256_tr[SP]_ran' / 'best'
SPN_PATH     = BASE_DIR / 'PWADV-N-1-SP_s1010_h256_tr[SP_SPADV]_ran_originaler_attack2' / 'best'
ADV_PATH     = BASE_DIR / 'ADV_s68_h512_tr[H]_ran_selfisher_attack2' / 'best'
ADVONLY_PATH = BASE_DIR / 'PWADV-N-1-SP_s1010_h256_tr[SPADV]_ran_originaler_attack2_advonly' / 'best'
CLASSIFIER_PATH = Path('/workspace/team_type_classifier_runs_ep50/attempt_1/model.pt')
# ------------------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)


class ProbAdversaryTeammate:
    """
    Teammate that switches between clone and adversary each episode.
    - With prob `adv_prob`: uses trained ADV agent (adversarial)
    - Otherwise: uses clone of the main agent (cooperative)
    """
    def __init__(self, clone_agent, adv_agent, adv_prob=0.5):
        self.clone = clone_agent
        self.adv = adv_agent
        self.adv_prob = adv_prob
        self.is_adversarial = False
        self.policy = clone_agent.policy
        self.args = clone_agent.args
        self.name = f"ProbAdv(p={adv_prob:.1f})"

    @property
    def encoding_fn(self):
        return self.adv.encoding_fn if self.is_adversarial else self.clone.encoding_fn

    def new_episode(self, is_adversarial=None):
        self.is_adversarial = (random.random() < self.adv_prob) if is_adversarial is None else is_adversarial
        if hasattr(self.clone, 'reset_episode'):
            self.clone.reset_episode()

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        agent = self.adv if self.is_adversarial else self.clone
        return agent.predict(obs, state=state, deterministic=deterministic)

    def set_encoding_params(self, p_idx, horizon, env=None, mdp=None, **kwargs):
        self.clone.set_encoding_params(p_idx, horizon, env=env, mdp=mdp, **kwargs)
        self.adv.set_encoding_params(p_idx, horizon, env=env, mdp=mdp, **kwargs)

    def get_start_position(self, *args, **kwargs):
        return None


class AdaptiveAgent:
    """
    Plays as `sp_agent` for the first `switch_step` steps of an episode, while
    recording all 3 players' observations (in fixed board-position order,
    matching how the team-type classifier's training data was built). At
    `switch_step`, runs the classifier once on the recorded sequence and
    switches to `sp_agent` (predicted SP) or `switch_agent` (predicted SPADV)
    for the rest of the episode.
    """
    def __init__(self, sp_agent, switch_agent, classifier, device, switch_step=SWITCH_STEP):
        self.sp_agent = sp_agent
        self.switch_agent = switch_agent
        self.classifier = classifier
        self.device = device
        self.switch_step = switch_step
        self.policy = sp_agent.policy
        self.args = sp_agent.args
        self.name = 'Adaptive'
        self.env = None
        self.last_prediction = None
        self.reset_episode()

    def set_env(self, env):
        self.env = env

    def reset_episode(self):
        self.step_count = 0
        self.obs_buffer = []
        self.active_agent = self.sp_agent
        self.last_prediction = None

    def get_start_position(self, *args, **kwargs):
        return None

    @property
    def encoding_fn(self):
        return self.active_agent.encoding_fn

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        if self.step_count < self.switch_step:
            assert self.env is not None, 'AdaptiveAgent.set_env(env) must be called before use'
            o0 = self.env.get_obs(0)['visual_obs']
            o1 = self.env.get_obs(1)['visual_obs']
            o2 = self.env.get_obs(2)['visual_obs']
            self.obs_buffer.append(np.stack([o0, o1, o2], axis=0))

        action, st = self.active_agent.predict(obs, state=state, episode_start=episode_start,
                                                deterministic=deterministic)
        self.step_count += 1
        if self.step_count == self.switch_step:
            self._classify_and_switch()
        return action, st

    def _classify_and_switch(self):
        seq = np.stack(self.obs_buffer, axis=0).astype(np.float32)  # (T, 3, C, 7, 7)
        x = seq.reshape(1, seq.shape[0], -1)
        x_t = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            logit = self.classifier(x_t)
            pred = int(torch.sigmoid(logit).item() > 0.5)
        self.last_prediction = pred
        self.active_agent = self.sp_agent if pred == 0 else self.switch_agent


def load_best_agent(agent_dir: Path, args):
    candidate = agent_dir / 'agents_dir' / 'agent_0'
    path = candidate if candidate.exists() else agent_dir
    return load_agent(path, args)


def make_env(args):
    return OvercookedGymEnv(
        args=args, layout_name=LAYOUT, ret_completed_subtasks=False,
        is_eval_env=True, horizon=HORIZON, learner_type='originaler',
    )


def render_frame(env):
    surface = StateVisualizer(tile_size=TILE_SIZE).render_state(env.state, grid=env.env.mdp.terrain_mtx)
    arr = pygame.surfarray.array3d(surface)
    return np.transpose(arr, (1, 0, 2))


def save_gif(frames, path: Path, duration=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(str(path), save_all=True, append_images=pil_frames[1:],
                        optimize=False, duration=duration, loop=0)


def evaluate(label, main_agent, clone_tm, prob_adv, args, extra_env_setup=None, episode_plan=None):
    env = make_env(args)
    if extra_env_setup:
        extra_env_setup(env)
    env.set_teammates([clone_tm, prob_adv])

    all_r, adv_r, nrm_r = [], [], []
    adv_predictions = []  # (is_adversarial_true, predicted_label) for Adaptive condition, else None
    gif_n = 0

    print(f"\n{'=' * 57}")
    print(f"  Agent : {label}")
    print(f"  Team  : {label}_clone x 2  (1 may become ADV p={ADV_PROB})")
    if episode_plan is not None:
        print(f"  Using fixed episode plan ({sum(episode_plan)}/{len(episode_plan)} adversarial)")
    print(f"{'=' * 57}")

    for ep in range(N_EVAL + N_RENDER):
        render = ep < N_RENDER
        prob_adv.new_episode(is_adversarial=None if episode_plan is None else episode_plan[ep])
        if hasattr(main_agent, 'reset_episode'):
            main_agent.reset_episode()
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
        if hasattr(main_agent, 'last_prediction'):
            adv_predictions.append((prob_adv.is_adversarial, main_agent.last_prediction))

        tag = "ADV" if prob_adv.is_adversarial else "NRM"
        mark = " \U0001F3AC" if render else ""
        pred_str = ''
        if hasattr(main_agent, 'last_prediction'):
            pred_str = f"  pred={'SPADV' if main_agent.last_prediction == 1 else 'SP'}"
        print(f"  ep {ep + 1:3d} [{tag}]  reward={total:6.1f}{pred_str}{mark}")

        if render and frames:
            path = GIFS_DIR / label / f"ep{ep + 1:02d}_{tag}_rew{total:.0f}.gif"
            save_gif(frames, path)
            gif_n += 1

    nrm_mean = np.mean(nrm_r) if nrm_r else 0.0
    adv_mean = np.mean(adv_r) if adv_r else 0.0
    print(f"\n  -- {label} --")
    print(f"  Overall        : {np.mean(all_r):.2f} +- {np.std(all_r):.2f}  (n={len(all_r)})")
    print(f"  Normal team    : {nrm_mean:.2f}  (n={len(nrm_r)})")
    print(f"  Adversary team : {adv_mean:.2f}  (n={len(adv_r)})")
    print(f"  Robustness     : {adv_mean / nrm_mean * 100:.1f}%  |  zero-reward: {adv_r.count(0)} / {len(adv_r)}")
    print(f"  GIFs saved     : {gif_n}  ->  {GIFS_DIR / label}/")

    if adv_predictions:
        correct = sum(1 for true_adv, pred in adv_predictions if pred is not None and int(true_adv) == pred)
        total_pred = sum(1 for _, pred in adv_predictions if pred is not None)
        if total_pred:
            print(f"  Classifier acc  : {correct}/{total_pred} ({correct / total_pred * 100:.1f}%) "
                  f"during these episodes")

    return {
        'all': np.array(all_r),
        'nrm': np.array(nrm_r) if nrm_r else np.array([0.]),
        'adv': np.array(adv_r) if adv_r else np.array([0.]),
        'zero_adv': adv_r.count(0),
        'n_adv': len(adv_r),
    }


def main():
    sys.argv = [
        'eval', '--layout-names', LAYOUT, '--num-players', '3', '--teammates-len', '2',
        '--n-envs', '1', '--horizon', str(HORIZON),
    ]
    args = get_arguments()
    args.base_dir = Path('/workspace')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    GIFS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading agents...")
    sp_agent = load_best_agent(SP_PATH, args)
    spn_agent = load_best_agent(SPN_PATH, args)
    adv_agent = load_best_agent(ADV_PATH, args)

    def fresh(path):
        return load_best_agent(path, args)

    # --- Adaptive condition setup ---
    num_channels = 27  # base(17) + ego_pos(5) + teammates_pos(5), matches OvercookedGymEnv.num_enc_channels
    input_dim = 3 * num_channels * 7 * 7
    classifier = TeamTypeLSTMClassifier(input_dim=input_dim, embed_dim=256, hidden_dim=256,
                                         num_layers=1, dropout=0.1, bidirectional=False).to(device)
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    classifier.eval()
    print(f"Loaded team-type classifier from {CLASSIFIER_PATH}")

    def make_adaptive():
        return AdaptiveAgent(sp_agent=fresh(SP_PATH), switch_agent=fresh(ADVONLY_PATH),
                              classifier=classifier, device=device)

    print("All agents loaded.\n")

    sp_res = evaluate(
        label='SP', main_agent=sp_agent, clone_tm=fresh(SP_PATH),
        prob_adv=ProbAdversaryTeammate(fresh(SP_PATH), adv_agent, ADV_PROB), args=args,
    )

    spn_res = evaluate(
        label='SPN_1ADV', main_agent=spn_agent, clone_tm=fresh(SPN_PATH),
        prob_adv=ProbAdversaryTeammate(fresh(SPN_PATH), adv_agent, ADV_PROB), args=args,
    )

    adaptive_main = make_adaptive()
    adaptive_clone_tm = make_adaptive()
    adaptive_prob_clone = make_adaptive()

    def wire_envs(env):
        adaptive_main.set_env(env)
        adaptive_clone_tm.set_env(env)
        adaptive_prob_clone.set_env(env)

    adaptive_res = evaluate(
        label='Adaptive', main_agent=adaptive_main, clone_tm=adaptive_clone_tm,
        prob_adv=ProbAdversaryTeammate(adaptive_prob_clone, adv_agent, ADV_PROB), args=args,
        extra_env_setup=wire_envs,
    )

    labels = ['SP', 'SPN_1ADV', 'Adaptive']
    results = [sp_res, spn_res, adaptive_res]

    print("\n" + "=" * 78)
    print("  FINAL COMPARISON  (each agent plays with clones of itself)")
    print("=" * 78)
    fmt = "{:<28} {:>12} {:>14} {:>14}"
    print(fmt.format("Metric", *labels))
    print("-" * 78)
    print(fmt.format(f"Overall (n={N_EVAL + N_RENDER})", *[f"{np.mean(r['all']):.2f}" for r in results]))
    print(fmt.format("  Normal teammate", *[f"{np.mean(r['nrm']):.2f}" for r in results]))
    print(fmt.format("  Adversarial teammate", *[f"{np.mean(r['adv']):.2f}" for r in results]))
    print(fmt.format("Robustness (ADV/NRM)",
                      *[f"{np.mean(r['adv']) / np.mean(r['nrm']) * 100:.1f}%" for r in results]))
    print(fmt.format("Zero-reward under ADV", *[f"{r['zero_adv']}/{r['n_adv']}" for r in results]))
    print(f"\n  GIFs -> {GIFS_DIR}")
    print("=" * 78)

    summary = {
        labels[i]: {
            'overall_mean': float(np.mean(results[i]['all'])),
            'nrm_mean': float(np.mean(results[i]['nrm'])),
            'adv_mean': float(np.mean(results[i]['adv'])),
            'robustness_pct': float(np.mean(results[i]['adv']) / np.mean(results[i]['nrm']) * 100),
            'zero_adv': f"{results[i]['zero_adv']}/{results[i]['n_adv']}",
        } for i in range(3)
    }
    import json
    out_path = Path('/workspace/eval_results/three_way_comparison_v2_switch50.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {out_path}")


if __name__ == '__main__':
    main()
