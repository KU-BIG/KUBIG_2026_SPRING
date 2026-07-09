"""Shared helpers for the Adaptive-Method experiment (Detector-based policy switching).

Reuses the rollout pattern from scripts/headless_visualize.py.
Ego observation is a dict {'visual_obs': (27,7,7) int}. We flatten to 1323-dim per step.
Team reward per step = sum(info['sparse_r_by_agent']).
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

REPO = os.environ.get("MULTIHRI_ROOT", "/workspace/rl_project/multiHRI")
os.chdir(REPO)  # make all relative paths (layouts, planner cache, agent dirs) resolve

from pathlib import Path
import numpy as np

from oai_agents.agents.agent_utils import load_agent
from oai_agents.common.arguments import get_arguments
from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv

LAYOUT = "dec_3_chefs_secret_heaven"
HORIZON = 400
DETECT_STEPS = 100  # first 100/400 steps used for detection
OBS_DIM = 27 * 7 * 7  # 1323

BASE = f"{REPO}/agent_models/SecretHeaven_N3"
SP_SEEDS = [1010, 13, 2020, 2602]
SP_PATHS = {s: f"{BASE}/SP_hd256_seed{s}/best" for s in SP_SEEDS}
ADV_PATH = f"{BASE}/ADV_s68_h512_tr[H]_ran_selfisher_attack2/best"
ROBUST_PATH = f"{BASE}/PWADV-N-1-SP_s1010_h256_tr[SP_SPADV]_ran_originaler_attack2/best"

DATA_DIR = f"{REPO}/adaptive/data"
DETECTOR_PATH = f"{REPO}/adaptive/detector.pt"


def make_args():
    args = get_arguments()
    args.num_players = 3
    args.layout = LAYOUT
    args.layout_names = [LAYOUT]
    args.n_envs = 1
    args.p_idx = 0
    return args


def load_all(args, sp_seed_for_ego=1010):
    """Load ego SP, robust (SPN_1ADV), adversary, and full SP pool."""
    sp_pool = {s: load_agent(Path(SP_PATHS[s]), args) for s in SP_SEEDS}
    for s, a in sp_pool.items():
        a.name = f"SP_seed{s}"
    ego_sp = sp_pool[sp_seed_for_ego]
    robust = load_agent(Path(ROBUST_PATH), args)
    robust.name = "SPN_1ADV"
    adv = load_agent(Path(ADV_PATH), args)
    adv.name = "ADV"
    return ego_sp, robust, adv, sp_pool


def build_env():
    return OvercookedGymEnv(layout_name=LAYOUT, args=make_args(), ret_completed_subtasks=False,
                            is_eval_env=True, horizon=HORIZON, learner_type="originaler")


def set_encoding_for(agent, idx, env, is_ego):
    agent.set_encoding_params(idx, HORIZON, env=env, is_haha=False,
                              tune_subtasks=(not is_ego))


def setup_condition(env, ego_agents, teammates, p_idx=0):
    """Set teammates and prime encoding params for every acting agent.

    ego_agents: list of agents that may act as ego (index p_idx) during the
    episode (e.g. [sp, robust] for adaptive). They share the visual_obs
    encoding, so env.encoding_fn is taken from the first one.
    """
    env.set_teammates(teammates)
    obs = env.reset(p_idx=p_idx)
    for a in ego_agents:
        set_encoding_for(a, p_idx, env, is_ego=True)
    env.encoding_fn = ego_agents[0].encoding_fn
    for t_idx, tm in enumerate(env.teammates):
        set_encoding_for(tm, t_idx + 1, env, is_ego=False)
    return obs


def flat_obs(obs):
    return np.asarray(obs["visual_obs"], dtype=np.float32).reshape(-1)


def act(agent, obs, env, deterministic=False):
    return agent.predict(obs, state=env.state, deterministic=deterministic)[0]
