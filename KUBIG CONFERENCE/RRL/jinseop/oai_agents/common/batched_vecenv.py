"""
BatchedTeammatesDummyVecEnv: replaces n_envs*n_teammates individual GPU calls
per step with 1 batched GPU call per teammate position (~3 calls for 3-player).
"""
from copy import deepcopy

import numpy as np
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv


class BatchedTeammatesDummyVecEnv(DummyVecEnv):

    def step_wait(self):
        if not getattr(self.envs[0], 'teammates', None):
            return super().step_wait()

        # Collect obs from all envs, grouped by player position (t_idx)
        # {t_idx: [(env_idx, obs), ...]}, tm_ref: {t_idx: teammate}
        tm_groups: dict = {}
        tm_ref: dict = {}

        for env_idx, env in enumerate(self.envs):
            if not env.teammates:
                continue
            for t_idx, (obs, teammate) in env.collect_teammate_obs().items():
                if t_idx not in tm_groups:
                    tm_groups[t_idx] = []
                    tm_ref[t_idx] = teammate
                tm_groups[t_idx].append((env_idx, obs))

        # One batched GPU forward pass per player position
        precomputed = [{} for _ in range(self.num_envs)]

        for t_idx, data_list in tm_groups.items():
            teammate = tm_ref[t_idx]
            obs_keys = list(data_list[0][1].keys())
            batched_obs = {k: np.stack([d[1][k] for d in data_list]) for k in obs_keys}

            with th.no_grad():
                actions, _ = teammate.predict(obs=batched_obs, deterministic=False)

            for i, (env_idx, _) in enumerate(data_list):
                a = actions[i]
                precomputed[env_idx][t_idx] = int(a) if hasattr(a, '__int__') else a

        # Inject pre-computed actions into the underlying env, then step normally
        for env_idx in range(self.num_envs):
            # Unwrap through Monitor (or any Wrapper) to reach OvercookedGymEnv
            env = self.envs[env_idx]
            underlying = env.env if hasattr(env, 'env') else env
            underlying._precomputed_tm_actions = precomputed[env_idx] or None

            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] = \
                env.step(self.actions[env_idx])
            if self.buf_dones[env_idx]:
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs = env.reset()
            self._save_obs(env_idx, obs)

        return (self._obs_from_buf(), np.copy(self.buf_rews),
                np.copy(self.buf_dones), deepcopy(self.buf_infos))
