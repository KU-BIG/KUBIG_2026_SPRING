"""Generate detector training data.

Roll out episodes with ego=SP for the first DETECT_STEPS(=100) steps and record
the ego's visual_obs sequence, labelled by whether an adversary is on the team.

  label 0 (normal / 30-team-type):    teammates = [SP, SP]
  label 1 (adversary / 21-team-type): teammates = [SP, ADV]  (adv at idx 1 or 2)

Saves adaptive/data/detector_dataset.npz with X (N, 100, 1323) float32, y (N,).
"""
import argparse
import os
import sys
import time

import numpy as np

import common as C


def record_episode(env, ego, teammates, p_idx=0, steps=C.DETECT_STEPS, deterministic=False):
    obs = C.setup_condition(env, [ego], teammates, p_idx=p_idx)
    seq = []
    for _ in range(steps):
        seq.append(C.flat_obs(obs))
        action = C.act(ego, obs, env, deterministic=deterministic)
        obs, _, done, _ = env.step(action)
        if done:
            break
    while len(seq) < steps:  # pad (episode shouldn't end within 100 of 400)
        seq.append(seq[-1])
    return np.stack(seq, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_class", type=int, default=200)
    ap.add_argument("--out", default=f"{C.DATA_DIR}/detector_dataset.npz")
    ap.add_argument("--seed", type=int, default=0)
    cli = ap.parse_args()
    sys.argv = [sys.argv[0]]  # neutralize argv so get_arguments() uses defaults
    np.random.seed(cli.seed)

    a = C.make_args()
    ego_sp, _robust, adv, sp_pool = C.load_all(a)
    env = C.build_env()
    sp_list = list(sp_pool.values())

    X, y = [], []
    t0 = time.time()
    for cls in (0, 1):
        for i in range(cli.per_class):
            sp_tm = sp_list[np.random.randint(len(sp_list))]
            if cls == 0:
                sp_tm2 = sp_list[np.random.randint(len(sp_list))]
                teammates = [sp_tm, sp_tm2]
            else:
                teammates = [sp_tm, adv] if i % 2 == 0 else [adv, sp_tm]
            seq = record_episode(env, ego_sp, teammates)
            X.append(seq)
            y.append(cls)
            if (i + 1) % 20 == 0:
                print(f"cls={cls} {i + 1}/{cli.per_class} elapsed={time.time() - t0:.0f}s", flush=True)

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    os.makedirs(os.path.dirname(cli.out), exist_ok=True)
    np.savez_compressed(cli.out, X=X, y=y)
    print(f"SAVED {cli.out} X={X.shape} y={y.shape} pos={int(y.sum())} time={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
