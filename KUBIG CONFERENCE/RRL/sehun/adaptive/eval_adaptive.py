"""Evaluate SP / SPN_1ADV / Adaptive under normal and adversary conditions and
build the comparison table (전체평균 / 정상 / 적대적 / 강건성).

Adaptive: play SP for the first 100 steps while recording obs, run the RNN
detector, then keep SP (predicted normal) or switch to the robust SPN_1ADV
policy (predicted adversary) for the remaining steps.
"""
import argparse
import json
import sys

import numpy as np
import torch

import common as C
from train_detector import Detector


def team_reward(info):
    return float(sum(info.get("sparse_r_by_agent", [0])))


def run_fixed(env, ego, teammates, p_idx=0, deterministic=False):
    obs = C.setup_condition(env, [ego], teammates, p_idx=p_idx)
    score = 0.0
    for _ in range(C.HORIZON):
        action = C.act(ego, obs, env, deterministic=deterministic)
        obs, _, done, info = env.step(action)
        score += team_reward(info)
        if done:
            break
    return score


def run_adaptive(env, sp, robust, teammates, detfn, p_idx=0, deterministic=False):
    obs = C.setup_condition(env, [sp, robust], teammates, p_idx=p_idx)
    seq, score, pred, ego = [], 0.0, None, sp
    for t in range(C.HORIZON):
        if t < C.DETECT_STEPS:
            seq.append(C.flat_obs(obs))
            action = C.act(sp, obs, env, deterministic=deterministic)
        else:
            if pred is None:
                pred = detfn(np.stack(seq))
                ego = robust if pred == 1 else sp
            action = C.act(ego, obs, env, deterministic=deterministic)
        obs, _, done, info = env.step(action)
        score += team_reward(info)
        if done:
            break
    if pred is None:  # horizon <= DETECT_STEPS safety
        pred = detfn(np.stack(seq))
    return score, int(pred)


def make_detector(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    cfg = ck["config"]
    model = Detector(in_dim=cfg["in_dim"], emb=cfg["emb"], hidden=cfg["hidden"], layers=cfg["layers"]).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    mean = torch.tensor(ck["mean"], device=dev)
    std = torch.tensor(ck["std"], device=dev)

    def detfn(seq):  # seq: (T, D) numpy
        x = (torch.tensor(seq, device=dev, dtype=torch.float32) - mean) / std
        with torch.no_grad():
            return int(model(x.unsqueeze(0)).argmax(1).item())

    return detfn, float(ck.get("val_acc", -1))


def teammates_for(condition, sp_list, adv, i):
    if condition == "normal":
        return [sp_list[i % len(sp_list)], sp_list[(i + 1) % len(sp_list)]]
    sp_tm = sp_list[i % len(sp_list)]
    return [sp_tm, adv] if i % 2 == 0 else [adv, sp_tm]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--detector", default=C.DETECTOR_PATH)
    ap.add_argument("--out", default=f"{C.DATA_DIR}/eval_results.json")
    ap.add_argument("--seed", type=int, default=1)
    cli = ap.parse_args()
    sys.argv = [sys.argv[0]]
    np.random.seed(cli.seed)

    a = C.make_args()
    ego_sp, robust, adv, sp_pool = C.load_all(a)
    sp_list = list(sp_pool.values())
    env = C.build_env()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    detfn, det_val_acc = make_detector(cli.detector, dev)
    print(f"detector loaded (val_acc={det_val_acc:.4f}) dev={dev}", flush=True)

    conditions = ["normal", "adversary"]
    methods = ["SP", "SPN_1ADV", "Adaptive"]
    scores = {m: {c: [] for c in conditions} for m in methods}
    det_correct, det_total = 0, 0

    for c in conditions:
        for i in range(cli.episodes):
            tms = teammates_for(c, sp_list, adv, i)
            scores["SP"][c].append(run_fixed(env, ego_sp, tms))
            scores["SPN_1ADV"][c].append(run_fixed(env, robust, tms))
            s, pred = run_adaptive(env, ego_sp, robust, tms, detfn)
            scores["Adaptive"][c].append(s)
            det_total += 1
            det_correct += int(pred == (1 if c == "adversary" else 0))
            if (i + 1) % 10 == 0:
                print(f"[{c}] {i + 1}/{cli.episodes}", flush=True)

    def m(v):
        return float(np.mean(v)) if v else 0.0

    table = {}
    for meth in methods:
        nrm = m(scores[meth]["normal"])
        adver = m(scores[meth]["adversary"])
        overall = m(scores[meth]["normal"] + scores[meth]["adversary"])
        rob = (adver / nrm * 100) if nrm > 0 else 0.0
        table[meth] = {"overall": overall, "normal": nrm, "adversary": adver, "robustness_pct": rob}

    print("\n================ RESULTS ================")
    print(f"episodes/condition={cli.episodes}  detector eval-time acc={det_correct}/{det_total}="
          f"{det_correct / max(det_total,1):.3f}")
    print(f"{'method':<10}{'전체평균':>10}{'정상':>10}{'적대적':>10}{'강건성':>10}")
    for meth in methods:
        r = table[meth]
        print(f"{meth:<10}{r['overall']:>10.2f}{r['normal']:>10.2f}{r['adversary']:>10.2f}{r['robustness_pct']:>9.1f}%")
    print("=========================================")

    with open(cli.out, "w") as f:
        json.dump({"episodes": cli.episodes, "detector_val_acc": det_val_acc,
                   "detector_eval_acc": det_correct / max(det_total, 1),
                   "table": table, "raw": scores}, f, indent=2)
    print(f"SAVED {cli.out}", flush=True)


if __name__ == "__main__":
    main()
