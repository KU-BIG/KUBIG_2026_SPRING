"""
Evaluates PWADV-N-1-SP_s1010_h256_tr[SPADV]_ran_originaler_attack2_advonly on its
own (self-play: 2 clones of itself + 1 probabilistic adversary), using the exact
same methodology (N, ADV_PROB, HORIZON, SEED) as evaluate_sp_vs_spn1adv_vs_adaptive.py,
so the number is directly comparable to the SP / SPN_1ADV / Adaptive rows.
"""
import sys
from pathlib import Path

sys.path.insert(0, '/workspace')

import numpy as np

import scripts.evaluate_sp_vs_spn1adv_vs_adaptive as ev

ev.GIFS_DIR = Path('/workspace/eval_results/gifs_3way_v2_switch50')  # reuse same GIFs root


def main():
    sys.argv = ['eval', '--layout-names', ev.LAYOUT, '--num-players', '3', '--teammates-len', '2',
                '--n-envs', '1', '--horizon', str(ev.HORIZON)]
    args = ev.get_arguments()
    args.base_dir = Path('/workspace')

    print("Loading agents...")
    advonly_agent = ev.load_best_agent(ev.ADVONLY_PATH, args)
    adv_agent = ev.load_best_agent(ev.ADV_PATH, args)

    def fresh(path):
        return ev.load_best_agent(path, args)

    print("All agents loaded.\n")

    res = ev.evaluate(
        label='PWADV_advonly', main_agent=advonly_agent, clone_tm=fresh(ev.ADVONLY_PATH),
        prob_adv=ev.ProbAdversaryTeammate(fresh(ev.ADVONLY_PATH), adv_agent, ev.ADV_PROB), args=args,
    )

    print("\n" + "=" * 60)
    print(f"  PWADV_advonly (n={ev.N_EVAL + ev.N_RENDER})")
    print("=" * 60)
    print(f"  Overall              : {np.mean(res['all']):.2f}")
    print(f"  Normal teammate      : {np.mean(res['nrm']):.2f}")
    print(f"  Adversarial teammate : {np.mean(res['adv']):.2f}")
    print(f"  Robustness (ADV/NRM) : {np.mean(res['adv']) / np.mean(res['nrm']) * 100:.1f}%")
    print(f"  Zero-reward under ADV: {res['zero_adv']}/{res['n_adv']}")

    import json
    summary = {
        'overall_mean': float(np.mean(res['all'])),
        'nrm_mean': float(np.mean(res['nrm'])),
        'adv_mean': float(np.mean(res['adv'])),
        'robustness_pct': float(np.mean(res['adv']) / np.mean(res['nrm']) * 100),
        'zero_adv': f"{res['zero_adv']}/{res['n_adv']}",
    }
    out_path = Path('/workspace/eval_results/advonly_standalone.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
