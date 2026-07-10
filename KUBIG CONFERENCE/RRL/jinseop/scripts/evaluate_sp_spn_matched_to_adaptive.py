"""
Re-evaluates SP and SPN_1ADV using the EXACT same per-episode NRM/ADV plan that
the Adaptive condition experienced in the previous 3-way run (parsed from
/workspace/logs/evaluate_3way.log), so all 3 conditions are compared on
matched/paired episodes instead of independently-drawn random sequences.
Reuses Adaptive's already-computed results (they are the reference plan).
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, '/workspace')

import numpy as np

from scripts.evaluate_sp_vs_spn1adv_vs_adaptive import (
    get_arguments, ProbAdversaryTeammate, load_best_agent, evaluate,
    SP_PATH, SPN_PATH, ADV_PATH, N_EVAL, N_RENDER, ADV_PROB, GIFS_DIR,
)

LOG_PATH = Path('/workspace/logs/evaluate_3way.log')
SUMMARY_PATH = Path('/workspace/eval_results/three_way_comparison.json')


def parse_adaptive_episode_plan(log_path):
    text = log_path.read_text()
    start = text.index('Agent : Adaptive')
    end = text.index('FINAL COMPARISON', start)
    section = text[start:end]
    tags = re.findall(r'ep\s+\d+\s+\[(NRM|ADV)\]', section)
    assert len(tags) == N_EVAL + N_RENDER, f'Expected {N_EVAL + N_RENDER} episodes, found {len(tags)}'
    return [t == 'ADV' for t in tags]


def main():
    episode_plan = parse_adaptive_episode_plan(LOG_PATH)
    print(f'Parsed fixed episode plan from Adaptive run: {sum(episode_plan)}/{len(episode_plan)} adversarial')

    sys.argv = [
        'eval', '--layout-names', '3_chefs_counter_circuit', '--num-players', '3',
        '--teammates-len', '2', '--n-envs', '1', '--horizon', '400',
    ]
    args = get_arguments()
    args.base_dir = Path('/workspace')

    print("Loading agents...")
    sp_agent = load_best_agent(SP_PATH, args)
    spn_agent = load_best_agent(SPN_PATH, args)
    adv_agent = load_best_agent(ADV_PATH, args)

    def fresh(path):
        return load_best_agent(path, args)

    print("All agents loaded.\n")

    sp_res = evaluate(
        label='SP_matched', main_agent=sp_agent, clone_tm=fresh(SP_PATH),
        prob_adv=ProbAdversaryTeammate(fresh(SP_PATH), adv_agent, ADV_PROB), args=args,
        episode_plan=episode_plan,
    )

    spn_res = evaluate(
        label='SPN_1ADV_matched', main_agent=spn_agent, clone_tm=fresh(SPN_PATH),
        prob_adv=ProbAdversaryTeammate(fresh(SPN_PATH), adv_agent, ADV_PROB), args=args,
        episode_plan=episode_plan,
    )

    with open(SUMMARY_PATH) as f:
        prior_summary = json.load(f)
    adaptive = prior_summary['Adaptive']

    labels = ['SP', 'SPN_1ADV', 'Adaptive']
    results = {
        'SP': sp_res,
        'SPN_1ADV': spn_res,
    }

    def row(metric_fn):
        vals = [metric_fn(results['SP']), metric_fn(results['SPN_1ADV'])]
        return vals

    print("\n" + "=" * 78)
    print("  FINAL COMPARISON (matched episode plan: same NRM/ADV sequence for all 3)")
    print("=" * 78)
    fmt = "{:<28} {:>12} {:>14} {:>14}"
    print(fmt.format("Metric", *labels))
    print("-" * 78)
    print(fmt.format(f"Overall (n={N_EVAL + N_RENDER})",
                      f"{np.mean(sp_res['all']):.2f}", f"{np.mean(spn_res['all']):.2f}",
                      f"{adaptive['overall_mean']:.2f}"))
    print(fmt.format("  Normal teammate",
                      f"{np.mean(sp_res['nrm']):.2f}", f"{np.mean(spn_res['nrm']):.2f}",
                      f"{adaptive['nrm_mean']:.2f}"))
    print(fmt.format("  Adversarial teammate",
                      f"{np.mean(sp_res['adv']):.2f}", f"{np.mean(spn_res['adv']):.2f}",
                      f"{adaptive['adv_mean']:.2f}"))
    print(fmt.format("Robustness (ADV/NRM)",
                      f"{np.mean(sp_res['adv']) / np.mean(sp_res['nrm']) * 100:.1f}%",
                      f"{np.mean(spn_res['adv']) / np.mean(spn_res['nrm']) * 100:.1f}%",
                      f"{adaptive['robustness_pct']:.1f}%"))
    print(fmt.format("Zero-reward under ADV",
                      f"{sp_res['zero_adv']}/{sp_res['n_adv']}",
                      f"{spn_res['zero_adv']}/{spn_res['n_adv']}",
                      adaptive['zero_adv']))
    print("=" * 78)

    summary = {
        'SP': {
            'overall_mean': float(np.mean(sp_res['all'])), 'nrm_mean': float(np.mean(sp_res['nrm'])),
            'adv_mean': float(np.mean(sp_res['adv'])),
            'robustness_pct': float(np.mean(sp_res['adv']) / np.mean(sp_res['nrm']) * 100),
            'zero_adv': f"{sp_res['zero_adv']}/{sp_res['n_adv']}",
        },
        'SPN_1ADV': {
            'overall_mean': float(np.mean(spn_res['all'])), 'nrm_mean': float(np.mean(spn_res['nrm'])),
            'adv_mean': float(np.mean(spn_res['adv'])),
            'robustness_pct': float(np.mean(spn_res['adv']) / np.mean(spn_res['nrm']) * 100),
            'zero_adv': f"{spn_res['zero_adv']}/{spn_res['n_adv']}",
        },
        'Adaptive': adaptive,
        'episode_plan_adv_indices': [i for i, v in enumerate(episode_plan) if v],
    }
    out_path = Path('/workspace/eval_results/three_way_comparison_matched.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {out_path}")


if __name__ == '__main__':
    main()
