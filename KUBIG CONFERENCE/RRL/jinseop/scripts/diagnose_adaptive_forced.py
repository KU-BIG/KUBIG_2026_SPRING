"""
Runs the REAL AdaptiveAgent/evaluate() code (unmodified, from
evaluate_sp_vs_spn1adv_vs_adaptive.py) for N forced-adversarial episodes
(episode_plan=[True]*N), to check whether the low 65.93 adversarial-mean
reward seen in the mixed 60-episode run reflects genuine variance or an
implementation issue -- as opposed to the hand-rolled diagnose_adaptive_gap.py
script, which reimplements switching logic itself.
"""
import sys
from pathlib import Path

sys.path.insert(0, '/workspace')

import numpy as np
import torch

import scripts.evaluate_sp_vs_spn1adv_vs_adaptive as ev

N = 30
ev.N_EVAL = N
ev.N_RENDER = 0
ev.GIFS_DIR = Path('/tmp/claude-0/-workspace/536bb632-b941-44d7-97b0-1ef8c1e61fa9/scratchpad/gifs_forced_check')


def main():
    sys.argv = ['eval', '--layout-names', ev.LAYOUT, '--num-players', '3', '--teammates-len', '2',
                '--n-envs', '1', '--horizon', str(ev.HORIZON)]
    args = ev.get_arguments()
    args.base_dir = Path('/workspace')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading agents...")
    sp_agent = ev.load_best_agent(ev.SP_PATH, args)

    def fresh(path):
        return ev.load_best_agent(path, args)

    adv_agent = ev.load_best_agent(ev.ADV_PATH, args)

    num_channels = 27
    input_dim = 3 * num_channels * 7 * 7
    classifier = ev.TeamTypeLSTMClassifier(input_dim=input_dim, embed_dim=256, hidden_dim=256,
                                            num_layers=1, dropout=0.1, bidirectional=False).to(device)
    classifier.load_state_dict(torch.load(ev.CLASSIFIER_PATH, map_location=device))
    classifier.eval()

    def make_adaptive():
        return ev.AdaptiveAgent(sp_agent=fresh(ev.SP_PATH), switch_agent=fresh(ev.ADVONLY_PATH),
                                 classifier=classifier, device=device)

    adaptive_main = make_adaptive()
    adaptive_clone_tm = make_adaptive()
    adaptive_prob_clone = make_adaptive()

    def wire_envs(env):
        adaptive_main.set_env(env)
        adaptive_clone_tm.set_env(env)
        adaptive_prob_clone.set_env(env)

    episode_plan = [True] * N  # force every episode adversarial

    print("Running forced-adversarial Adaptive evaluation with the REAL AdaptiveAgent code...")
    res = ev.evaluate(
        label='Adaptive_forced', main_agent=adaptive_main, clone_tm=adaptive_clone_tm,
        prob_adv=ev.ProbAdversaryTeammate(adaptive_prob_clone, adv_agent, ev.ADV_PROB),
        args=args, extra_env_setup=wire_envs, episode_plan=episode_plan,
    )
    print(f"\nForced-adversarial Adaptive mean reward (n={N}): {np.mean(res['all']):.2f}")
    print(f"individual rewards: {list(res['all'])}")


if __name__ == '__main__':
    main()
