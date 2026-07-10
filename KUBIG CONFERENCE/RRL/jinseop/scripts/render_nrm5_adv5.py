"""
Renders exactly 5 normal-teammate and 5 adversarial-teammate episode GIFs for
each of the 4 conditions (SP, SPN_1ADV, Adaptive, PWADV_advonly), using a fixed
episode_plan (5x False + 5x True) so the NRM/ADV split is exact rather than
whatever a random draw happens to produce.
"""
import sys
from pathlib import Path

sys.path.insert(0, '/workspace')

import torch

import scripts.evaluate_sp_vs_spn1adv_vs_adaptive as ev

GIFS_ROOT = Path('/workspace/eval_results/gifs_nrm5_adv5')
EPISODE_PLAN = [False] * 5 + [True] * 5  # 5 NRM then 5 ADV

# Render every one of these 10 episodes (evaluate()'s render flag is `ep < N_RENDER`)
ev.N_EVAL = 0
ev.N_RENDER = 10


def main():
    sys.argv = ['render', '--layout-names', ev.LAYOUT, '--num-players', '3', '--teammates-len', '2',
                '--n-envs', '1', '--horizon', str(ev.HORIZON)]
    args = ev.get_arguments()
    args.base_dir = Path('/workspace')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading agents...")
    sp_agent = ev.load_best_agent(ev.SP_PATH, args)
    spn_agent = ev.load_best_agent(ev.SPN_PATH, args)
    adv_agent = ev.load_best_agent(ev.ADV_PATH, args)
    advonly_agent = ev.load_best_agent(ev.ADVONLY_PATH, args)

    def fresh(path):
        return ev.load_best_agent(path, args)

    num_channels = 27
    input_dim = 3 * num_channels * 7 * 7
    classifier = ev.TeamTypeLSTMClassifier(input_dim=input_dim, embed_dim=256, hidden_dim=256,
                                            num_layers=1, dropout=0.1, bidirectional=False).to(device)
    classifier.load_state_dict(torch.load(ev.CLASSIFIER_PATH, map_location=device))
    classifier.eval()
    print(f"Loaded classifier from {ev.CLASSIFIER_PATH} (switch_step={ev.SWITCH_STEP})\n")

    def make_adaptive():
        return ev.AdaptiveAgent(sp_agent=fresh(ev.SP_PATH), switch_agent=fresh(ev.ADVONLY_PATH),
                                 classifier=classifier, device=device)

    ev.GIFS_DIR = GIFS_ROOT

    # --- SP ---
    ev.evaluate(label='SP', main_agent=sp_agent, clone_tm=fresh(ev.SP_PATH),
                prob_adv=ev.ProbAdversaryTeammate(fresh(ev.SP_PATH), adv_agent, ev.ADV_PROB),
                args=args, episode_plan=EPISODE_PLAN)

    # --- SPN_1ADV ---
    ev.evaluate(label='SPN_1ADV', main_agent=spn_agent, clone_tm=fresh(ev.SPN_PATH),
                prob_adv=ev.ProbAdversaryTeammate(fresh(ev.SPN_PATH), adv_agent, ev.ADV_PROB),
                args=args, episode_plan=EPISODE_PLAN)

    # --- PWADV_advonly ---
    ev.evaluate(label='PWADV_advonly', main_agent=advonly_agent, clone_tm=fresh(ev.ADVONLY_PATH),
                prob_adv=ev.ProbAdversaryTeammate(fresh(ev.ADVONLY_PATH), adv_agent, ev.ADV_PROB),
                args=args, episode_plan=EPISODE_PLAN)

    # --- Adaptive ---
    adaptive_main = make_adaptive()
    adaptive_clone_tm = make_adaptive()
    adaptive_prob_clone = make_adaptive()

    def wire_envs(env):
        adaptive_main.set_env(env)
        adaptive_clone_tm.set_env(env)
        adaptive_prob_clone.set_env(env)

    ev.evaluate(label='Adaptive', main_agent=adaptive_main, clone_tm=adaptive_clone_tm,
                prob_adv=ev.ProbAdversaryTeammate(adaptive_prob_clone, adv_agent, ev.ADV_PROB),
                args=args, extra_env_setup=wire_envs, episode_plan=EPISODE_PLAN)

    print(f"\nAll GIFs saved under {GIFS_ROOT}/<condition>/ (5 NRM + 5 ADV each)")


if __name__ == '__main__':
    main()
