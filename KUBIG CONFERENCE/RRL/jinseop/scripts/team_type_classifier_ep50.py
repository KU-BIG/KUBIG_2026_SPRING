"""
Predicts whether an Overcooked episode is an SP team (3 copies of the same
primary agent) or an SPADV team (2 copies of the primary agent + 1 adversary),
from the first EPISODE_LEN steps of all 3 players' observations, using an LSTM.

Variant of team_type_classifier.py using only the first 50 steps (instead of
100) and a stricter 99% validation-accuracy target.

Both agents act according to fixed, pretrained (frozen) policies:
  - primary: SP_s1010_h256_tr[SP]_ran (best)
  - adversary: ADV_s68_h512_tr[H]_ran_selfisher_attack2 (best)

Runs dataset generation once (growable across attempts), then trains the
classifier in a loop: if validation accuracy doesn't reach TARGET_VAL_ACC, the
current model + a diagnostic report are saved, the config is adjusted based on
the diagnosis, and training is retried, up to MAX_ATTEMPTS times.
"""
import os
import sys
import json
import time
import random
import argparse
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '/workspace')

from oai_agents.common.arguments import get_arguments
from oai_agents.agents.rl import RLAgentTrainer  # noqa: F401 (resolves circular import ordering)
from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv
from oai_agents.common.batched_vecenv import BatchedTeammatesDummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from scripts.utils.common import load_agents

LAYOUT = '3_chefs_counter_circuit'
EXP_DIR = 'Classic/3'
SP_AGENT_NAME = 'SP_s1010_h256_tr[SP]_ran'
ADV_AGENT_NAME = 'ADV_s68_h512_tr[H]_ran_selfisher_attack2'
EPISODE_LEN = 50
TARGET_VAL_ACC = 0.99
MAX_ATTEMPTS = 5
RUN_ROOT = Path('/workspace/team_type_classifier_runs_ep50')
DATA_DIR = RUN_ROOT / 'dataset'
X_PATH = DATA_DIR / 'X.npy'
Y_PATH = DATA_DIR / 'y.npy'
SCEN_PATH = DATA_DIR / 'scenario.npy'

# scenario_id -> (team_type label, adv position or None)
SCENARIOS = [
    ('SP', None, 0.5),
    ('SPADV', 0, 1 / 6),
    ('SPADV', 1, 1 / 6),
    ('SPADV', 2, 1 / 6),
]
SCENARIO_NAMES = ['SP', 'SPADV_adv@0', 'SPADV_adv@1', 'SPADV_adv@2']


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# --------------------------------------------------------------------------- #
# Environment / agents setup
# --------------------------------------------------------------------------- #
def build_args(n_envs):
    sys.argv = [
        'team_type_classifier_ep50',
        '--layout-names', LAYOUT,
        '--num-players', '3',
        '--teammates-len', '2',
        '--n-envs', str(n_envs),
        '--horizon', '400',
        '--exp-dir', EXP_DIR,
        '--epoch-timesteps', '50000',
    ]
    args = get_arguments()
    args.base_dir = Path('/workspace')
    return args


def load_fixed_agents(args):
    sp_agents = load_agents(args, name=SP_AGENT_NAME, tag='best')
    adv_agents = load_agents(args, name=ADV_AGENT_NAME, tag='best')
    assert sp_agents, f'Could not load {SP_AGENT_NAME} (best)'
    assert adv_agents, f'Could not load {ADV_AGENT_NAME} (best)'
    return sp_agents[0], adv_agents[0]


def make_env_batch(args, n_envs):
    env_kwargs = {
        'shape_rewards': False, 'full_init': False, 'stack_frames': False,
        'deterministic': False, 'args': args, 'learner_type': 'originaler',
        'is_eval_env': True,
    }
    vec_env = make_vec_env(OvercookedGymEnv, n_envs=n_envs, vec_env_cls=BatchedTeammatesDummyVecEnv,
                            env_kwargs=env_kwargs)
    for i in range(n_envs):
        vec_env.env_method('set_env_layout', indices=i, env_index=0, layout_name=LAYOUT, unique_env_idx=i)
    return vec_env


def sample_scenario(rng):
    r = rng.random()
    cum = 0.0
    for team_type, adv_pos, p in SCENARIOS:
        cum += p
        if r < cum:
            return team_type, adv_pos
    team_type, adv_pos, _ = SCENARIOS[-1]
    return team_type, adv_pos


def scenario_id(team_type, adv_pos):
    if team_type == 'SP':
        return 0
    return 1 + adv_pos


def run_round(vec_env, sp_agent, adv_agent, team_type, adv_pos, episode_len, n_envs, num_channels):
    position_agents = [sp_agent, sp_agent, sp_agent]
    if adv_pos is not None:
        position_agents[adv_pos] = adv_agent
    ego_agent = position_agents[0]
    teammates = [position_agents[1], position_agents[2]]

    vec_env.env_method('set_reset_p_idx', 0)
    vec_env.env_method('set_teammates', teammates)
    obs = vec_env.reset()

    seq = np.zeros((n_envs, episode_len, 3, num_channels, 7, 7), dtype=np.uint8)
    for t in range(episode_len):
        obs1_list = vec_env.env_method('get_obs', 1)
        obs2_list = vec_env.env_method('get_obs', 2)
        seq[:, t, 0] = obs['visual_obs']
        seq[:, t, 1] = np.stack([d['visual_obs'] for d in obs1_list])
        seq[:, t, 2] = np.stack([d['visual_obs'] for d in obs2_list])

        action, _ = ego_agent.predict(obs, deterministic=False)
        obs, rewards, dones, infos = vec_env.step(action)

    label = 1 if team_type == 'SPADV' else 0
    labels = np.full((n_envs,), label, dtype=np.int64)
    scen = np.full((n_envs,), scenario_id(team_type, adv_pos), dtype=np.int64)
    return seq, labels, scen


def ensure_dataset(target_n, n_envs=128, episode_len=EPISODE_LEN, seed=0):
    """Grows the saved dataset on disk to at least target_n episodes; returns (X, y, scenario) arrays (memmap)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing_n = 0
    if X_PATH.exists() and Y_PATH.exists() and SCEN_PATH.exists():
        existing_n = np.load(Y_PATH).shape[0]

    if existing_n >= target_n:
        log(f'Reusing existing dataset with {existing_n} episodes (>= target {target_n}).')
        return np.load(X_PATH, mmap_mode='r'), np.load(Y_PATH), np.load(SCEN_PATH)

    n_needed = target_n - existing_n
    log(f'Generating {n_needed} additional episodes (existing={existing_n}, target={target_n})...')

    args = build_args(n_envs)
    sp_agent, adv_agent = load_fixed_agents(args)
    vec_env = make_env_batch(args, n_envs)
    num_channels = vec_env.get_attr('num_enc_channels')[0]

    rng = random.Random(seed + existing_n)
    n_rounds = (n_needed + n_envs - 1) // n_envs
    new_seq, new_labels, new_scen = [], [], []
    t0 = time.time()
    for r in range(n_rounds):
        team_type, adv_pos = sample_scenario(rng)
        seq, labels, scen = run_round(vec_env, sp_agent, adv_agent, team_type, adv_pos, episode_len, n_envs,
                                       num_channels)
        new_seq.append(seq)
        new_labels.append(labels)
        new_scen.append(scen)
        elapsed = time.time() - t0
        log(f'  round {r + 1}/{n_rounds} scenario={team_type}'
            f'{("@" + str(adv_pos)) if adv_pos is not None else ""} '
            f'({(r + 1) * n_envs} eps, {elapsed:.1f}s elapsed)')
    vec_env.close()

    X_new = np.concatenate(new_seq, axis=0)
    y_new = np.concatenate(new_labels, axis=0)
    s_new = np.concatenate(new_scen, axis=0)

    if existing_n > 0:
        X_old = np.load(X_PATH)
        y_old = np.load(Y_PATH)
        s_old = np.load(SCEN_PATH)
        X = np.concatenate([X_old, X_new], axis=0)
        y = np.concatenate([y_old, y_new], axis=0)
        s = np.concatenate([s_old, s_new], axis=0)
    else:
        X, y, s = X_new, y_new, s_new

    np.save(X_PATH, X)
    np.save(Y_PATH, y)
    np.save(SCEN_PATH, s)
    log(f'Dataset saved: {X.shape[0]} episodes, X shape={X.shape}, y balance={y.mean():.3f}')
    return np.load(X_PATH, mmap_mode='r'), np.load(Y_PATH), np.load(SCEN_PATH)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class TeamTypeLSTMClassifier(nn.Module):
    def __init__(self, input_dim, embed_dim=256, hidden_dim=256, num_layers=1, dropout=0.0, bidirectional=False):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Linear(out_dim, out_dim // 2), nn.ReLU(), nn.Dropout(dropout),
                                   nn.Linear(out_dim // 2, 1))

    def forward(self, x):  # x: (B, T, input_dim) float
        b, t, d = x.shape
        x = self.embed(x.reshape(b * t, d)).reshape(b, t, -1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)  # logits (B,)


class TeamTypeDataset(Dataset):
    def __init__(self, X, y, indices):
        self.X = X
        self.y = y
        self.indices = indices
        self.T = X.shape[1]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        x = np.asarray(self.X[idx]).reshape(self.T, -1).astype(np.float32)
        y = np.float32(self.y[idx])
        return x, y


# --------------------------------------------------------------------------- #
# Training + diagnostics
# --------------------------------------------------------------------------- #
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x).cpu()
            all_logits.append(logits)
            all_labels.append(y)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds = (torch.sigmoid(logits) > 0.5).float()
    acc = (preds == labels).float().mean().item()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    return {'acc': acc, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'n': len(labels)}


def per_scenario_accuracy(model, X, y, scen, val_idx, device, batch_size=64):
    model.eval()
    results = {}
    for sid, name in enumerate(SCENARIO_NAMES):
        idx = val_idx[scen[val_idx] == sid]
        if len(idx) == 0:
            continue
        ds = TeamTypeDataset(X, y, idx)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        res = evaluate(model, loader, device)
        results[name] = res['acc']
    return results


def train_one_attempt(attempt, config, X, y, scen, device):
    n = len(y)
    rng = np.random.RandomState(1000 + attempt)
    perm = rng.permutation(n)
    n_val = max(1, int(n * 0.2))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_ds = TeamTypeDataset(X, y, train_idx)
    val_ds = TeamTypeDataset(X, y, val_idx)
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)

    input_dim = X.shape[2] * X.shape[3] * X.shape[4] * X.shape[5]
    model = TeamTypeLSTMClassifier(
        input_dim=input_dim, embed_dim=config['embed_dim'], hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'], dropout=config['dropout'], bidirectional=config['bidirectional'],
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = -1.0
    best_state = None
    patience_left = config['early_stop_patience']
    history = []

    for epoch in range(config['epochs']):
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0
        for x, y_batch in train_loader:
            x, y_batch = x.to(device), y_batch.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(y_batch)
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == y_batch).sum().item()
            total_n += len(y_batch)

        train_acc = total_correct / total_n
        train_loss = total_loss / total_n
        val_res = evaluate(model, val_loader, device)
        scheduler.step(val_res['acc'])
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'val_acc': val_res['acc']})
        log(f'  [attempt {attempt}] epoch {epoch + 1}/{config["epochs"]} '
            f'train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_res["acc"]:.4f}')

        if val_res['acc'] > best_val_acc:
            best_val_acc = val_res['acc']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = config['early_stop_patience']
        else:
            patience_left -= 1
            if patience_left <= 0:
                log(f'  [attempt {attempt}] early stopping at epoch {epoch + 1} (best val_acc={best_val_acc:.4f})')
                break

    model.load_state_dict(best_state)
    final_train_res = evaluate(model, DataLoader(train_ds, batch_size=config['batch_size'], shuffle=False), device)
    final_val_res = evaluate(model, val_loader, device)
    scenario_acc = per_scenario_accuracy(model, X, y, scen, val_idx, device, config['batch_size'])

    diagnostics = {
        'attempt': attempt,
        'config': config,
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'train_acc': final_train_res['acc'],
        'val_acc': final_val_res['acc'],
        'val_confusion': {k: final_val_res[k] for k in ('tp', 'tn', 'fp', 'fn')},
        'val_scenario_acc': scenario_acc,
        'history': history,
    }
    return model, diagnostics


def diagnose_and_adjust(diagnostics, config, dataset_n):
    """Looks at what went wrong and returns (new_config, new_target_dataset_n, reasoning_text)."""
    train_acc = diagnostics['train_acc']
    val_acc = diagnostics['val_acc']
    gap = train_acc - val_acc
    scenario_acc = diagnostics['val_scenario_acc']
    weakest = min(scenario_acc.items(), key=lambda kv: kv[1]) if scenario_acc else (None, None)

    reasons = []
    new_config = dict(config)
    new_n = dataset_n

    reasons.append(f'train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, gap={gap:.4f}')
    if weakest[0] is not None:
        reasons.append(f'weakest scenario: {weakest[0]} (acc={weakest[1]:.4f})')

    if gap > 0.04:
        reasons.append('Large train/val gap => overfitting. Increasing regularization (dropout, weight_decay) '
                        'and growing the dataset so the model sees more variety.')
        new_config['dropout'] = min(0.5, config['dropout'] + 0.15)
        new_config['weight_decay'] = max(config['weight_decay'], 1e-4) * 3
        new_n = int(dataset_n * 1.5)
    elif train_acc < 0.95:
        reasons.append('Train accuracy itself is low => underfitting (harder task with only 50 steps of '
                        'evidence). Increasing model capacity (hidden/embed dim, bidirectional) and training '
                        'longer.')
        new_config['hidden_dim'] = min(512, config['hidden_dim'] * 2)
        new_config['embed_dim'] = min(512, config['embed_dim'] * 2)
        new_config['bidirectional'] = True
        new_config['epochs'] = config['epochs'] + 20
    else:
        reasons.append('Close to target but not quite there => growing the dataset and training a bit longer '
                        'with a slightly larger model, focused on the weakest scenario.')
        new_n = int(dataset_n * 1.5)
        new_config['epochs'] = config['epochs'] + 15
        new_config['hidden_dim'] = min(512, int(config['hidden_dim'] * 1.5))

    reasoning = ' '.join(reasons)
    return new_config, new_n, reasoning


def save_attempt_artifacts(attempt, model, diagnostics, reasoning=None):
    out_dir = RUN_ROOT / f'attempt_{attempt}'
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / 'model.pt')
    with open(out_dir / 'diagnostics.json', 'w') as f:
        json.dump(diagnostics, f, indent=2, default=float)
    report_lines = [
        f'# Attempt {attempt} report (EPISODE_LEN={EPISODE_LEN}, TARGET={TARGET_VAL_ACC})',
        '',
        f'- train_acc: {diagnostics["train_acc"]:.4f}',
        f'- val_acc:   {diagnostics["val_acc"]:.4f}',
        f'- n_train:   {diagnostics["n_train"]}',
        f'- n_val:     {diagnostics["n_val"]}',
        f'- val confusion (tp/tn/fp/fn): {diagnostics["val_confusion"]}',
        f'- per-scenario val accuracy: {diagnostics["val_scenario_acc"]}',
        f'- config: {diagnostics["config"]}',
    ]
    if reasoning:
        report_lines += ['', '## Diagnosis / what was changed for the next attempt', reasoning]
    with open(out_dir / 'report.md', 'w') as f:
        f.write('\n'.join(report_lines))
    log(f'Saved attempt {attempt} artifacts to {out_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial-n-episodes', type=int, default=2000)
    parser.add_argument('--n-envs', type=int, default=128)
    parser.add_argument('--gen-seed', type=int, default=0)
    args_cli = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'Using device: {device}')
    log(f'EPISODE_LEN={EPISODE_LEN}, TARGET_VAL_ACC={TARGET_VAL_ACC}')

    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    config = dict(
        batch_size=64, embed_dim=256, hidden_dim=256, num_layers=1, dropout=0.1,
        bidirectional=False, lr=1e-3, weight_decay=1e-5, epochs=30, early_stop_patience=6,
    )
    dataset_n = args_cli.initial_n_episodes

    summary = {'attempts': [], 'success': False}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        log(f'=== Attempt {attempt}/{MAX_ATTEMPTS} | dataset_n={dataset_n} | config={config} ===')
        X, y, scen = ensure_dataset(dataset_n, n_envs=args_cli.n_envs, seed=args_cli.gen_seed)
        model, diagnostics = train_one_attempt(attempt, config, X, y, scen, device)
        val_acc = diagnostics['val_acc']
        success = val_acc >= TARGET_VAL_ACC

        reasoning = None
        if not success and attempt < MAX_ATTEMPTS:
            config, dataset_n, reasoning = diagnose_and_adjust(diagnostics, config, dataset_n)
            log(f'Diagnosis: {reasoning}')

        save_attempt_artifacts(attempt, model, diagnostics, reasoning)
        summary['attempts'].append({'attempt': attempt, 'val_acc': val_acc, 'train_acc': diagnostics['train_acc']})

        if success:
            summary['success'] = True
            summary['final_attempt'] = attempt
            log(f'TARGET REACHED at attempt {attempt}: val_acc={val_acc:.4f} >= {TARGET_VAL_ACC}')
            break
    else:
        summary['final_attempt'] = MAX_ATTEMPTS
        log(f'Did NOT reach target accuracy after {MAX_ATTEMPTS} attempts. '
            f'Best results saved under {RUN_ROOT}.')

    with open(RUN_ROOT / 'SUMMARY.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    log(f'Done. Summary: {summary}')


if __name__ == '__main__':
    main()
