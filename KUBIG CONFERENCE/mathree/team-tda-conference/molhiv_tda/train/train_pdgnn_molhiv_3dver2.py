#!/usr/bin/env python3
"""
PDGNN-MolHIV + 3D 컨포머 기반 전기적 성질 feature 추가 버전
=================================================================
Colab 노트북(pdgnn_molhiv.py)의 OGBG-MolHIV 파트(셀 A~F)를 서버에서
바로 실행 가능한 단일 스크립트로 정리하고, 아래 feature를 추가했습니다.

추가된 것 (저비용 버전 — 비공유 가상 엣지는 아직 미포함):
  1. Node feature: RDKit Gasteiger partial charge (9d -> 10d)
     - 3D 컨포머 없이도 계산 가능 (2D 연결 정보만으로 근사)
  2. Edge feature: 기존 공유결합 엣지에 한해서
     - 3D 거리(Å)
     - 정전기 점수 (q_i * q_j / distance, "쿨롱 항")
     (3d -> 5d, 컨포머 임베딩이 필요해서 이 부분만 느림)

비공유(non-bonded) 가상 엣지, HAN 이종 엣지 타입 구분은 이번 버전엔
아직 넣지 않았습니다 (2단계 확장 예정). 저비용 feature만으로 먼저
ablation 하기 위한 버전입니다.

사용법
-----
  # 파이프라인 검증 (빠르게, 일부 분자만)
  python train_pdgnn_molhiv_3d.py --subset 2000 --epochs 30

  # baseline과 비교하고 싶으면 (3D feature 끄고 원본과 동일하게)
  python train_pdgnn_molhiv_3d.py --subset 2000 --epochs 30 --no-3d-features

  # 파이프라인 확인 끝나면 전체 데이터로
  python train_pdgnn_molhiv_3d.py --epochs 100
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 경로 설정 (train_pdgnn_tda.py와 동일한 관례) ──────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "train" \
    else Path(__file__).resolve().parent
DATA_ROOT   = PROJECT_ROOT / "dataset"          # PygGraphPropPredDataset root
CACHE_DIR   = PROJECT_ROOT / "cache" / "conformer_features"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# 1. RDKit 기반 3D feature 계산
# =====================================================================
def compute_mol_features(smiles: str, seed: int = 42, n_confs: int = 1):
    """
    한 분자에 대해:
      - Gasteiger partial charge (원자 개수만큼, 2D만으로 계산 가능)
      - 3D 컨포머 임베딩 후, 공유결합 쌍(i, j)의 실제 3D 거리
    를 계산해서 반환한다. 임베딩 실패 시 거리는 None으로 채워지고,
    charge는 그대로 반환된다 (charge 계산은 3D가 필요 없기 때문).

    n_confs=1  : 컨포머 1개만 생성 (기존과 동일, 빠름)
    n_confs>1 : 여러 개 생성 후 MMFF(실패시 UFF) 에너지가 가장 낮은
                컨포머를 선택 (더 정확하지만 n_confs배 정도 느려짐)
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, rdPartialCharges
    RDLogger.DisableLog("rdApp.*")  # RDKit의 시끄러운 경고 로그 끄기

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    n_atoms = mol.GetNumAtoms()

    # ---- (1) Gasteiger partial charge : 3D 불필요 ----
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = np.array(
            [float(a.GetProp("_GasteigerCharge")) if a.HasProp("_GasteigerCharge") else 0.0
             for a in mol.GetAtoms()],
            dtype=np.float32,
        )
        charges = np.nan_to_num(charges, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        charges = np.zeros(n_atoms, dtype=np.float32)

    # OGB와 동일한 순서로 bond 목록 확보 (양방향)
    bonded_pairs = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bonded_pairs.append((i, j))
        bonded_pairs.append((j, i))

    # ---- (2) 3D 컨포머 임베딩 (결합 거리 계산용) ----
    bonded_dist = {}
    has_3d = False
    n_confs = max(1, int(n_confs))

    try:
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed

        if n_confs == 1:
            # ---- 기존 방식: 컨포머 1개만 생성 ----
            ok = AllChem.EmbedMolecule(mol_h, params)
            cids = [0] if ok == 0 else []
        else:
            # ---- 멀티 컨포머: n_confs개 생성 ----
            cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=n_confs, params=params))

        if cids:
            best_cid = cids[0]
            if n_confs == 1:
                # 컨포머 1개면 에너지 비교 없이 그냥 최적화만
                try:
                    AllChem.MMFFOptimizeMolecule(mol_h, confId=best_cid)
                except Exception:
                    try:
                        AllChem.UFFOptimizeMolecule(mol_h, confId=best_cid)
                    except Exception:
                        pass
            else:
                # 여러 컨포머 각각 최적화 후, 에너지 가장 낮은 것 선택
                energies = []
                try:
                    mmff_props = AllChem.MMFFGetMoleculeProperties(mol_h)
                except Exception:
                    mmff_props = None

                for cid in cids:
                    try:
                        if mmff_props is not None:
                            ff = AllChem.MMFFGetMoleculeForceField(mol_h, mmff_props, confId=cid)
                        else:
                            ff = None
                        if ff is None:
                            ff = AllChem.UFFGetMoleculeForceField(mol_h, confId=cid)
                        if ff is None:
                            continue
                        ff.Minimize(maxIts=200)
                        energies.append((ff.CalcEnergy(), cid))
                    except Exception:
                        continue

                if energies:
                    best_cid = min(energies, key=lambda t: t[0])[1]
                # energies가 비어도(전부 실패) best_cid=cids[0] (최적화 안 된 초기 임베딩)로 그대로 진행

            mol_noh = Chem.RemoveHs(mol_h)
            if mol_noh.GetNumAtoms() == n_atoms:
                pos = mol_noh.GetConformer(best_cid).GetPositions()
                for (i, j) in bonded_pairs:
                    bonded_dist[(i, j)] = float(np.linalg.norm(pos[i] - pos[j]))
                has_3d = True
    except Exception:
        pass  # 임베딩 실패 -> bonded_dist 비어있는 채로 진행 (fallback)

    return {
        "n_atoms": n_atoms,
        "charges": charges,
        "bonded_dist": bonded_dist,  # {(i,j): distance} , 실패시 {}
        "has_3d": has_3d,
    }


# =====================================================================
# 2. PyG Data 객체에 feature 병합
# =====================================================================
def augment_data(data, feats):
    """
    data.x        : [N, 9]  -> [N, 10]  (partial charge 추가)
    data.edge_attr: [E, 3]  -> [E, 5]  (distance, coulomb score 추가)
    feats가 None이거나 3D 실패면, 새 차원은 0으로 채워서 shape만 맞춘다
    (그래야 배치 내 분자마다 차원이 안 어긋남).
    """
    n = data.x.size(0)
    x = data.x.float()

    if feats is not None and feats["n_atoms"] == n:
        charge = torch.tensor(feats["charges"], dtype=torch.float32).view(-1, 1)
    else:
        charge = torch.zeros(n, 1, dtype=torch.float32)
    x_new = torch.cat([x, charge], dim=1)  # [N, 10]

    ei = data.edge_index
    ea = data.edge_attr.float()
    E = ei.size(1)
    dist = torch.zeros(E, 1, dtype=torch.float32)
    coulomb = torch.zeros(E, 1, dtype=torch.float32)

    if feats is not None and feats["bonded_dist"]:
        src, dst = ei[0].tolist(), ei[1].tolist()
        bonded_dist = feats["bonded_dist"]
        charges_np = feats["charges"]
        for idx, (i, j) in enumerate(zip(src, dst)):
            d = bonded_dist.get((i, j))
            if d is not None:
                dist[idx, 0] = d
                coulomb[idx, 0] = float(charges_np[i] * charges_np[j] / max(d, 0.5))

    ea_new = torch.cat([ea, dist, coulomb], dim=1)  # [E, 5]

    new_data = data.clone()
    new_data.x = x_new
    new_data.edge_attr = ea_new
    return new_data


def strip_to_baseline(data):
    """--no-3d-features 옵션용: 원본과 동일한 차원(9d/3d)을 유지하되
    파이프라인(모델 dim 등)은 동일하게 맞추기 위해 0-padding만 추가."""
    n = data.x.size(0)
    x_new = torch.cat([data.x.float(), torch.zeros(n, 1)], dim=1)
    E = data.edge_index.size(1)
    ea_new = torch.cat([data.edge_attr.float(), torch.zeros(E, 2)], dim=1)
    new_data = data.clone()
    new_data.x = x_new
    new_data.edge_attr = ea_new
    return new_data


# =====================================================================
# 3. 데이터셋 로드 + 전처리(캐싱)
def stratified_subsample(indices, labels, target_size, rng):
    """
    주어진 indices(예: 원래 train 또는 valid 또는 test 그룹) 안에서,
    양성(1) / 음성(0) 비율을 그 그룹 원래 비율 그대로 유지하면서
    target_size개를 무작위로 뽑는다.

    예) 원래 이 그룹의 양성 비율이 3.5%라면, target_size가 몇이든
        뽑힌 결과도 항상 양성 약 3.5%를 유지한다 (반올림 오차 정도만 발생).
    """
    indices = np.asarray(indices)
    if target_size >= len(indices):
        return indices.tolist()

    idx_labels = labels[indices]
    pos_idx = indices[idx_labels == 1]
    neg_idx = indices[idx_labels == 0]

    pos_ratio = len(pos_idx) / len(indices) if len(indices) > 0 else 0.0
    n_pos = min(int(round(target_size * pos_ratio)), len(pos_idx))
    n_neg = min(target_size - n_pos, len(neg_idx))

    chosen_pos = rng.choice(pos_idx, size=n_pos, replace=False) if n_pos > 0 else np.array([], dtype=int)
    chosen_neg = rng.choice(neg_idx, size=n_neg, replace=False) if n_neg > 0 else np.array([], dtype=int)

    chosen = np.concatenate([chosen_pos, chosen_neg])
    rng.shuffle(chosen)
    return chosen.tolist()


# =====================================================================
def load_and_preprocess(args, device):
    from ogb.graphproppred import PygGraphPropPredDataset
    import pandas as pd

    print(f"[1/4] OGB 데이터셋 로드 중... (root={DATA_ROOT})")
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=str(DATA_ROOT))
    split_idx = dataset.get_idx_split()
    print(f"      전체 그래프 수: {len(dataset):,}")

    # ---- subset 선택 (파이프라인 빠른 검증용, 양성/음성 비율 층화추출) ----
    orig_train = split_idx["train"].tolist()
    orig_valid = split_idx["valid"].tolist()
    orig_test  = split_idx["test"].tolist()

    if args.subset is not None and args.subset < len(dataset):
        rng = np.random.default_rng(42)

        # 라벨 벡터 한 번만 읽어옴 (그래프 전체를 순회하지 않고 텐서에서 바로 추출)
        try:
            all_labels = dataset.data.y.view(-1).numpy()
        except AttributeError:
            # 일부 OGB 버전에서 .data.y 접근이 안 되는 경우 fallback (조금 느리지만 안전)
            all_labels = np.array([int(dataset[i].y[0]) for i in range(len(dataset))])

        total_orig = len(orig_train) + len(orig_valid) + len(orig_test)
        n_train = int(round(args.subset * len(orig_train) / total_orig))
        n_valid = int(round(args.subset * len(orig_valid) / total_orig))
        n_test  = args.subset - n_train - n_valid  # 나머지로 맞춰서 총합 정확히 유지

        train_idx = stratified_subsample(orig_train, all_labels, n_train, rng)
        valid_idx = stratified_subsample(orig_valid, all_labels, n_valid, rng)
        test_idx  = stratified_subsample(orig_test,  all_labels, n_test,  rng)

        def pos_ratio(idx_list):
            if not idx_list:
                return 0.0, 0
            n_pos = int(all_labels[idx_list].sum())
            return n_pos / len(idx_list) * 100, n_pos

        tr_r, tr_p = pos_ratio(train_idx)
        va_r, va_p = pos_ratio(valid_idx)
        te_r, te_p = pos_ratio(test_idx)

        print(f"      [subset 모드 - 층화추출] {args.subset}개로 축소")
        print(f"        train={len(train_idx)} (양성 {tr_p}개, {tr_r:.1f}%)")
        print(f"        valid={len(valid_idx)} (양성 {va_p}개, {va_r:.1f}%)")
        print(f"        test ={len(test_idx)} (양성 {te_p}개, {te_r:.1f}%)")
    else:
        train_idx = orig_train
        valid_idx = orig_valid
        test_idx  = orig_test

    used_idx = sorted(set(train_idx) | set(valid_idx) | set(test_idx))

    if args.no_3d_features:
        print("[2/4] --no-3d-features 지정됨 -> RDKit 계산 스킵 (baseline, 0-padding만)")
        processed = {i: strip_to_baseline(dataset[i]) for i in used_idx}
    else:
        print(f"[2/4] Partial charge + 3D 결합거리 계산 중... (대상 {len(used_idx)}개, n_confs={args.n_confs})")
        # SMILES 매핑 로드
        mapping_path = DATA_ROOT / "ogbg_molhiv" / "mapping" / "mol.csv.gz"
        smiles_df = pd.read_csv(mapping_path)

        # n_confs가 다르면 완전히 다른 3D 구조가 나오므로 캐시 파일을 분리
        cache_file = CACHE_DIR / f"gasteiger_dist_cache_conf{args.n_confs}.pt"

        # 캐시 로드 (이미 계산된 분자는 재계산 스킵)
        cache = {}
        if cache_file.exists():
            cache = torch.load(cache_file, weights_only=False)
            print(f"      캐시에서 {len(cache)}개 로드됨")

        new_count = 0
        t0 = time.time()
        for k, i in enumerate(used_idx):
            if i in cache:
                continue
            smiles = smiles_df.iloc[i]["smiles"]
            feats = compute_mol_features(smiles, n_confs=args.n_confs)
            cache[i] = feats
            new_count += 1
            if new_count % 200 == 0:
                elapsed = time.time() - t0
                print(f"      진행: {new_count}/{len(used_idx) - (len(used_idx) - new_count)} "
                      f"신규 계산, {elapsed:.0f}s 경과", end="\r")

        if new_count > 0:
            torch.save(cache, cache_file)
            print(f"\n      신규 계산 {new_count}개 완료, 캐시 저장: {cache_file}")

        fail_count = sum(1 for i in used_idx if cache[i] is None or not cache[i]["has_3d"])
        print(f"      3D 임베딩 실패/스킵: {fail_count}/{len(used_idx)} "
              f"(해당 분자는 거리=0, coulomb=0으로 처리됨. partial charge는 정상 반영)")

        print("[3/4] PyG Data 객체에 feature 병합 중...")
        processed = {i: augment_data(dataset[i], cache[i]) for i in used_idx}

    train_list = [processed[i] for i in train_idx]
    valid_list = [processed[i] for i in valid_idx]
    test_list  = [processed[i] for i in test_idx]

    print(f"[4/4] 준비 완료. node_dim={train_list[0].x.shape[1]}  "
          f"edge_dim={train_list[0].edge_attr.shape[1]}")
    return train_list, valid_list, test_list


# =====================================================================
# 4. 모델 (LearnableFilteration / MolGINE / PDGNNMolHIV)
#    -> node_dim, edge_dim만 새 차원(10, 5)에 맞게 조정. 구조는 원본과 동일.
# =====================================================================
class LearnableFilteration(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden=32):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(node_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x, edge_attr):
        v = torch.sigmoid(self.node_proj(x.float()).squeeze(-1))
        e = torch.sigmoid(self.edge_proj(edge_attr.float()).squeeze(-1))
        return v, e


def compute_pi_batch(v_vals, edge_index, batch, res=8, sigma=0.05):
    device = v_vals.device
    B = int(batch.max().item()) + 1
    lin = torch.linspace(0.0, 1.0, res, device=device)
    gx, gy = torch.meshgrid(lin, lin, indexing="ij")
    gx = gx.flatten(); gy = gy.flatten()
    src, dst = edge_index
    pi_list = []

    for g_id in range(B):
        node_mask = (batch == g_id)
        edge_mask = node_mask[src]
        if node_mask.sum() < 2 or edge_mask.sum() == 0:
            pi_list.append(torch.zeros(res * res, device=device))
            continue

        gv = v_vals[node_mask]
        base = node_mask.nonzero(as_tuple=True)[0].min()
        gs_loc = (src[edge_mask] - base).clamp(0, len(gv) - 1)
        gd_loc = (dst[edge_mask] - base).clamp(0, len(gv) - 1)

        birth = torch.min(gv[gs_loc], gv[gd_loc])
        death = torch.max(gv[gs_loc], gv[gd_loc])
        pers = (death - birth).clamp(min=1e-6)

        dx = birth.unsqueeze(1) - gx.unsqueeze(0)
        dy = death.unsqueeze(1) - gy.unsqueeze(0)
        gauss = torch.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))
        pi = (pers.unsqueeze(1) * gauss).sum(0)
        pi_list.append(pi / (pi.max() + 1e-8))

    return torch.stack(pi_list, dim=0)


class MolGINE(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden=256, num_layers=5):
        super().__init__()
        from torch_geometric.nn import GINEConv
        self.atom_emb = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.ReLU(),
                                 nn.Linear(hidden * 2, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
            self.bns.append(nn.BatchNorm1d(hidden))

    def forward(self, x, edge_index, edge_attr, batch):
        from torch_geometric.nn import global_mean_pool
        h = self.atom_emb(x.float())
        for conv, bn in zip(self.convs, self.bns):
            h = F.relu(bn(conv(h, edge_index, edge_attr.float())))
        return global_mean_pool(h, batch)


class PDGNNMolHIV(nn.Module):
    def __init__(self, node_dim, edge_dim, gin_hidden=256, pi_res=8, pi_sigma=0.05):
        super().__init__()
        self.filtration = LearnableFilteration(node_dim, edge_dim)
        self.gin = MolGINE(node_dim, edge_dim, gin_hidden)
        self.pi_res = pi_res
        self.pi_sigma = pi_sigma
        pi_dim = pi_res * pi_res
        self.classifier = nn.Sequential(
            nn.Linear(gin_hidden + pi_dim, 256),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        v_vals, _ = self.filtration(x, ea)
        pi = compute_pi_batch(v_vals, ei, batch, self.pi_res, self.pi_sigma)
        gin_out = self.gin(x, ei, ea, batch)
        return self.classifier(torch.cat([gin_out, pi], dim=-1))


# =====================================================================
# 5. 학습 루프 (원본과 동일)
# =====================================================================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n_seen = 0
    for data in loader:
        data = data.to(device)
        if data.x is None:
            continue
        optimizer.zero_grad()
        out = model(data)
        y = data.y.float().to(device)
        mask = ~torch.isnan(y.squeeze())
        loss = criterion(out.squeeze()[mask], y.squeeze()[mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * mask.sum().item()
        n_seen += mask.sum().item()
    return total_loss / max(n_seen, 1)


@torch.no_grad()
def evaluate(model, loader, evaluator, device):
    model.eval()
    y_true, y_pred = [], []
    for data in loader:
        data = data.to(device)
        if data.x is None:
            continue
        out = torch.sigmoid(model(data))
        y_true.append(data.y.cpu())
        y_pred.append(out.cpu())
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    mask = ~np.isnan(y_true.squeeze())
    return evaluator.eval({
        "y_true": y_true[mask].reshape(-1, 1),
        "y_pred": y_pred[mask].reshape(-1, 1),
    })["rocauc"]


def main():
    parser = argparse.ArgumentParser(description="PDGNN-MolHIV + 3D 컨포머 전기적 성질 feature")
    parser.add_argument("--subset", type=int, default=2000,
                         help="빠른 검증용 부분집합 크기 (기본 2000, 전체 쓰려면 --subset 0)")
    parser.add_argument("--no-3d-features", action="store_true",
                         help="RDKit 3D feature 계산 끄고 0-padding만 (baseline 비교용)")
    parser.add_argument("--n-confs", type=int, default=1,
                         help="분자당 생성할 3D 컨포머 개수. 1=기존과 동일(빠름), "
                              "2 이상이면 여러 개 생성 후 에너지 최저 컨포머 선택 (느리지만 더 정확)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tag", type=str, default="3d_feat",
                         help="결과 파일 이름에 붙는 태그 (baseline vs 3d 비교용)")
    args = parser.parse_args()
    if args.subset == 0:
        args.subset = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_list, valid_list, test_list = load_and_preprocess(args, device)

    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_list, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(valid_list, batch_size=256, shuffle=False)
    test_loader  = DataLoader(test_list,  batch_size=256, shuffle=False)

    node_dim = train_list[0].x.shape[1]
    edge_dim = train_list[0].edge_attr.shape[1]

    model = PDGNNMolHIV(node_dim=node_dim, edge_dim=edge_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"파라미터 수: {total_params:,}  (node_dim={node_dim}, edge_dim={edge_dim})")

    from ogb.graphproppred import Evaluator
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    evaluator = Evaluator(name="ogbg-molhiv")
    pos_weight = torch.tensor([27.0], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_val, best_test, patience_cnt = 0.0, 0.0, 0
    history = []
    model_path = RESULTS_DIR / f"pdgnn_molhiv_{args.tag}_best.pt"

    print(f"\n학습 시작  epochs={args.epochs}")
    print(f"  {'Epoch':>5}  {'Loss':>8}  {'Val AUC':>9}  {'Test AUC':>9}  {'Time':>6}")
    print("-" * 52)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_auc = evaluate(model, val_loader, evaluator, device)
        test_auc = evaluate(model, test_loader, evaluator, device)
        scheduler.step(val_auc)
        elapsed = time.time() - t0
        history.append((epoch, loss, val_auc, test_auc))

        marker = ""
        if val_auc > best_val:
            best_val, best_test = val_auc, test_auc
            patience_cnt = 0
            torch.save(model.state_dict(), model_path)
            marker = " *"
        else:
            patience_cnt += 1

        print(f"  {epoch:>5}  {loss:>8.4f}  {val_auc:>9.4f}  {test_auc:>9.4f}  {elapsed:>5.1f}s{marker}")

        if patience_cnt >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\n결과: Best Val AUC={best_val:.4f}  Test AUC={best_test:.4f}")

    result = {
        "tag": args.tag,
        "subset": args.subset,
        "no_3d_features": args.no_3d_features,
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "best_val_auc": best_val,
        "best_test_auc": best_test,
        "history": history,
    }
    out_path = RESULTS_DIR / f"molhiv_result_{args.tag}.json"
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"저장 완료: {out_path}")
    print(f"모델 저장: {model_path}")


if __name__ == "__main__":
    main()
