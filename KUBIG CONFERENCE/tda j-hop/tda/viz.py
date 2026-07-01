"""Run 산출물 자동 시각화 — GTN 메타패스 + PDGNN EPD(persistence image).

학습엔 영향 없는 post-hoc 저장(train.py 에서 try/except 로 호출). matplotlib 은 함수
내부에서 lazy import(Agg backend) 하므로 이 모듈 import 만으론 matplotlib 의존이 없다.

저장물(output_dir):
  - metapath.png : gtn_attentions(채널×합성단계×관계 가중치) + fusion_beta(채널 중요도)
  - epd_pi.png   : 채널×스케일 평균 persistence image (PDGNN 이 근사한 EPD 결과)
  - topo_pi.npy  : (nch, N, res^2*K) 위상 특징 원본 (나중에 재시각화용)
"""
from __future__ import annotations

import os

import numpy as np


def plot_metapath(record: dict, rels, out_path: str) -> bool:
    att = record.get("gtn_attentions")
    if not att:
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    beta = np.array(record.get("fusion_beta", []), dtype=float)
    steps = []
    for li, layer in enumerate(att):
        a = np.array(layer)                       # (nconv, nch, R)
        for ci in range(a.shape[0]):
            steps.append((f"L{li}.{ci}", a[ci]))
    nch = steps[0][1].shape[0]
    R = steps[0][1].shape[1]
    labels = (list(rels) + [f"r{i}" for i in range(R)])[:R]
    ncol = nch + (1 if beta.size else 0)
    fig, axes = plt.subplots(1, ncol, figsize=(3.1 * ncol, 3.4), squeeze=False)
    axes = axes[0]
    for ch in range(nch):
        M = np.array([s[1][ch] for s in steps])    # (nsteps, R)
        ax = axes[ch]
        ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(R)); ax.set_xticklabels(labels, rotation=45, fontsize=7)
        ax.set_yticks(range(len(steps))); ax.set_yticklabels([s[0] for s in steps], fontsize=7)
        title = f"Channel {ch}" + (f" (beta={beta[ch]:.2f})" if ch < beta.size else "")
        ax.set_title(title, fontsize=9)
        for i in range(len(steps)):
            for j in range(R):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=5, color="white" if M[i, j] < 0.6 else "black")
    if beta.size:
        ax = axes[-1]
        ax.bar(range(beta.size), beta, color="steelblue")
        ax.set_xticks(range(beta.size)); ax.set_xticklabels([f"ch{c}" for c in range(beta.size)])
        ax.set_ylim(0, 1); ax.set_title("Fusion beta", fontsize=9)
    fig.suptitle(f"{record.get('dataset', '?')} — GTN meta-paths (relation weights/step) + fusion",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_epd_pi(topo_np: np.ndarray, out_path: str, K: int, res: int, dataset: str = "?") -> None:
    """topo_np: (nch, N, res^2*K). 노드 평균 persistence image 를 채널×스케일 격자로 표시."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nch = topo_np.shape[0]
    fig, axes = plt.subplots(nch, K, figsize=(2.2 * K, 2.2 * nch), squeeze=False)
    for ch in range(nch):
        mean = topo_np[ch].mean(axis=0)            # (res^2*K,)
        for k in range(K):
            img = mean[k * res * res:(k + 1) * res * res].reshape(res, res)
            ax = axes[ch][k]
            ax.imshow(img, cmap="magma", origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            if ch == 0:
                ax.set_title(f"scale {k}", fontsize=8)
            if k == 0:
                ax.set_ylabel(f"ch{ch}", fontsize=8)
    fig.suptitle(f"{dataset} — mean persistence image (PDGNN EPD) : channel × scale", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_run_figures(record: dict, topo_channels, output_dir: str, dataset: str, config: dict) -> None:
    """run() 에서 호출: 메타패스 그림 + EPD PI 그림 + 위상 원본(.npy) 저장."""
    os.makedirs(output_dir, exist_ok=True)
    rels = list(config.get("base_relations", {}).keys()) + ["I"]
    rec = dict(record); rec.setdefault("dataset", dataset)
    plot_metapath(rec, rels, os.path.join(output_dir, "metapath.png"))
    topo_np = np.stack([t.detach().cpu().numpy() for t in topo_channels], axis=0)  # (nch,N,dim)
    np.save(os.path.join(output_dir, "topo_pi.npy"), topo_np)
    pc = config.get("pdgnn", {})
    K, res = int(pc.get("hks_K", 3)), int(pc.get("pi_resolution", 5))
    plot_epd_pi(topo_np, os.path.join(output_dir, "epd_pi.png"), K, res, dataset)
