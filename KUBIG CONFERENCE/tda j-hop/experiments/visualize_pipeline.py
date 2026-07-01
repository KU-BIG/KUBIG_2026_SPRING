"""파이프라인 과정 시각화 (acm 예시):
  Fig1 metapath_construct.png : base relations + GTN 가중치 -> 채널 인접행렬(그룹 구조)
  Fig2 epd_process.png        : 채널 그래프 -> 한 노드 ego + HKS 필터 -> persistence diagram -> PI
저장된 gtn_attentions 로 채널 인접행렬 H 를 *복원*(재학습 X). CPU 전용."""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "/mnt/data/users/junyoungpark/code/TDA")
from tda.data.registry import get_dataset  # noqa: E402
from tda.models.gtn import _gtn_norm  # noqa: E402
from tda.topology.epd import _edge_index_of, _ego_filt_edges, _exact_epd, knn_graph_from_dense  # noqa: E402
from tda.topology.hks import compute_hks  # noqa: E402
from tda.topology.persistence_image import PersistenceImage  # noqa: E402
from tda.train import _build_A_stack  # noqa: E402
from tda.utils import load_json, set_seed  # noqa: E402

DATA_ROOT = "/mnt/data/users/junyoungpark/code/hetero_pdg_lp/data"
DS = sys.argv[1] if len(sys.argv) > 1 else "acm"   # 데이터셋 인자 (기본 acm)
OUT = os.environ.get("VIZ_OUT", f"results/figures/{DS}")
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cpu")

mfile = sorted(glob.glob(f"runs/campaign/{DS}__c2_gtn_s*/metrics.json"))[0]
M = json.load(open(mfile))
seed = M["seed"]
print(f"[viz] {DS} seed={seed}  (gtn_attentions from {mfile})", flush=True)

cfg = load_json(f"configs/{DS}.json"); cfg["seed"] = seed
set_seed(seed)
bundle = get_dataset(DS, cfg, DATA_ROOT)
y = bundle.y.numpy()
rels = list(bundle.base_relations.keys())
A_stack = _build_A_stack(bundle, dev)            # (R, N, N), R = rels + identity
R = A_stack.size(0)
rel_labels = (rels + ["I"])[:R]

# ---- 채널 인접행렬 H 복원 (discover() 재현, 저장된 softmax 가중치 사용) ----
att = M["gtn_attentions"]                         # [[w1,w2],[w3]]  (각 (C,R))
def conv(w):                                       # Σ_r w_cr A_r
    return torch.einsum("cr,rij->cij", torch.tensor(w, dtype=A_stack.dtype), A_stack)
H = torch.bmm(conv(att[0][0]), conv(att[0][1]))   # layer0
for li in range(1, len(att)):
    Hn = torch.stack([_gtn_norm(H[c], add=False) for c in range(H.size(0))], 0)
    H = torch.bmm(Hn, conv(att[li][0]))           # layer li
C = H.size(0)
beta = np.array(M.get("fusion_beta", [1.0] * C))
ch = int(np.argmax(beta))                         # 가장 중요한(fusion β 최대) 채널
print(f"[viz] channels={C}, fusion_beta={beta.round(3).tolist()}, featured channel={ch}", flush=True)

# 대표 메타패스(각 합성 단계 argmax 관계)
dom = []
for layer in att:
    for w in layer:
        dom.append(rel_labels[int(np.argmax(np.array(w)[ch]))])
metapath_str = " ∘ ".join(dom)

# ================= Fig 1: 메타패스 구성 =================
g_ch = knn_graph_from_dense(H[ch], cfg["pdgnn"]["knn_k"])
# 연결 부분그래프 ~70노드 샘플
import collections
start = max(g_ch.nodes, key=lambda n: g_ch.degree(n))
seen, q = {start}, collections.deque([start])
while q and len(seen) < 70:
    u = q.popleft()
    for v in g_ch.neighbors(u):
        if v not in seen and len(seen) < 70:
            seen.add(v); q.append(v)
sub = g_ch.subgraph(seen)

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
# (좌) 채널의 관계 가중치 (합성 단계별)
W = np.array([np.array(w)[ch] for layer in att for w in layer])   # (steps, R)
im = ax[0].imshow(W, cmap="viridis", vmin=0, vmax=1, aspect="auto")
ax[0].set_xticks(range(R)); ax[0].set_xticklabels(rel_labels, rotation=45, fontsize=9)
ax[0].set_yticks(range(W.shape[0])); ax[0].set_yticklabels([f"step {i}" for i in range(W.shape[0])])
ax[0].set_title(f"(1) GTN relation weights — channel {ch}\ndominant meta-path: {metapath_str}", fontsize=10)
for i in range(W.shape[0]):
    for j in range(R):
        ax[0].text(j, i, f"{W[i,j]:.2f}", ha="center", va="center", fontsize=7,
                   color="white" if W[i, j] < 0.6 else "black")
plt.colorbar(im, ax=ax[0], fraction=0.046)
# (우) 복원된 채널 인접행렬을 그래프로 (클래스별 색) — 메타패스가 묶는 구조
pos = nx.spring_layout(sub, seed=0, k=0.5)
cols = [int(y[n]) for n in sub.nodes]
nx.draw_networkx_edges(sub, pos, ax=ax[1], alpha=0.25, width=0.6)
sc = nx.draw_networkx_nodes(sub, pos, ax=ax[1], node_size=70, node_color=cols, cmap="tab10")
ax[1].set_title(f"(2) channel-{ch} adjacency (meta-path graph), ~{len(seen)} nodes\n"
                f"node color = class label → same-class grouping", fontsize=10)
ax[1].axis("off")
fig.suptitle(f"{DS}: full graph -> meta-path (channel) construction  [seed {seed}]", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUT}/metapath_construct.png", dpi=130, bbox_inches="tight"); plt.close(fig)
print(f"[viz] saved metapath_construct.png", flush=True)

# ================= Fig 2: EPD 생성 과정 =================
ei = _edge_index_of(g_ch)
hks = compute_hks(ei, num_nodes=H.size(1), K=cfg["pdgnn"]["hks_K"], device=dev)  # (N, K)
import gudhi
K = cfg["pdgnn"]["hks_K"]
filts_by_k = [{nd: float(hks[nd, kk]) for nd in g_ch.nodes()} for kk in range(K)]


def epd_full(filt, ei):
    """(points(P,2) birth-death(inf→max_filt), 유한·비capping 점수, 루프수)."""
    st = gudhi.SimplexTree()
    for i, f in enumerate(filt):
        st.insert([int(i)], filtration=float(f))
    seen = set()
    for a, b in zip(ei[0], ei[1]):
        e = (int(min(a, b)), int(max(a, b)))
        if e in seen:
            continue
        seen.add(e); st.insert([e[0], e[1]], filtration=float(max(filt[e[0]], filt[e[1]])))
    mx = float(filt.max()) if filt.size else 1.0
    pts, ncap, loops = [], 0, 0
    for dim, (b, d) in st.persistence(homology_coeff_field=2, min_persistence=0.0):
        if dim == 1:
            loops += 1
        if d == float("inf"):
            d = mx
        elif d - b > 1e-6:
            ncap += 1
        if d > b:
            pts.append((b, d))
    return np.array(pts, dtype=np.float64), ncap, loops


# ===== 랜덤 100개 노드 각각의 EPD 그림을 *개별 파일*로 저장 (ego + PD + PI) =====
res = cfg["pdgnn"]["pi_resolution"]
epddir = os.path.join(OUT, "epd")
os.makedirs(epddir, exist_ok=True)


def save_epd_figure(vv, kk, out_path):
    """노드 vv 의 ego + persistence diagram + persistence image 를 한 파일로 저장."""
    r = _ego_filt_edges(g_ch, int(vv), cfg["pdgnn"]["hop"], filts_by_k[kk], cfg["pdgnn"]["max_nodes"])
    if r is None or r[1].shape[1] == 0:
        return False
    filt, eei, nodes = r
    gt, _, _ = epd_full(filt, eei)
    if not len(gt):
        return False
    births, deaths = gt[:, 0], gt[:, 1]
    persg = deaths - births
    mx = float(filt.max())
    capped = deaths >= mx - 1e-6
    img_t = PersistenceImage(resolution=res, sigma=max(persg.max() / 6, 0.05),
                             birth_range=(float(births.min()), float(births.max())),
                             pers_range=(0.0, float(max(persg.max(), 1e-3))))
    pi = img_t.transform(gt).reshape(res, res)
    ego = nx.Graph(); ego.add_nodes_from(range(len(nodes)))
    for a_, b_ in zip(eei[0], eei[1]):
        ego.add_edge(int(a_), int(b_))
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    posE = nx.spring_layout(ego, seed=0)
    nx.draw_networkx_edges(ego, posE, ax=ax[0], alpha=0.35, width=0.6)
    sc = nx.draw_networkx_nodes(ego, posE, ax=ax[0], node_size=90, node_color=filt, cmap="coolwarm")
    plt.colorbar(sc, ax=ax[0], fraction=0.046, label="HKS filter f(v)")
    ax[0].set_title(f"(1) node {int(vv)} ego ({len(nodes)} nodes, {eei.shape[1]//2} edges)\ncolor = HKS value",
                    fontsize=10); ax[0].axis("off")
    ax[1].scatter(births[~capped], deaths[~capped], s=45, c="crimson", zorder=3, label=f"finite ({int((~capped).sum())})")
    ax[1].scatter(births[capped], deaths[capped], s=25, c="gray", alpha=0.6, zorder=2, label=f"essential ({int(capped.sum())})")
    lim = [float(births.min()) - 0.2, float(deaths.max()) + 0.2]
    ax[1].plot(lim, lim, "k--", alpha=0.5); ax[1].legend(fontsize=8)
    ax[1].set_xlabel("birth"); ax[1].set_ylabel("death")
    ax[1].set_title(f"(2) persistence diagram ({len(gt)} pts)\nmax persistence = {persg.max():.2f}", fontsize=10)
    im = ax[2].imshow(pi, cmap="magma", origin="lower")
    ax[2].set_title(f"(3) persistence image (res {res})", fontsize=10)
    ax[2].set_xticks([]); ax[2].set_yticks([]); plt.colorbar(im, ax=ax[2], fraction=0.046)
    fig.suptitle(f"{DS} ch{ch} node {int(vv)}: ego + HKS -> EPD -> persistence image", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return True


_, _cnt = np.unique(y[y >= 0], return_counts=True)
RAND_HOMO = float(((_cnt / _cnt.sum()) ** 2).sum())   # 무작위 same-class 기대치


def save_metapath_figure(vv, out_path):
    """노드 vv 의 메타패스 이웃을 metapath_result 스타일 3패널로:
    (A) node-link(클래스색) (B) 클래스정렬 인접(블록) (C) same-class edge 비율 vs 무작위."""
    egon = list(nx.ego_graph(g_ch, int(vv), radius=cfg["pdgnn"]["hop"]).nodes())
    if len(egon) < 3:
        return False
    sg = g_ch.subgraph(egon)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    # (A) node-link colored by class
    pos = nx.spring_layout(sg, seed=0)
    nx.draw_networkx_edges(sg, pos, ax=ax[0], alpha=0.3, width=0.6)
    nc = nx.draw_networkx_nodes(sg, pos, ax=ax[0], node_size=80,
                                node_color=[int(y[n]) for n in sg.nodes()], cmap="tab10",
                                vmin=0, vmax=bundle.num_classes - 1)
    nx.draw_networkx_nodes(sg, pos, ax=ax[0], nodelist=[int(vv)], node_size=230,
                           node_color="none", edgecolors="black", linewidths=2)
    ax[0].set_title(f"(A) node {int(vv)} meta-path neighborhood ({len(egon)} nodes)\nnode color = class label",
                    fontsize=10); ax[0].axis("off")
    plt.colorbar(nc, ax=ax[0], fraction=0.046, ticks=range(bundle.num_classes), label="class")
    # (B) adjacency sorted by class -> block-diagonal = same-class linked
    order = sorted(sg.nodes(), key=lambda n: int(y[n]))
    A_ = nx.to_numpy_array(sg, nodelist=order)
    ax[1].imshow(A_, cmap="Greys", interpolation="nearest")
    ys = np.array([int(y[n]) for n in order])
    for b in (np.where(np.diff(ys) != 0)[0] + 0.5):
        ax[1].axhline(b, color="red", lw=0.6, alpha=0.7); ax[1].axvline(b, color="red", lw=0.6, alpha=0.7)
    ax[1].set_title(f"(B) adjacency sorted by class ({len(order)} nodes)\nblock-diagonal = same-class linked",
                    fontsize=10); ax[1].set_xticks([]); ax[1].set_yticks([])
    # (C) homophily of this neighborhood vs random baseline
    same = float(np.mean([y[u] == y[w] for u, w in sg.edges()])) if sg.number_of_edges() else 0.0
    ax[2].bar(["meta-path", "random"], [same, RAND_HOMO], color=["crimson", "gray"])
    for i, val in enumerate([same, RAND_HOMO]):
        ax[2].text(i, val + 0.01, f"{val:.2f}", ha="center", fontsize=11)
    ax[2].set_ylim(0, 1); ax[2].set_ylabel("same-class edge fraction")
    ax[2].set_title(f"(C) homophily {same:.2f} vs {RAND_HOMO:.2f} random", fontsize=10)
    fig.suptitle(f"{DS} ch{ch} node {int(vv)}: meta-path RESULT (neighborhood grouping)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return True


mpdir = os.path.join(OUT, "metapath")
os.makedirs(mpdir, exist_ok=True)
rng2 = np.random.RandomState(1)
epd_made = mp_made = 0
for vv in rng2.permutation(H.size(1)).tolist():
    if epd_made >= 100 and mp_made >= 100:
        break
    if epd_made < 100 and save_epd_figure(vv, 0, os.path.join(epddir, f"epd_node{int(vv)}.png")):
        epd_made += 1
    if mp_made < 100 and save_metapath_figure(vv, os.path.join(mpdir, f"metapath_node{int(vv)}.png")):
        mp_made += 1
print(f"[viz] saved {epd_made} EPD + {mp_made} meta-path per-node figures -> {OUT}/{{epd,metapath}}", flush=True)
