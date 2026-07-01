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


# 유한·비capping 점 + 루프가 많은 (node, scale) 탐색 → 의미있는 위상 예시
rng = np.random.RandomState(0)
sample = list(sub.nodes) + rng.choice(H.size(1), size=min(300, H.size(1)), replace=False).tolist()
best = None
for vv in sample:
    for kk in range(K):
        r = _ego_filt_edges(g_ch, int(vv), cfg["pdgnn"]["hop"], filts_by_k[kk], cfg["pdgnn"]["max_nodes"])
        if r is None or r[1].shape[1] == 0:
            continue
        pts, ncap, loops = epd_full(r[0], r[1])
        score = ncap + 2 * loops
        if best is None or score > best[4]:
            best = (int(vv), kk, r, pts, score)
    if best is not None and best[4] >= 20:
        break
v, k, res, gt, _ = best
filt, eei, nodes = res
node_filt = filts_by_k[k]
print(f"[viz] EPD example: node {v}, scale {k}, ego {len(nodes)} nodes/{eei.shape[1]//2} edges, "
      f"{len(gt)} pts (score={best[4]})", flush=True)
res = cfg["pdgnn"]["pi_resolution"]
max_filt = float(filt.max())
births, deaths = gt[:, 0], gt[:, 1]
persg = deaths - births
capped = deaths >= max_filt - 1e-6        # essential components (capped at max_filt)
# persistence image (range fit to actual point spread, so structure is visible)
img_t = PersistenceImage(resolution=res, sigma=max(persg.max() / 6, 0.05),
                         birth_range=(float(births.min()), float(births.max())),
                         pers_range=(0.0, float(max(persg.max(), 1e-3))))
pi = img_t.transform(gt).reshape(res, res)

ego = nx.Graph(); ego.add_nodes_from(range(len(nodes)))
for a_, b_ in zip(eei[0], eei[1]):
    ego.add_edge(int(a_), int(b_))
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
# (1) ego subgraph colored by HKS filter value
posE = nx.spring_layout(ego, seed=0)
nx.draw_networkx_edges(ego, posE, ax=ax[0], alpha=0.35, width=0.6)
sc = nx.draw_networkx_nodes(ego, posE, ax=ax[0], node_size=90, node_color=filt, cmap="coolwarm")
plt.colorbar(sc, ax=ax[0], fraction=0.046, label="HKS filter f(v)")
ax[0].set_title(f"(1) node {v} ego ({len(nodes)} nodes, {eei.shape[1]//2} edges)\ncolor = HKS filtration value",
                fontsize=10); ax[0].axis("off")
# (2) persistence diagram: finite (red) vs essential/capped (gray)
ax[1].scatter(births[~capped], deaths[~capped], s=45, c="crimson", zorder=3, label=f"finite ({int((~capped).sum())})")
ax[1].scatter(births[capped], deaths[capped], s=25, c="gray", alpha=0.6, zorder=2, label=f"essential ({int(capped.sum())})")
lim = [float(births.min()) - 0.2, float(deaths.max()) + 0.2]
ax[1].plot(lim, lim, "k--", alpha=0.5); ax[1].legend(fontsize=8)
ax[1].set_xlabel("birth"); ax[1].set_ylabel("death")
ax[1].set_title(f"(2) persistence diagram ({len(gt)} points)\nmax persistence = {persg.max():.2f}", fontsize=10)
# (3) persistence image (vectorized EPD -> model feature; range fit to data)
im = ax[2].imshow(pi, cmap="magma", origin="lower")
ax[2].set_title(f"(3) persistence image (res {res})\nvectorized EPD -> model feature", fontsize=10)
ax[2].set_xticks([]); ax[2].set_yticks([]); plt.colorbar(im, ax=ax[2], fraction=0.046)
fig.suptitle(f"{DS} channel {ch}, scale {k}: channel graph -> node ego + HKS -> EPD -> persistence image",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUT}/epd_process.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# ===== EPD gallery: ~100 random nodes' persistence images (scale 0) =====
rng2 = np.random.RandomState(1)
gal = rng2.choice(H.size(1), size=min(100, H.size(1)), replace=False)
imgr = PersistenceImage(resolution=res, sigma=0.4, birth_range=(-3.0, 3.0), pers_range=(0.0, 0.5))
ncol = 10
nrow = int(np.ceil(len(gal) / ncol))
figg, axg = plt.subplots(nrow, ncol, figsize=(ncol * 1.2, nrow * 1.2))
nemp = 0
for ax_, vv in zip(axg.flat, gal):
    r = _ego_filt_edges(g_ch, int(vv), cfg["pdgnn"]["hop"], filts_by_k[0], cfg["pdgnn"]["max_nodes"])
    if r is None or r[1].shape[1] == 0:
        ax_.axis("off"); nemp += 1; continue
    pts, _, _ = epd_full(r[0], r[1])
    img = imgr.transform(pts).reshape(res, res) if len(pts) else np.zeros((res, res))
    ax_.imshow(img, cmap="magma", origin="lower"); ax_.set_xticks([]); ax_.set_yticks([])
for ax_ in axg.flat[len(gal):]:
    ax_.axis("off")
figg.suptitle(f"{DS} channel {ch}: persistence images of {len(gal)} random nodes (scale 0)", fontsize=12)
figg.tight_layout(rect=[0, 0, 1, 0.97])
figg.savefig(f"{OUT}/epd_gallery.png", dpi=110, bbox_inches="tight"); plt.close(figg)
print(f"[viz] saved epd_gallery.png ({len(gal)} nodes, {nemp} empty egos)", flush=True)
print(f"[viz] saved epd_process.png  (node {v}, ego {len(nodes)} nodes, EPD {len(gt)} pts)", flush=True)
