import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import torch
import torch.nn as nn
import implicit
import gdown
import os

class BPRMF(nn.Module):
    def __init__(self, n_users, n_items, embed_dim):
        super().__init__()
        self.user_emb  = nn.Embedding(n_users, embed_dim)
        self.item_emb  = nn.Embedding(n_items, embed_dim)
        self.item_bias = nn.Embedding(n_items, 1)

    @torch.no_grad()
    def recommend(self, user_idx, user_items, N=50,
                  filter_already_liked_items=True, items=None):
        u = torch.tensor([user_idx])
        eu = self.user_emb(u)
        scores = (eu @ self.item_emb.weight.T +
                  self.item_bias.weight.squeeze(-1)).squeeze(0)
        if items is not None:
            mask = torch.zeros(scores.shape[0], dtype=torch.bool)
            mask[torch.tensor(items)] = True
            scores[~mask] = -1e9
        if filter_already_liked_items:
            liked = user_items.indices
            scores[liked] = -1e9
        topk = torch.topk(scores, N)
        return topk.indices.numpy(), topk.values.numpy()

class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'BPRMF':
            return BPRMF
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b),
                                        map_location='cpu',
                                        weights_only=False)
        return super().find_class(module, name)

FILE_IDS = {
    "model_data.pkl":          "15QgJ-JLQyZUODo1yHS98h_tvDaqwdFcp",
    "dml_results.pkl":         "1TbOzVb3zmOEEgD_4NWiBKnhvDbhp162J",
    "step3_results.csv":       "1UhJjYejB9iuSO2_OzHBpSNGzIKX5gwXx",
    "item_daily_features.csv": "1Mc-bPy2DCtMfALS-L6_bKm3rCEhIeITs",
    "big_matrix.csv":          "1YvqPveVOjyqLId255UgehwTGzzOfiGVZ",
    "small_matrix.csv":        "1u4jGFv9KeyFCkYFingxWy_GZrGmtoBRC",
    "item_categories.csv":     "1TcUCbKjMf_tkfpxU5OdDCxltQczoOMLW",
}

def download_file(filename):
    os.makedirs("data", exist_ok=True)
    path = f"data/{filename}"
    if not os.path.exists(path):
        file_id = FILE_IDS[filename]
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False)
    return path

@st.cache_resource
def load_all():
    with st.spinner("Downloading data from Google Drive..."):
        for fname in FILE_IDS:
            download_file(fname)

    with open("data/model_data.pkl", "rb") as f:
        md = CPUUnpickler(f).load()
    with open("data/dml_results.pkl", "rb") as f:
        dml = pickle.load(f)

    step3      = pd.read_csv("data/step3_results.csv")
    item_daily = pd.read_csv("data/item_daily_features.csv")
    item_info  = (item_daily.sort_values("date")
                             .groupby("video_id").last()[
                                 ["video_tag_name", "like_cnt"]
                             ].reset_index())
    big = pd.read_csv("data/big_matrix.csv",
                      usecols=["user_id", "video_id", "watch_ratio"])

    er_raw = md["exposure_raw"]
    ic     = md["item_code2id"]
    if not isinstance(er_raw, dict):
        exp_raw = {int(ic[code]): int(er_raw.iloc[code])
                   for code in range(len(er_raw)) if code in ic}
    else:
        exp_raw = er_raw
    md["exposure_raw"] = exp_raw

    pb = md["propensity_big"]
    if isinstance(pb, dict) and not all(
            isinstance(v, (int, float)) for v in pb.values()):
        alpha_key = 0.5 if 0.5 in pb else list(pb.keys())[1]
        p_arr = pb[alpha_key]
        md["propensity_big"] = {int(ic[code]): float(p_arr[code])
                                 for code in range(len(p_arr)) if code in ic}
    elif not isinstance(pb, dict):
        md["propensity_big"] = {int(ic[code]): float(pb.iloc[code])
                                 for code in range(len(pb)) if code in ic}

    return md, dml, step3, item_info, big

md, dml, step3, item_info, big = load_all()

model      = md["model_baseline"]
prop       = md["propensity_big"]
exp_raw    = md["exposure_raw"]
id2code    = md["user_id2code"]
code2id    = md["item_code2id"]
candidates = md["candidate_item_codes"]
n_users    = md["n_users"]
n_items    = md["n_items"]

big_clean = big.dropna(subset=["user_id","video_id"]).reset_index(drop=True)
user_cat  = pd.Categorical(big_clean["user_id"])
item_cat  = pd.Categorical(big_clean["video_id"])
train_ui  = sparse.csr_matrix(
    (np.ones(len(big_clean)), (user_cat.codes, item_cat.codes)),
    shape=(len(user_cat.categories), len(item_cat.categories))
)

def get_recommendations(user_idx, lam, N=10, pool=50):
    rec_ids, rec_scores = model.recommend(
        user_idx, train_ui[user_idx],
        N=pool, filter_already_liked_items=True,
        items=candidates
    )
    reranked = []
    for code, score in zip(rec_ids, rec_scores):
        item_id = int(code2id[int(code)])
        p = prop.get(item_id, 1.0)
        ips_score = score * (1 - lam) + (score / max(p, 0.01)) * lam
        reranked.append((item_id, score, p, ips_score))
    reranked.sort(key=lambda x: x[3], reverse=True)
    return reranked[:N]

def get_item_info(item_id):
    row = item_info[item_info["video_id"] == item_id]
    if len(row) == 0:
        return "Unknown", 0
    tag  = row["video_tag_name"].values[0]
    like = row["like_cnt"].values[0]
    return (tag if pd.notna(tag) else "No tag",
            int(like) if pd.notna(like) else 0)

st.set_page_config(page_title="Causal Debiasing for RecSys", layout="wide")

st.title("Causal Debiasing for Recommender Systems")
st.markdown("Diagnosing and correcting popularity bias in recommendation logs using causal inference.")

tab1, tab2, tab3 = st.tabs(["IPS Debiasing Demo", "Bias Dashboard", "DML Analysis"])

with tab1:
    st.markdown("Compare Baseline vs IPS Re-ranking recommendations.")
    col_side, col_main = st.columns([1, 3])

    with col_side:
        st.subheader("Settings")
        user_list = list(id2code.keys())
        user_idx_slider = st.slider(
             "Select User (index)",
             min_value=0,
             max_value=len(user_list)-1,
             value=0,
            step=1
        )
        selected_user = user_list[user_idx_slider]
        st.caption(f"User ID: {selected_user}")
        lam = st.slider("lambda (IPS strength)", 0.0, 1.0, 0.0, 0.1)
        st.markdown("""
**lambda interpretation:**
- 0.0 → Baseline (no debiasing)
- 0.5 → Moderate debiasing
- 1.0 → Maximum IPS debiasing
""")

    user_idx = id2code.get(selected_user, list(id2code.values())[0])

    with col_main:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Baseline (lambda=0)")
            recs_base = get_recommendations(user_idx, lam=0.0)
            rows = []
            for rank, (item_id, score, p, _) in enumerate(recs_base, 1):
                tag, like = get_item_info(item_id)
                pop = exp_raw.get(item_id, 0)
                rows.append({"Rank": rank, "video_id": item_id,
                              "Tag": tag, "Popularity": int(pop), "Likes": like})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        with col2:
            st.subheader(f"IPS Re-ranking (lambda={lam})")
            recs_ips = get_recommendations(user_idx, lam=lam)
            rows = []
            for rank, (item_id, score, p, _) in enumerate(recs_ips, 1):
                tag, like = get_item_info(item_id)
                pop = exp_raw.get(item_id, 0)
                rows.append({"Rank": rank, "video_id": item_id,
                              "Tag": tag, "Popularity": int(pop), "Likes": like})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.divider()
        st.subheader("Popularity Comparison")
        pop_base = [exp_raw.get(r[0], 0) for r in recs_base]
        pop_ips  = [exp_raw.get(r[0], 0) for r in recs_ips]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(range(1, 11), pop_base, color="steelblue", alpha=0.8)
        axes[0].set_title("Baseline Top-10 Popularity")
        axes[0].set_xlabel("Rank")
        axes[0].set_ylabel("Interaction Count")
        axes[1].bar(range(1, 11), pop_ips, color="coral", alpha=0.8)
        axes[1].set_title(f"IPS (lambda={lam}) Top-10 Popularity")
        axes[1].set_xlabel("Rank")
        axes[1].set_ylabel("Interaction Count")
        plt.tight_layout()
        st.pyplot(fig)

        st.divider()
        c1, c2, c3 = st.columns(3)
        avg_base = np.mean(pop_base) if pop_base else 0
        avg_ips  = np.mean(pop_ips)  if pop_ips  else 0
        c1.metric("Baseline Avg Popularity", f"{avg_base:.1f}")
        c2.metric("IPS Avg Popularity", f"{avg_ips:.1f}")
        if avg_base > 0:
            c3.metric("Popularity Reduction",
                      f"{avg_base - avg_ips:.1f}",
                      delta=f"-{(avg_base-avg_ips)/avg_base*100:.1f}%")

with tab2:
    st.markdown("Visualizing popularity bias in KuaiRec big_matrix.")

    item_pop = big.groupby("video_id").size().sort_values(ascending=False)
    g_sorted = np.sort(item_pop.values.astype(float))
    n        = len(g_sorted)
    cum      = np.cumsum(g_sorted) / g_sorted.sum()
    x        = np.arange(1, n+1) / n
    gini     = float((2*np.sum(np.arange(1,n+1)*g_sorted) -
                      (n+1)*g_sorted.sum()) / (n*g_sorted.sum()))

    col1, col2, col3 = st.columns(3)
    col1.metric("Gini Coefficient", f"{gini:.4f}")
    top10 = item_pop.head(int(n*0.1)).sum() / item_pop.sum()
    col2.metric("Top 10% Items Exposure Share", f"{top10*100:.1f}%")
    col3.metric("Total Items", f"{n:,}")

    st.divider()
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    axes2[0].plot(x, cum, color="crimson", lw=2, label=f"Lorenz (Gini={gini:.3f})")
    axes2[0].plot([0,1],[0,1], "k--", alpha=0.5, label="Perfect equality")
    axes2[0].fill_between(x, x, cum, alpha=0.15, color="crimson")
    axes2[0].set_xlabel("Cumulative share of items")
    axes2[0].set_ylabel("Cumulative share of exposure")
    axes2[0].set_title("Lorenz Curve")
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)

    axes2[1].plot(np.arange(1, n+1), item_pop.values, color="steelblue", lw=1.2)
    axes2[1].set_xscale("log")
    axes2[1].set_yscale("log")
    axes2[1].set_xlabel("Item rank (log)")
    axes2[1].set_ylabel("Exposure count (log)")
    axes2[1].set_title("Long-tail Distribution")
    axes2[1].grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)

    st.divider()
    st.subheader("Alpha Sweep — Accuracy vs Diversity Trade-off")
    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
    axes3[0].plot(step3["alpha"], step3["recall@20"], "o-", color="steelblue", lw=2, ms=8, label="Recall@20")
    axes3[0].plot(step3["alpha"], step3["recall@50"], "o-", color="green", lw=2, ms=8, label="Recall@50")
    axes3[0].set_xlabel("alpha")
    axes3[0].set_ylabel("Recall")
    axes3[0].set_title("Accuracy vs alpha")
    axes3[0].legend()
    axes3[0].grid(True, alpha=0.3)

    axes3[1].plot(step3["alpha"], step3["coverage@20"], "o-", color="orange", lw=2, ms=8)
    axes3[1].set_xlabel("alpha")
    axes3[1].set_ylabel("Coverage@20")
    axes3[1].set_title("Coverage vs alpha")
    axes3[1].grid(True, alpha=0.3)

    axes3[2].plot(step3["coverage@20"], step3["recall@20"], "o-", color="purple", lw=2, ms=10)
    for _, row in step3.iterrows():
        axes3[2].annotate(f"a={row['alpha']}", (row["coverage@20"], row["recall@20"]),
                          textcoords="offset points", xytext=(6, 4), fontsize=9)
    axes3[2].set_xlabel("Coverage@20")
    axes3[2].set_ylabel("Recall@20")
    axes3[2].set_title("Trade-off Curve")
    axes3[2].grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3)

with tab3:
    st.markdown("Estimating the pure causal effect of popularity using Double Machine Learning.")

    theta_dml   = 0.0021
    theta_naive = -0.0649
    ci_dml      = dml.get("conf_int", [-0.0023, 0.0065])
    pval_dml    = 0.349
    r2_T        = 0.067
    r2_Y        = 0.431
    T_resid     = dml["T_resid"]
    Y_resid     = dml["Y_resid"]

    st.subheader("Causal Effect Estimation")
    c1, c2, c3 = st.columns(3)
    c1.metric("Naive theta", f"{theta_naive:.4f}")
    c2.metric("DML theta", f"{theta_dml:.4f}")
    c3.metric("p-value (DML)", f"{pval_dml:.3f}",
              delta="Not significant" if pval_dml > 0.05 else "Significant")

    st.info(f"""
**Interpretation:** After controlling for user/item features X,
the effect of popularity converges from **{theta_naive:.4f} to {theta_dml:.4f}**.
~103% of the negative correlation is explained by confounding.

→ "Popularity itself has no causal effect on watch_ratio" (p={pval_dml:.3f} > 0.05)
""")

    st.divider()
    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))

    methods = ["Naive\n(no control)", "DML\n(controlled)"]
    thetas  = [theta_naive, theta_dml]
    colors  = ["#E8593C", "#3B8BD4"]
    bars = axes4[0].bar(methods, thetas, color=colors, alpha=0.8, width=0.4)
    axes4[0].axhline(0, color="gray", lw=1, linestyle="--")
    for bar, val in zip(bars, thetas):
        axes4[0].text(bar.get_x() + bar.get_width()/2,
                      val + (0.001 if val > 0 else -0.003),
                      f"{val:.4f}", ha="center", va="bottom", fontsize=11)
    axes4[0].set_ylabel("theta (popularity → watch_ratio)")
    axes4[0].set_title("Naive vs DML — Confounding Size")
    axes4[0].grid(True, alpha=0.3, axis="y")

    sample_idx = np.random.choice(len(T_resid), min(5000, len(T_resid)), replace=False)
    axes4[1].scatter(T_resid[sample_idx], Y_resid[sample_idx], alpha=0.2, s=5, color="purple")
    x_line = np.linspace(T_resid.min(), T_resid.max(), 100)
    axes4[1].plot(x_line, theta_dml * x_line, color="red", lw=2, label=f"theta = {theta_dml:.4f}")
    axes4[1].axhline(0, color="gray", lw=0.5, linestyle="--")
    axes4[1].axvline(0, color="gray", lw=0.5, linestyle="--")
    axes4[1].set_xlabel("T residual (e_T)")
    axes4[1].set_ylabel("Y residual (e_Y)")
    axes4[1].set_title("DML Stage 3: e_Y ~ e_T")
    axes4[1].legend()
    plt.tight_layout()
    st.pyplot(fig4)

    st.divider()
    st.subheader("Nuisance R-squared Diagnostics")
    r1, r2 = st.columns(2)
    r1.metric("T ~ X  R2", f"{r2_T:.3f}")
    r2.metric("Y ~ X  R2", f"{r2_Y:.3f}")
    st.markdown(f"""
> **R2(T~X) = {r2_T:.3f}** → Popularity is barely explained by user/item features  
> **R2(Y~X) = {r2_Y:.3f}** → Watch ratio is well explained by features  
> → The asymmetry is the core evidence of 100% confounding
""")
