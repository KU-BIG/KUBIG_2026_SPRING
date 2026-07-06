"""
visualize.py — poles_decision.csv 시각화
  1. output/map_danger.html   : Folium 인터랙티브 지도
  2. output/stats_dashboard.png : Matplotlib 통계 대시보드
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import folium
from folium.plugins import HeatMap, MarkerCluster
from pathlib import Path

OUT = Path("output")
df = pd.read_csv("output/poles_decision.csv")

print(f"전봇대 총 수: {len(df):,}")
print(f"decision=1 (위험): {df['decision'].sum():,} ({df['decision'].mean()*100:.1f}%)")
print(f"decision=0 (안전): {(df['decision']==0).sum():,}")

# ──────────────────────────────────────────────
# 1. Folium 인터랙티브 지도
# ──────────────────────────────────────────────
print("\n[1/2] Folium 지도 생성 중...")

center_lat = df["lat"].mean()
center_lon = df["lon"].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=11,
    tiles="CartoDB positron"
)

# ── HeatMap: adjusted_score 기반 위험 밀도 ──
heat_data = df[["lat", "lon", "adjusted_score"]].values.tolist()
HeatMap(
    heat_data,
    name="위험 밀도 (Heatmap)",
    min_opacity=0.3,
    max_zoom=15,
    radius=12,
    blur=10,
    gradient={0.2: "#ffffb2", 0.4: "#fecc5c", 0.6: "#fd8d3c", 0.8: "#f03b20", 1.0: "#bd0026"},
).add_to(m)

# ── 위험 전봇대(decision=1) 마커 ──
danger = df[df["decision"] == 1]
cluster = MarkerCluster(name="위험 전봇대 (decision=1)", show=False)
for _, row in danger.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=4,
        color="#d73027",
        fill=True,
        fill_color="#d73027",
        fill_opacity=0.7,
        popup=folium.Popup(
            f"<b>pole_id: {int(row['pole_id'])}</b><br>"
            f"adjusted_score: {row['adjusted_score']:.4f}<br>"
            f"terrain_weight: {row['terrain_weight']:.4f}<br>"
            f"lightning_count_2km: {row['lightning_count_2km']:.0f}",
            max_width=220
        ),
    ).add_to(cluster)
cluster.add_to(m)

# ── 범례 ──
legend_html = """
<div style="position: fixed; bottom: 40px; left: 40px; z-index: 1000;
     background-color: white; padding: 14px 18px; border-radius: 8px;
     border: 1px solid #ccc; font-family: Arial; font-size: 13px;
     box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
  <b>강원도 전봇대 산불 위험</b><br><br>
  <span style="color:#bd0026;">●</span> 위험 전봇대 (decision=1)<br>
  <span style="color:#4575b4;">●</span> 안전 전봇대 (decision=0)<br><br>
  <span style="background: linear-gradient(to right, #ffffb2, #fecc5c, #fd8d3c, #bd0026);
               display:inline-block; width:100px; height:12px; border-radius:3px;"></span><br>
  낮음 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 높음<br>
  <small>adjusted_score 기반 위험 밀도</small>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# ── 제목 ──
title_html = """
<div style="position: fixed; top: 15px; left: 50%; transform: translateX(-50%);
     z-index: 1000; background-color: rgba(255,255,255,0.92);
     padding: 10px 24px; border-radius: 8px; border: 1px solid #aaa;
     font-family: Arial; font-size: 16px; font-weight: bold;
     box-shadow: 2px 2px 6px rgba(0,0,0,0.15);">
  강원도 전봇대 산불 위험 예측 지도 &nbsp;|&nbsp;
  전봇대 115,511개 &nbsp;·&nbsp; 위험 20,604개 (17.8%)
</div>
"""
m.get_root().html.add_child(folium.Element(title_html))

folium.LayerControl(collapsed=False).add_to(m)

map_path = OUT / "map_danger.html"
m.save(str(map_path))
print(f"   저장: {map_path}")

# ──────────────────────────────────────────────
# 2. Matplotlib 통계 대시보드
# ──────────────────────────────────────────────
print("[2/2] 통계 대시보드 생성 중...")

fig = plt.figure(figsize=(18, 12), facecolor="#0f1117")
fig.suptitle(
    "강원도 전봇대 산불 위험 예측 — 분석 결과 대시보드",
    fontsize=18, fontweight="bold", color="white", y=0.98
)

axes_color = "#1a1d27"
text_color = "white"
grid_color = "#2a2d3a"

import matplotlib.font_manager as fm
fm.fontManager.addfont("C:/Windows/Fonts/malgun.ttf")
plt.rcParams["font.family"] = fm.FontProperties(fname="C:/Windows/Fonts/malgun.ttf").get_name()
plt.rcParams["axes.unicode_minus"] = False

# ── (1) 지도: 전봇대 분포 scatter ──
ax1 = fig.add_axes([0.03, 0.52, 0.44, 0.40], facecolor=axes_color)
safe = df[df["decision"] == 0]

ax1.scatter(safe["lon"], safe["lat"], s=0.3, c="#4575b4", alpha=0.3, label="안전 (0)", rasterized=True)
ax1.scatter(danger["lon"], danger["lat"], s=0.8, c="#d73027", alpha=0.6, label="위험 (1)", rasterized=True)
ax1.set_title("전봇대 위험 분포 지도", color=text_color, fontsize=13, pad=8)
ax1.set_xlabel("경도", color=text_color, fontsize=10)
ax1.set_ylabel("위도", color=text_color, fontsize=10)
ax1.tick_params(colors=text_color)
for sp in ax1.spines.values(): sp.set_color(grid_color)
ax1.legend(loc="upper left", facecolor=axes_color, labelcolor=text_color, fontsize=10,
           markerscale=6, framealpha=0.8)

# ── (2) adjusted_score 분포 ──
ax2 = fig.add_axes([0.55, 0.52, 0.20, 0.40], facecolor=axes_color)
bins = np.linspace(df["adjusted_score"].min(), df["adjusted_score"].max(), 60)
ax2.hist(safe["adjusted_score"], bins=bins, color="#4575b4", alpha=0.7, label="안전")
ax2.hist(danger["adjusted_score"], bins=bins, color="#d73027", alpha=0.7, label="위험")
thresh = df[df["decision"]==1]["adjusted_score"].min()
ax2.axvline(thresh, color="#fecc5c", linewidth=1.5, linestyle="--", label=f"임계값 {thresh:.3f}")
ax2.set_title("adjusted_score 분포", color=text_color, fontsize=13, pad=8)
ax2.set_xlabel("adjusted_score", color=text_color, fontsize=10)
ax2.set_ylabel("전봇대 수", color=text_color, fontsize=10)
ax2.tick_params(colors=text_color)
for sp in ax2.spines.values(): sp.set_color(grid_color)
ax2.legend(facecolor=axes_color, labelcolor=text_color, fontsize=9)

# ── (3) terrain_weight 분포 ──
ax3 = fig.add_axes([0.79, 0.52, 0.20, 0.40], facecolor=axes_color)
ax3.hist(df["terrain_weight"], bins=50, color="#74add1", edgecolor="none")
ax3.set_title("terrain_weight 분포", color=text_color, fontsize=13, pad=8)
ax3.set_xlabel("terrain_weight", color=text_color, fontsize=10)
ax3.set_ylabel("전봇대 수", color=text_color, fontsize=10)
ax3.tick_params(colors=text_color)
for sp in ax3.spines.values(): sp.set_color(grid_color)

# ── (4) 도넛 차트: decision 비율 ──
ax4 = fig.add_axes([0.03, 0.04, 0.18, 0.40], facecolor=axes_color)
sizes = [len(safe), len(danger)]
colors_pie = ["#4575b4", "#d73027"]
wedges, texts, autotexts = ax4.pie(
    sizes, labels=["안전 (0)", "위험 (1)"],
    colors=colors_pie, autopct="%1.1f%%",
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(width=0.5, edgecolor=axes_color, linewidth=2)
)
for t in texts: t.set_color(text_color); t.set_fontsize(11)
for at in autotexts: at.set_color("white"); at.set_fontsize(11); at.set_fontweight("bold")
ax4.set_title("Decision 비율", color=text_color, fontsize=13, pad=8)

# ── (5) 낙뢰 밀도 비교 ──
ax5 = fig.add_axes([0.27, 0.04, 0.20, 0.40], facecolor=axes_color)
lightning_col = "lightning_count_2km"
bins_l = np.linspace(0, df[lightning_col].quantile(0.99), 40)
ax5.hist(safe[lightning_col].clip(upper=df[lightning_col].quantile(0.99)),
         bins=bins_l, color="#4575b4", alpha=0.7, label="안전", density=True)
ax5.hist(danger[lightning_col].clip(upper=df[lightning_col].quantile(0.99)),
         bins=bins_l, color="#d73027", alpha=0.7, label="위험", density=True)
ax5.set_title("낙뢰 밀도 분포 (반경 2km)", color=text_color, fontsize=13, pad=8)
ax5.set_xlabel("낙뢰 건수", color=text_color, fontsize=10)
ax5.set_ylabel("밀도", color=text_color, fontsize=10)
ax5.tick_params(colors=text_color)
for sp in ax5.spines.values(): sp.set_color(grid_color)
ax5.legend(facecolor=axes_color, labelcolor=text_color, fontsize=9)

# ── (6) 피처 박스플롯 비교 ──
ax6 = fig.add_axes([0.55, 0.04, 0.44, 0.40], facecolor=axes_color)
features = ["max_anomaly_score", "terrain_weight", "adjusted_score", "lightning_count_2km"]
feat_labels = ["Anomaly\nScore", "Terrain\nWeight", "Adjusted\nScore", "Lightning\nCount"]

pos_safe   = [1, 4, 7, 10]
pos_danger = [2, 5, 8, 11]

for i, (feat, pos_s, pos_d) in enumerate(zip(features, pos_safe, pos_danger)):
    s_data = safe[feat].clip(lower=safe[feat].quantile(0.01),
                              upper=safe[feat].quantile(0.99)).values
    d_data = danger[feat].clip(lower=danger[feat].quantile(0.01),
                                upper=danger[feat].quantile(0.99)).values
    bp1 = ax6.boxplot(s_data, positions=[pos_s], widths=0.7,
                      patch_artist=True, notch=False,
                      boxprops=dict(facecolor="#4575b4", alpha=0.8),
                      medianprops=dict(color="white", linewidth=2),
                      whiskerprops=dict(color="#74add1"),
                      capprops=dict(color="#74add1"),
                      flierprops=dict(marker=".", color="#74add1", alpha=0.3, markersize=2))
    bp2 = ax6.boxplot(d_data, positions=[pos_d], widths=0.7,
                      patch_artist=True, notch=False,
                      boxprops=dict(facecolor="#d73027", alpha=0.8),
                      medianprops=dict(color="white", linewidth=2),
                      whiskerprops=dict(color="#f03b20"),
                      capprops=dict(color="#f03b20"),
                      flierprops=dict(marker=".", color="#f03b20", alpha=0.3, markersize=2))

ax6.set_xticks([1.5, 4.5, 7.5, 10.5])
ax6.set_xticklabels(feat_labels, color=text_color, fontsize=10)
ax6.set_title("피처별 안전 vs 위험 비교 (박스플롯)", color=text_color, fontsize=13, pad=8)
ax6.tick_params(colors=text_color)
for sp in ax6.spines.values(): sp.set_color(grid_color)
patch_s = mpatches.Patch(color="#4575b4", label="안전 (0)")
patch_d = mpatches.Patch(color="#d73027", label="위험 (1)")
ax6.legend(handles=[patch_s, patch_d], facecolor=axes_color, labelcolor=text_color, fontsize=10)

# ── 요약 텍스트 ──
summary = (
    f"총 전봇대: {len(df):,}개  |  "
    f"위험 (decision=1): {len(danger):,}개 ({len(danger)/len(df)*100:.1f}%)  |  "
    f"Precision: 0.3779  ·  Recall: 0.7087  ·  F2: 0.6032"
)
fig.text(0.5, 0.005, summary, ha="center", va="bottom",
         fontsize=11, color="#aaaaaa")

stats_path = OUT / "stats_dashboard.png"
plt.savefig(str(stats_path), dpi=150, bbox_inches="tight",
            facecolor="#0f1117", edgecolor="none")
plt.close()
print(f"   저장: {stats_path}")

print("\n완료!")
print(f"  → {map_path}")
print(f"  → {stats_path}")
