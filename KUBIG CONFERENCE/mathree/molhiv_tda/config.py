"""Project-wide paths and hyperparameters."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
CACHE_ROOT = PROJECT_ROOT / "cache"
RESULTS_ROOT = PROJECT_ROOT / "results"

DATASET_NAME = "ogbg-molhiv"
SMILES_CSV = DATASET_ROOT / "ogbg_molhiv" / "mapping" / "hiv.csv"

# Feature cache files
MW_CACHE = CACHE_ROOT / "molecular_weight.pt"
BOND_TDA_CACHE = CACHE_ROOT / "bond_tda.pt"
TDA_3D_CACHE = CACHE_ROOT / "tda_3d.pt"
EDGE_DIST_CACHE = CACHE_ROOT / "edge_dist_3d.pt"
EDGE_ELECTRO_CACHE = CACHE_ROOT / "edge_electrostatic.pt"
MW_STATS_CACHE = CACHE_ROOT / "mw_stats.pt"
TDA_STATS_CACHE = CACHE_ROOT / "tda_stats.pt"
TDA_3D_STATS_CACHE = CACHE_ROOT / "tda_3d_stats.pt"

# Model defaults (PDGNN backbone from TLC-GNN)
EMB_DIM = 300
NUM_BACKBONE_LAYERS = 4  # PDGNN uses 4 effective conv layers in TLC-GNN
DROPOUT = 0.5
# Training defaults
DEFAULT_DEVICE = "auto"  # cuda if available, else cpu
BATCH_SIZE = 32
NUM_WORKERS = 4  # DataLoader workers (use 0 on CPU-only debug)
LR = 1e-3
WEIGHT_DECAY = 0.0
EPOCHS = 100
PATIENCE = 10

# TDA defaults
PI_RESOLUTION = 5
PI_SIGMA = 0.05
BOND_TDA_DIM = PI_RESOLUTION * PI_RESOLUTION * 2  # H0 + H1 persistence images

# 3D TDA defaults
RIPS_MAX_EDGE = 4.0
RIPS_MAX_DIM = 2
TDA_3D_DIM = PI_RESOLUTION * PI_RESOLUTION * 3  # H0, H1, H2

# 3D distance filtration for PDGNN (conformer edge lengths)
DIST_FILTRATION_TAUS = [1.5, 2.0, 2.5, 3.0, 4.0]

# Graph-topology (hop-distance) Rips multifiltration lenses
GRAPH_RIPS_TDA_CACHE = CACHE_ROOT / "graph_rips_tda.pt"        # A: full graph hop distance
AROMATIC_RIPS_TDA_CACHE = CACHE_ROOT / "aromatic_rips_tda.pt"  # B: aromatic-bond subgraph
RIPS_HOP_MAX = 5.0  # cap on hop distance (captures rings up to ~size 10)
GRAPH_RIPS_DIM = PI_RESOLUTION * PI_RESOLUTION * 2  # H0 + H1 persistence images
# Dedicated PI sigma for hop-distance Rips images: the (0..RIPS_HOP_MAX) birth/pers
# range spans ~5 units over 5 pixels (~1 unit/pixel), so a too-narrow sigma (e.g.
# the 0.05 used for other caches) yields near all-zero images. 0.5 ~= half a pixel.
GRAPH_RIPS_SIGMA = 0.5
