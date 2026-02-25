## Overview

This project aims to predict **3D voxel occupancy** (3D Occupancy Prediction, 3DOP) by combining:

- **VoxelNet-style LiDAR voxel feature encoding** for geometric precision
- **BEVDepth-inspired camera branch** for dense semantic/contextual cues
- **Fusion + occupancy head** to produce voxel-wise occupancy predictions in 3D grids

The main goal is to study how **multimodal fusion (LiDAR + Camera)** improves occupancy prediction.

## Architecture
<img width="1347" height="664" alt="image" src="https://github.com/user-attachments/assets/021a9ff7-c650-4e15-99a6-dc630d9c5330" />

### Input
- LiDAR point cloud (x, y, z, intensity, ...)
- Multi-camera images (6-view setup)

### LiDAR Branch (VoxelNet-style)
1. Point voxelization within predefined 3D range
2. Voxel feature encoding (VFE / pointwise + voxel aggregation)
3. Sparse 4D Tensor

### Camera Branch (BEVDepth-inspired)
1. Image backbone + neck
2. Depth-aware feature lifting (image -> Voxel Space)
3. Sparse 4D Tensor

### Fusion
- Feature-level fusion in Voxel space (e.g., concat + conv fusion)

### Occupancy Head
- Predict voxel-wise occupancy logits on a fixed 3D grid (D x H x W)
- Binary occupancy or semantic occupancy (depending on setup)


## Visualization

<img width="1226" height="351" alt="image" src="https://github.com/user-attachments/assets/c06d32cd-37f8-4943-a19c-937e23cc1374" />

## DataSet
NuScenes-via-Occ3D-2Hz
- https://www.nuscenes.org/
- https://github.com/tasl-lab/UniOcc

## Team
- 최민석 (22기)
- 조호평 (23기)
- 김석우 (23기)
