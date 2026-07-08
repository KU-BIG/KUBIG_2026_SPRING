# team-tda-conference (shared elice checkout)

Copy of elice `~/tda-conference` — upstream [HyunjuYun1009/tda-conference](https://github.com/HyunjuYun1009/tda-conference).

Contains team-shared MolHIV/TDA experiments not in the jonghyun fork:

- `molhiv_tda/CURRENT_APPROACH.md`, `EXPERIMENT_LOG.md`
- Multifiltration: `train/train_pdgnn_multifiltration.py`, `scripts/screen_multifiltration.py`, `confirm_multifiltration.py`, `sweep_dropout_multifilt.py`
- Charge TDA: `features/charge_filtration_tda.py`, `scripts/preprocess_charge_tda.py`
- 3D train variants: `train_pdgnn_molhiv_3d.py`, `3d2.py`, `3dver2.py`
- `scripts/multiseed_baseline_vs_bondtda.py`, `sweep_pdgnn_hparams.py`

Dataset/cache/checkpoints excluded (regenerate on GPU).
