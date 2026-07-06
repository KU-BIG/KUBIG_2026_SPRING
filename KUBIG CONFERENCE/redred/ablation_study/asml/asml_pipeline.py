"""asml_pipeline.py — ASML 파이프라인 순차 실행기"""
import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("Step 1: Feature Matrix",   os.path.join(BASE_DIR, "asml1_features.py")),
    ("Step 2: Ablation + SHAP",  os.path.join(BASE_DIR, "asml2_ablation.py")),
]

def main() -> None:
    print("=" * 55)
    print("  ASML Pipeline — 전체 실행")
    print("=" * 55)
    for name, script in STEPS:
        print(f"\n{'─'*55}")
        print(f"  {name}")
        print(f"{'─'*55}")
        subprocess.run([sys.executable, script], check=True)
    print("\n" + "=" * 55)
    print("  전체 파이프라인 완료.")
    print("=" * 55)

if __name__ == "__main__":
    main()
