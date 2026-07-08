from setuptools import find_packages, setup

setup(
    name="tda",
    version="0.1.0",
    description="HAN + GTN + PDGNN 파이프라인 (이종 그래프 위상 특징 노드 분류)",
    packages=find_packages(include=["tda", "tda.*"]),
    python_requires=">=3.8",
)
