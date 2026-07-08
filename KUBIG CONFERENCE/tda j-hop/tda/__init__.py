"""TDA: HAN + GTN + PDGNN 파이프라인 (이종 그래프 위상 특징 노드 분류).

파이프라인 = GTN(메타패스 자동 발견) -> 채널별 동종 그래프 -> PDGNN(neural EPD)
-> per-node persistence image -> semantic attention fusion -> HAN 노드 분류.

자세한 설계는 docs/design.ko.md 참고.
"""

__version__ = "0.1.0"
