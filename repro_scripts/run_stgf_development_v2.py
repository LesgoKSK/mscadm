from __future__ import annotations

from repro_scripts import run_stgf_development as runner
from stgf_flow.training_v2 import PhysicalCenterSTGFTrainer


if __name__ == "__main__":
    runner.STGFTrainer = PhysicalCenterSTGFTrainer
    runner.main()
