from __future__ import annotations

import json
import sys
from pathlib import Path

import repro_scripts.qrgbm_gpu_sharded as sharded_package
from repro.classical import QRGBMConfig as OriginalQRGBMConfig


QUANTILES = 19
ESTIMATORS = 100


def practical_config(**values):
    values = dict(values)
    values["quantiles"] = QUANTILES
    values["estimators"] = ESTIMATORS
    return OriginalQRGBMConfig(**values)


def main() -> None:
    # The paper does not disclose these two hyperparameters.  Override only the
    # tree count and marginal grid; every other declared setting and all ECC
    # semantics remain identical to the formal sharded implementation.
    sharded_package._implementation.QRGBMConfig = practical_config
    sharded_package.main()
    zone = None
    if "--zone" in sys.argv:
        zone = int(sys.argv[sys.argv.index("--zone") + 1])
    root = Path("outputs/full_reproduction")
    run = root / "qrgbm_sharded" if zone is None else root / "single_zone" / f"zone{zone}" / "qrgbm_sharded"
    declaration = {
        "baseline": "QRGBM practical declared reconstruction",
        "quantiles": QUANTILES,
        "estimators": ESTIMATORS,
        "reason": "the article does not disclose either value; 99x300 exceeded one day on the reproduction GPU",
        "unchanged": [
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "CUDA histogram training",
            "residual-rank ensemble copula coupling",
        ],
        "not_an_exact_author_implementation": True,
    }
    run.mkdir(parents=True, exist_ok=True)
    (run / "declared_practical_config.json").write_text(
        json.dumps(declaration, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
