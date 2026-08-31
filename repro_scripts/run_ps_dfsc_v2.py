"""Numerically stable PS-DFSC CLI.

This is the preferred executable entry point. It activates monetary scaling in
the CVXPYLayer before delegating to the versioned pipeline implementation.
"""

from ps_dfsc.stable_differentiable_suc import activate_stable_training_layer

activate_stable_training_layer()

from repro_scripts.run_ps_dfsc import main


if __name__ == "__main__":
    main()
