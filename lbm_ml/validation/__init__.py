"""A-posteriori validation of learned collision operators.

Currently exposes the freely-decaying turbulence test case of
Ortali/Gabbana et al. (2025), adapted to the 2D D2Q9 framework of this repo.
"""

from lbm_ml.validation.free_turbulence import (
    TurbulenceConfig,
    build_turbulent_ic,
    bgk_collide,
    ml_collide,
    run_free_decay,
    total_energy,
    log_derivative,
    aposteriori_error,
    run_validation,
)

__all__ = [
    "TurbulenceConfig",
    "build_turbulent_ic",
    "bgk_collide",
    "ml_collide",
    "run_free_decay",
    "total_energy",
    "log_derivative",
    "aposteriori_error",
    "run_validation",
]
