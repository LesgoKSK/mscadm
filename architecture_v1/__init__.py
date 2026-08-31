"""Architecture-v1 common-basis experiments."""

from .data import (
    ArchitectureDataBundle,
    ArchitectureFitDataBundle,
    ArchitectureSplitData,
    ArchitectureTrainDataBundle,
    build_architecture_v1_data,
    build_architecture_v1_fit_data,
    build_architecture_v1_train_data,
    load_nwp_hours,
)
from .protocol import (
    ArchitectureProtocol,
    build_architecture_protocol,
    write_protocol_manifest,
)

__all__ = [
    "ArchitectureDataBundle",
    "ArchitectureFitDataBundle",
    "ArchitectureProtocol",
    "ArchitectureSplitData",
    "ArchitectureTrainDataBundle",
    "build_architecture_protocol",
    "build_architecture_v1_data",
    "build_architecture_v1_fit_data",
    "build_architecture_v1_train_data",
    "load_nwp_hours",
    "write_protocol_manifest",
]
