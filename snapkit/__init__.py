"""
SnapKit v2 — Eisenstein lattice snap, temporal analysis, connectome detection,
harmony governor, and hypothesis sandbox.

Zero external dependencies. stdlib only.
"""

from snapkit.eisenstein import EisensteinInteger, eisenstein_snap, eisenstein_distance, eisenstein_round, eisenstein_round_naive
from snapkit.eisenstein_voronoi import eisenstein_snap_voronoi, eisenstein_snap_naive as eisenstein_snap_naive_voronoi
from snapkit.temporal import TemporalSnap, TemporalResult, BeatGrid
from snapkit.spectral import entropy, hurst_exponent, autocorrelation, spectral_summary
from snapkit.midi import FluxTensorMIDI, Room, TempoMap, MIDIEvent
from snapkit.connectome import (
    TemporalConnectome,
    CouplingType,
    RoomPair,
    ConnectomeResult,
)
from snapkit.governor import (
    HarmonyGovernor,
    ChannelState,
    FrictionAlarm,
    FrictionLevel,
)
from snapkit.sandbox import (
    HypothesisSandbox,
    LinearModel,
    CorrelationModel,
    HypothesisResult,
    SandboxScore,
)
from snapkit.executive import (
    ExecutiveAgent,
    DiagnosticEngine,
    ExecutiveAction,
    AgentConfig,
    ImprovisationResult,
)
from snapkit.midi_io import (
    MIDIBridge,
    SensorMapper,
    TempoDeriver,
    SensorReading,
    AgentAction,
    SENSOR_CC_MAP,
    ACTION_NOTE_MAP,
)
from snapkit.clever_tokens import (
    TokenLattice,
    CleverToken,
    ConstraintType,
    create_maritime_lattice,
)

__version__ = "2.3.0"
__all__ = [
    # Eisenstein
    "EisensteinInteger",
    "eisenstein_snap",
    "eisenstein_distance",
    "eisenstein_round",
    "eisenstein_round_naive",
    "eisenstein_snap_voronoi",
    # Temporal
    "TemporalSnap",
    "TemporalResult",
    "BeatGrid",
    # Spectral
    "entropy",
    "hurst_exponent",
    "autocorrelation",
    "spectral_summary",
    # MIDI
    "FluxTensorMIDI",
    "Room",
    "TempoMap",
    "MIDIEvent",
    # Connectome
    "TemporalConnectome",
    "CouplingType",
    "RoomPair",
    "ConnectomeResult",
    # Governor (v2.1)
    "HarmonyGovernor",
    "ChannelState",
    "FrictionAlarm",
    "FrictionLevel",
    # Sandbox (v2.1)
    "HypothesisSandbox",
    "LinearModel",
    "CorrelationModel",
    "HypothesisResult",
    "SandboxScore",
    # Executive (v2.2)
    "ExecutiveAgent",
    "DiagnosticEngine",
    "ExecutiveAction",
    "AgentConfig",
    "ImprovisationResult",
    # MIDI I/O (v2.2)
    "MIDIBridge",
    "SensorMapper",
    "TempoDeriver",
    "SensorReading",
    "AgentAction",
    "SENSOR_CC_MAP",
    "ACTION_NOTE_MAP",
    # Clever Tokens (v2.3)
    "TokenLattice",
    "CleverToken",
    "ConstraintType",
    "create_maritime_lattice",
]
