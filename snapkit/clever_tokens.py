"""
Clever Tokens — Constraint token generation from Eisenstein lattice coordinates.

System prompts become lattice addresses instead of free text. Snapping
forces output to the nearest valid state. The lattice IS the conservation law.

The idea: generic natural language creates a massive unconstrained probability
distribution. A clever token is a specialty fastener — a highly specific,
structurally engineered token that collapses the model's degrees of freedom
and forces attention through a stabilized manifold.

Each token encodes:
  - A lattice coordinate (a, b) in the Eisenstein A₂ lattice
  - A constraint type (behavioral boundary)
  - A snap radius (how far the agent can deviate before snapping back)
  - A friction signature (what entropy/Hurst values are expected in harmony)

Zero external dependencies. stdlib only. Python ≥ 3.10.
"""

import math
import hashlib
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

from snapkit.eisenstein import EisensteinInteger, eisenstein_snap, eisenstein_round
from snapkit.eisenstein_voronoi import eisenstein_snap_voronoi


class ConstraintType(IntEnum):
    """Types of behavioral constraints a token can enforce."""
    RIGID = 0       # No deviation allowed (hard boundary)
    ELASTIC = 1     # Small deviation within snap radius (soft boundary)
    PERIODIC = 2    # Must return to lattice point within period (rhythmic)
    COUPLED = 3     # Must resonate with another token (harmonic)
    DIRECTIVE = 4   # Points toward a target state (T-minus)


@dataclass(frozen=True, slots=True)
class CleverToken:
    """A constraint token anchored to an Eisenstein lattice point.

    The token is a specialty fastener: when injected into a prompt,
    it collapses the model's degrees of freedom around the lattice point.
    """
    identifier: str           # Human-readable token name
    lattice_point: EisensteinInteger  # Position in constraint space
    constraint_type: ConstraintType
    snap_radius: float        # Max allowed deviation from lattice point
    expected_entropy: float   # Expected spectral entropy in harmony (bits)
    expected_hurst: float     # Expected Hurst exponent in harmony
    directive_target: Optional[EisensteinInteger] = None  # For DIRECTIVE type
    coupled_with: Optional[str] = None  # For COUPLED type
    metadata: str = ""        # Free-form annotation

    @property
    def complex_position(self) -> complex:
        return self.lattice_point.complex

    @property
    def norm(self) -> float:
        return math.sqrt(self.lattice_point.norm_squared)

    def deviation_of(self, observed_entropy: float, observed_hurst: float) -> float:
        """How far the observed state is from this token's harmony point.

        Returns Euclidean distance in (entropy, hurst) space.
        """
        # Normalize entropy to 0-1 range (assume max ~8 bits)
        norm_e_obs = min(observed_entropy / 8.0, 1.0)
        norm_e_exp = min(self.expected_entropy / 8.0, 1.0)
        return math.sqrt(
            (norm_e_obs - norm_e_exp) ** 2
            + (observed_hurst - self.expected_hurst) ** 2
        )

    def is_in_harmony(
        self, observed_entropy: float, observed_hurst: float,
    ) -> bool:
        """Check if observed state is within snap radius of this token."""
        return self.deviation_of(observed_entropy, observed_hurst) <= self.snap_radius

    def render(self) -> str:
        """Render the token as a string for injection into prompts."""
        parts = [
            f"[{self.identifier}",
            f"  lat=({self.lattice_point.a},{self.lattice_point.b})",
            f"  type={self.constraint_type.name.lower()}",
            f"  snap={self.snap_radius:.3f}",
        ]
        if self.directive_target:
            parts.append(
                f"  target=({self.directive_target.a},{self.directive_target.b})"
            )
        if self.coupled_with:
            parts.append(f"  couple={self.coupled_with}")
        if self.metadata:
            parts.append(f"  note={self.metadata}")
        parts.append("]")
        return "\n".join(parts)


class TokenLattice:
    """Manages a collection of clever tokens as a constraint space.

    Tokens are placed on the Eisenstein lattice. An agent's behavioral
    state maps to a point in (entropy, hurst) space, which is snapped
    to the nearest token. If no token is within snap radius, the agent
    is "off-lattice" and the Executive should be called.

    Usage:
        lattice = TokenLattice()
        helm_token = lattice.register_token(
            "helm:steady",
            lattice_coord=(3, 1),
            constraint_type=ConstraintType.ELASTIC,
            snap_radius=0.3,
            expected_entropy=0.5,
            expected_hurst=0.7,
        )
        # Check if agent is on-lattice:
        nearest = lattice.snap(entropy=0.6, hurst=0.65)
        # Returns the nearest token + deviation distance
    """

    def __init__(self):
        self._tokens: Dict[str, CleverToken] = {}
        # Map from (entropy, hurst) space to lattice space
        # We use a scaling factor to convert behavioral metrics to lattice coords
        self._scale: float = 10.0  # 1 unit of entropy/hurst = 10 lattice units

    def register_token(
        self,
        identifier: str,
        lattice_coord: Tuple[int, int],
        constraint_type: ConstraintType = ConstraintType.ELASTIC,
        snap_radius: float = 0.3,
        expected_entropy: float = 0.5,
        expected_hurst: float = 0.6,
        directive_target: Optional[Tuple[int, int]] = None,
        coupled_with: Optional[str] = None,
        metadata: str = "",
    ) -> CleverToken:
        """Register a new constraint token on the lattice."""
        if identifier in self._tokens:
            raise ValueError(f"token '{identifier}' already exists")

        ei = EisensteinInteger(lattice_coord[0], lattice_coord[1])
        target = None
        if directive_target:
            target = EisensteinInteger(directive_target[0], directive_target[1])

        token = CleverToken(
            identifier=identifier,
            lattice_point=ei,
            constraint_type=constraint_type,
            snap_radius=snap_radius,
            expected_entropy=expected_entropy,
            expected_hurst=expected_hurst,
            directive_target=target,
            coupled_with=coupled_with,
            metadata=metadata,
        )
        self._tokens[identifier] = token
        return token

    def snap(
        self, entropy: float, hurst: float,
    ) -> Tuple[Optional[CleverToken], float, bool]:
        """Snap observed behavioral state to nearest token.

        Returns (nearest_token, deviation_distance, is_on_lattice).
        If no tokens registered, returns (None, inf, False).
        """
        if not self._tokens:
            return None, float('inf'), False

        best_token = None
        best_deviation = float('inf')

        for token in self._tokens.values():
            dev = token.deviation_of(entropy, hurst)
            if dev < best_deviation:
                best_deviation = dev
                best_token = token

        is_on = best_deviation <= best_token.snap_radius if best_token else False
        return best_token, best_deviation, is_on

    def render_prompt(self, token_ids: List[str]) -> str:
        """Render multiple tokens as a constraint block for a system prompt.

        This replaces free-text instructions with structured lattice anchors.
        """
        lines = ["# Constraint Lattice", "# Active tokens:"]
        for tid in token_ids:
            if tid in self._tokens:
                lines.append(self._tokens[tid].render())
            else:
                lines.append(f"# WARNING: token '{tid}' not found")
        lines.append(f"# Total constraints: {len(token_ids)}")
        lines.append(
            f"# Lattice coverage: {len(token_ids)}/{len(self._tokens)} tokens active"
        )
        return "\n".join(lines)

    def get_token(self, identifier: str) -> CleverToken:
        return self._tokens[identifier]

    @property
    def tokens(self) -> List[str]:
        return list(self._tokens.keys())

    def lattice_distances(self) -> Dict[Tuple[str, str], float]:
        """Compute pairwise distances between all tokens on the lattice.

        Useful for understanding constraint density — tokens that are
        close together provide redundant constraints; tokens that are
        far apart cover different behavioral regions.

        Returns the Euclidean distance between Eisenstein integer
        coordinates in C. The Eisenstein lattice has hexagonal packing
        symmetry; the metric on the lattice IS the standard Euclidean
        distance.
        """
        result: Dict[Tuple[str, str], float] = {}
        ids = list(self._tokens.keys())
        for i, id_a in enumerate(ids):
            for id_b in ids[i + 1:]:
                ta = self._tokens[id_a].lattice_point
                tb = self._tokens[id_b].lattice_point
                # Euclidean distance in C — the natural metric on the lattice.
                # Earlier versions used sqrt(norm_squared) which is the same
                # numerically; using abs(complex) keeps it explicit.
                dist = abs(ta.complex - tb.complex)
                result[(id_a, id_b)] = dist
        return result

    def generate_token_id(self, entropy: float, hurst: float, prefix: str = "") -> str:
        """Generate a deterministic token ID from behavioral metrics.

        This creates a stable identifier for a behavioral region.
        Same metrics → same ID (deterministic naming).
        """
        # Map to nearest lattice point
        e_scaled = entropy * self._scale / 8.0  # Normalize entropy to 0-1 then scale
        h_scaled = hurst * self._scale
        z = complex(e_scaled, h_scaled)
        ei = eisenstein_round(z)
        raw = f"{prefix}({ei.a},{ei.b})"
        # Short hash for uniqueness
        h = hashlib.md5(raw.encode()).hexdigest()[:6]
        return f"{prefix}{ei.a}_{ei.b}_{h}" if prefix else f"t_{ei.a}_{ei.b}_{h}"

    def state(self) -> Dict:
        """Snapshot of the lattice state."""
        return {
            'total_tokens': len(self._tokens),
            'tokens': {
                tid: {
                    'lattice': (t.lattice_point.a, t.lattice_point.b),
                    'type': t.constraint_type.name,
                    'snap_radius': t.snap_radius,
                    'expected_e': t.expected_entropy,
                    'expected_h': t.expected_hurst,
                    'directive': (
                        (t.directive_target.a, t.directive_target.b)
                        if t.directive_target else None
                    ),
                    'coupled': t.coupled_with,
                }
                for tid, t in self._tokens.items()
            },
        }


# ─── Standard Maritime Token Presets ─────────────────────────────────

def create_maritime_lattice() -> TokenLattice:
    """Create a pre-populated token lattice for a fishing vessel.

    These tokens encode the standard behavioral regions a helm/nav
    agent operates in. Each maps to a specific (entropy, hurst) signature.
    """
    lattice = TokenLattice()

    # Steady-state helm operation — low entropy, high Hurst (trending/stable)
    lattice.register_token(
        "helm:steady",
        lattice_coord=(5, 1),
        constraint_type=ConstraintType.ELASTIC,
        snap_radius=0.25,
        expected_entropy=0.3,
        expected_hurst=0.7,
        metadata="Straight-line cruising, minimal corrections",
    )

    # Active course correction — moderate entropy, moderate Hurst
    lattice.register_token(
        "helm:correcting",
        lattice_coord=(3, 2),
        constraint_type=ConstraintType.PERIODIC,
        snap_radius=0.35,
        expected_entropy=1.5,
        expected_hurst=0.5,
        metadata="Active corrections, rudder working",
    )

    # Rough seas adaptation — high entropy, low Hurst (random/chaotic)
    lattice.register_token(
        "helm:rough",
        lattice_coord=(1, 4),
        constraint_type=ConstraintType.ELASTIC,
        snap_radius=0.4,
        expected_entropy=3.0,
        expected_hurst=0.35,
        metadata="Rough seas, high-frequency corrections needed",
    )

    # Harbor approach — directive, very tight constraints
    approach_target = EisensteinInteger(7, 0)
    lattice.register_token(
        "nav:harbor_approach",
        lattice_coord=(6, 0),
        constraint_type=ConstraintType.DIRECTIVE,
        snap_radius=0.15,
        expected_entropy=0.2,
        expected_hurst=0.8,
        directive_target=(7, 0),
        metadata="Precision navigation, T-minus harbor entry",
    )

    # Gear deployment — coupled with helm
    lattice.register_token(
        "deck:gear_deploy",
        lattice_coord=(2, 3),
        constraint_type=ConstraintType.COUPLED,
        snap_radius=0.3,
        expected_entropy=1.0,
        expected_hurst=0.6,
        coupled_with="helm:steady",
        metadata="Deploying gear, helm must hold steady",
    )

    # Emergency — any state, Executive takes over
    lattice.register_token(
        "system:emergency",
        lattice_coord=(0, 0),
        constraint_type=ConstraintType.RIGID,
        snap_radius=0.5,
        expected_entropy=5.0,
        expected_hurst=0.2,
        metadata="Emergency — Executive improvisation required",
    )

    return lattice
