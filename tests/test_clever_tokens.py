"""Tests for Clever Tokens."""

import pytest
from snapkit.clever_tokens import (
    TokenLattice, CleverToken, ConstraintType,
    create_maritime_lattice,
)
from snapkit.eisenstein import EisensteinInteger


class TestCleverToken:
    def test_creation(self):
        ei = EisensteinInteger(3, 1)
        token = CleverToken(
            identifier="test",
            lattice_point=ei,
            constraint_type=ConstraintType.ELASTIC,
            snap_radius=0.3,
            expected_entropy=0.5,
            expected_hurst=0.7,
        )
        assert token.identifier == "test"
        assert token.lattice_point == ei
        assert token.complex_position == ei.complex
        assert token.norm > 0

    def test_deviation_calculation(self):
        token = CleverToken(
            identifier="test",
            lattice_point=EisensteinInteger(0, 0),
            constraint_type=ConstraintType.ELASTIC,
            snap_radius=0.2,
            expected_entropy=1.0,
            expected_hurst=0.5,
        )
        # Zero deviation
        assert token.deviation_of(1.0, 0.5) == 0.0
        # Some deviation
        dev = token.deviation_of(2.0, 0.5)
        assert dev > 0

    def test_harmony_check(self):
        token = CleverToken(
            identifier="test",
            lattice_point=EisensteinInteger(0, 0),
            constraint_type=ConstraintType.ELASTIC,
            snap_radius=0.2,
            expected_entropy=1.0,
            expected_hurst=0.5,
        )
        assert token.is_in_harmony(1.0, 0.5)
        assert token.is_in_harmony(1.1, 0.52)
        assert not token.is_in_harmony(5.0, 0.1)

    def test_render(self):
        token = CleverToken(
            identifier="helm:steady",
            lattice_point=EisensteinInteger(5, 1),
            constraint_type=ConstraintType.ELASTIC,
            snap_radius=0.25,
            expected_entropy=0.3,
            expected_hurst=0.7,
            metadata="Cruising",
        )
        rendered = token.render()
        assert "[helm:steady" in rendered
        assert "lat=(5,1)" in rendered
        assert "elastic" in rendered
        assert "Cruising" in rendered


class TestTokenLattice:
    def test_register_token(self):
        lattice = TokenLattice()
        token = lattice.register_token(
            "test",
            lattice_coord=(3, 1),
            expected_entropy=1.0,
            expected_hurst=0.5,
        )
        assert "test" in lattice.tokens

    def test_duplicate_raises(self):
        lattice = TokenLattice()
        lattice.register_token("dup", (0, 0))
        with pytest.raises(ValueError):
            lattice.register_token("dup", (1, 1))

    def test_snap_to_nearest(self):
        lattice = TokenLattice()
        lattice.register_token(
            "harmony", (5, 1),
            expected_entropy=0.5, expected_hurst=0.7,
        )
        lattice.register_token(
            "chaos", (0, 4),
            expected_entropy=4.0, expected_hurst=0.3,
        )

        # Low entropy, high hurst → should snap to harmony
        token, dev, on = lattice.snap(0.6, 0.65)
        assert token.identifier == "harmony"
        assert on

        # High entropy, low hurst → should snap to chaos
        token, dev, on = lattice.snap(4.5, 0.25)
        assert token.identifier == "chaos"

    def test_snap_empty_lattice(self):
        lattice = TokenLattice()
        token, dev, on = lattice.snap(1.0, 0.5)
        assert token is None
        assert not on

    def test_render_prompt(self):
        lattice = TokenLattice()
        lattice.register_token("a", (1, 0), expected_entropy=0.5, expected_hurst=0.6)
        lattice.register_token("b", (2, 1), expected_entropy=1.0, expected_hurst=0.5)
        prompt = lattice.render_prompt(["a", "b"])
        assert "Constraint Lattice" in prompt
        assert "[a" in prompt
        assert "[b" in prompt

    def test_render_prompt_missing_token(self):
        lattice = TokenLattice()
        lattice.register_token("a", (1, 0))
        prompt = lattice.render_prompt(["a", "nonexistent"])
        assert "WARNING" in prompt

    def test_lattice_distances(self):
        lattice = TokenLattice()
        lattice.register_token("a", (0, 0))
        lattice.register_token("b", (3, 1))
        lattice.register_token("c", (6, 2))
        dists = lattice.lattice_distances()
        assert ("a", "b") in dists
        assert ("a", "c") in dists
        assert dists[("a", "b")] < dists[("a", "c")]

    def test_generate_token_id(self):
        lattice = TokenLattice()
        id1 = lattice.generate_token_id(1.0, 0.5)
        id2 = lattice.generate_token_id(1.0, 0.5)
        id3 = lattice.generate_token_id(4.0, 0.3)
        assert id1 == id2  # Deterministic
        assert id1 != id3  # Different metrics → different ID

    def test_state_snapshot(self):
        lattice = TokenLattice()
        lattice.register_token("x", (1, 1))
        state = lattice.state()
        assert state["total_tokens"] == 1
        assert "x" in state["tokens"]


class TestMaritimeLattice:
    def test_create_maritime_lattice(self):
        lattice = create_maritime_lattice()
        assert "helm:steady" in lattice.tokens
        assert "nav:harbor_approach" in lattice.tokens
        assert "system:emergency" in lattice.tokens

    def test_maritime_snap_scenarios(self):
        lattice = create_maritime_lattice()

        # Steady cruising
        token, dev, on = lattice.snap(0.3, 0.72)
        assert token.identifier == "helm:steady"
        assert on

        # Rough seas
        token, dev, on = lattice.snap(3.2, 0.33)
        assert token.identifier == "helm:rough"
        assert on

        # Emergency
        token, dev, on = lattice.snap(5.5, 0.15)
        assert token.identifier == "system:emergency"

    def test_maritime_render(self):
        lattice = create_maritime_lattice()
        prompt = lattice.render_prompt(["helm:steady", "deck:gear_deploy"])
        assert "helm:steady" in prompt
        assert "deck:gear_deploy" in prompt
        assert "couple=helm:steady" in prompt

    def test_coupled_token(self):
        lattice = create_maritime_lattice()
        gear = lattice.get_token("deck:gear_deploy")
        assert gear.coupled_with == "helm:steady"
