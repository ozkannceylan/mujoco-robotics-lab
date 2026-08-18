"""Lab 9 — M2 tests: the ACT policy and the instruction bank.

The failures these catch do not raise. A model whose output ignores its
instruction still trains, still produces a falling loss curve, and only shows up
as a mediocre success rate hours later; a checkpoint that cannot denormalise its
own predictions loads cleanly and is wrong by a scale factor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from act_policy import ACTPolicy, load_checkpoint, save_checkpoint  # noqa: E402
from dataset import NormStats  # noqa: E402
from lab9_common import CHUNK_SIZE, IMAGE_SIZE, STATE_DIM  # noqa: E402
from observations import ACTION_DIMS  # noqa: E402
from text_encoder import TEXT_EMBED_DIM, InstructionBank  # noqa: E402


def _model(**kwargs) -> ACTPolicy:
    kwargs.setdefault("pretrained_backbone", False)
    return ACTPolicy(**kwargs)


def _batch(model: ACTPolicy, size: int = 2) -> tuple[dict, torch.Tensor, torch.Tensor]:
    images = {
        camera: torch.rand(size, 3, model.image_size, model.image_size)
        for camera in model.cameras
    }
    state = torch.randn(size, model.state_dim)
    if model.conditioning == "text":
        instruction = torch.randn(size, TEXT_EMBED_DIM)
    else:
        instruction = torch.randint(0, 4, (size,))
    return images, state, instruction


class TestShapes:
    @pytest.mark.parametrize("head", sorted(ACTION_DIMS))
    def test_output_shape_matches_the_head(self, head):
        model = _model(action_head=head)
        images, state, instruction = _batch(model, size=3)
        output = model(images, state, instruction)
        assert output.shape == (3, CHUNK_SIZE, ACTION_DIMS[head])

    def test_token_count_is_derived_not_hardcoded(self):
        """Upstream hardcodes 49 tokens, which is only right at 224 px."""
        assert _model(image_size=128).tokens_per_camera == 16
        assert _model(image_size=224).tokens_per_camera == 49
        assert _model(image_size=IMAGE_SIZE).tokens_per_camera == 16

    def test_both_cameras_contribute_tokens(self):
        model = _model()
        images, state, instruction = _batch(model)
        tokens = model._vision_tokens(images)
        assert tokens.shape[1] == model.tokens_per_camera * len(model.cameras)

    def test_backbone_is_frozen_except_layer4(self):
        model = _model()
        trainable = {
            name for name, p in model.backbone.named_parameters() if p.requires_grad
        }
        assert trainable, "layer4 must be fine-tuned"
        assert all(name.startswith("7.") for name in trainable), (
            f"only layer4 (index 7) should train, got {sorted(trainable)[:3]}"
        )


class TestConditioning:
    def test_instruction_changes_the_output(self):
        """A necessary condition for the instruction-swap test to mean anything.

        If two different instructions on an identical observation produced the
        same action, the policy could not follow instructions no matter how well
        it trained, and every success rate would be about the scene instead.
        """
        model = _model().eval()
        images, state, _ = _batch(model, size=1)
        first = model.predict(images, state, torch.randn(1, TEXT_EMBED_DIM))
        second = model.predict(images, state, torch.randn(1, TEXT_EMBED_DIM))
        assert (first - second).abs().max() > 1e-4

    def test_same_instruction_is_deterministic(self):
        model = _model().eval()
        images, state, instruction = _batch(model, size=1)
        first = model.predict(images, state, instruction)
        second = model.predict(images, state, instruction)
        assert torch.allclose(first, second, atol=1e-6)

    def test_task_id_conditioning_builds_and_runs(self):
        model = _model(conditioning="task_id", num_tasks=4)
        images, state, instruction = _batch(model, size=2)
        assert model(images, state, instruction).shape[0] == 2

    def test_rejects_unknown_configuration(self):
        with pytest.raises(ValueError):
            _model(conditioning="telepathy")
        with pytest.raises(ValueError):
            _model(action_head="wrist_wiggle")


class TestNormalisation:
    def _stats(self, model: ACTPolicy) -> NormStats:
        rng = np.random.default_rng(0)
        return NormStats(
            state_mean=rng.normal(size=model.state_dim).astype(np.float32),
            state_scale=(rng.random(model.state_dim) + 0.5).astype(np.float32),
            action_mean=rng.normal(size=model.action_dim).astype(np.float32),
            action_scale=(rng.random(model.action_dim) + 0.5).astype(np.float32),
        )

    def test_round_trip(self):
        model = _model()
        model.set_norm_stats(self._stats(model))
        action = torch.randn(4, CHUNK_SIZE, model.action_dim)
        assert torch.allclose(
            model.denormalize_action(model.normalize_action(action)), action, atol=1e-5
        )

    def test_stats_are_buffers_so_they_travel_with_the_weights(self):
        model = _model()
        model.set_norm_stats(self._stats(model))
        keys = model.state_dict().keys()
        for name in ("state_mean", "state_scale", "action_mean", "action_scale"):
            assert name in keys, f"{name} would not survive a checkpoint"

    def test_zero_variance_dimension_does_not_produce_inf(self):
        values = np.stack([np.array([1.0, 2.0, 3.0])] * 16)
        stats = NormStats.fit(values, values)
        assert np.isfinite(stats.state_scale).all()
        assert (stats.state_scale > 0).all()


class TestCheckpoint:
    def test_round_trip_preserves_predictions(self, tmp_path):
        model = _model().eval()
        bank = InstructionBank({"pick up the red cup": np.ones(TEXT_EMBED_DIM, np.float32)})
        rng = np.random.default_rng(0)
        stats = NormStats(
            state_mean=rng.normal(size=STATE_DIM).astype(np.float32),
            state_scale=(rng.random(STATE_DIM) + 0.5).astype(np.float32),
            action_mean=rng.normal(size=model.action_dim).astype(np.float32),
            action_scale=(rng.random(model.action_dim) + 0.5).astype(np.float32),
        )
        model.set_norm_stats(stats)
        images, state, instruction = _batch(model, size=1)
        before = model.predict(images, state, instruction)

        path = tmp_path / "policy.pt"
        save_checkpoint(path, model, bank, stats, extra={"note": "test"})
        restored, restored_bank, restored_stats, extra = load_checkpoint(path)
        after = restored.predict(images, state, instruction)

        assert torch.allclose(before, after, atol=1e-6)
        assert "pick up the red cup" in restored_bank
        assert np.allclose(restored_stats.action_scale, stats.action_scale)
        assert extra["note"] == "test"

    def test_checkpoint_is_self_contained(self, tmp_path):
        """Evaluation must not need the dataset or the network to run."""
        model = _model()
        bank = InstructionBank({"walk to the red cup": np.zeros(TEXT_EMBED_DIM, np.float32)})
        stats = NormStats(
            state_mean=np.zeros(STATE_DIM, np.float32),
            state_scale=np.ones(STATE_DIM, np.float32),
            action_mean=np.zeros(model.action_dim, np.float32),
            action_scale=np.ones(model.action_dim, np.float32),
        )
        path = tmp_path / "policy.pt"
        save_checkpoint(path, model, bank, stats)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("state_dict", "config", "instruction_bank", "norm_stats"):
            assert key in payload


class TestLearning:
    def test_overfits_a_single_batch(self):
        """The optimisation path works end to end.

        Not a quality claim — a model that cannot memorise eight samples has a
        wiring bug, and finding that here costs a minute instead of an epoch.

        Scored against the best **constant** predictor rather than the initial
        loss: with N(0,1) targets the constant predictor already scores 0.76, so
        a ratio to the starting loss cannot tell "memorised the batch" from
        "learned its mean". Learning rate 1e-3 — at 3e-3 this transformer
        destabilises and plateaus exactly at the constant predictor, which reads
        like an architecture bug and is not one.
        """
        torch.manual_seed(0)
        model = _model()
        images, state, instruction = _batch(model, size=8)
        target = torch.randn(8, CHUNK_SIZE, model.action_dim)
        baseline = float((target - target.mean(dim=0, keepdim=True)).abs().mean())
        optimiser = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        for _ in range(120):
            loss = (model(images, state, instruction) - target).abs().mean()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimiser.step()
        model.eval()
        with torch.no_grad():
            final = float((model(images, state, instruction) - target).abs().mean())
        assert final < 0.5 * baseline, (
            f"final {final:.3f} vs constant-predictor {baseline:.3f}: the model "
            "is not distinguishing the samples"
        )

    def test_gradients_reach_every_trainable_block(self):
        model = _model()
        images, state, instruction = _batch(model, size=2)
        model(images, state, instruction).abs().mean().backward()
        missing = [
            name for name, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())
        ]
        assert not missing, f"no finite gradient for {missing[:4]}"


class TestInstructionBank:
    def test_lookup_and_round_trip(self):
        bank = InstructionBank({
            "walk to the red cup": np.arange(TEXT_EMBED_DIM, dtype=np.float32),
        })
        restored = InstructionBank.from_dict(bank.to_dict())
        assert np.allclose(
            restored.get("walk to the red cup"), bank.get("walk to the red cup")
        )

    def test_separation_reports_the_closest_pair(self):
        one = np.zeros(TEXT_EMBED_DIM, np.float32)
        one[0] = 1.0
        two = np.zeros(TEXT_EMBED_DIM, np.float32)
        two[1] = 1.0
        bank = InstructionBank({"a": one, "b": two, "c": one.copy()})
        summary = bank.separation()
        assert summary["count"] == 3
        assert summary["closest_similarity"] == pytest.approx(1.0, abs=1e-5)
        assert set(summary["closest_pair"]) == {"a", "c"}
