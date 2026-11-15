"""Experiment-matrix tests: names, seeds and override keys stay consistent."""
from __future__ import annotations

from run_experiments import core_plan, extra_plan


def test_core_plan_has_three_seeds_and_all_ablations():
    cfg = {
        "ablations": {
            "no_influence_weight": {"loss_use_influence_weight": False},
            "no_ticker_fragment": {"use_ticker_fragment": False},
            "text_only": {"use_market_features": False},
            "frozen_encoder": {"freeze_layers": 12},
            "no_aux_loss": {"use_aux_class_loss": False},
            "no_noise_meme": {"drop_noise_meme": True},
            "no_class_probs": {"use_class_probs": False},
            "weighted_ce_only": {"focal_gamma": 0.0},
            "single_horizon_h5": {"horizons": [5]},
        }
    }
    plan = core_plan(cfg)
    seeds = [p for p in plan if p["kind"] == "seed"]
    ablations = [p for p in plan if p["kind"] == "ablation"]
    assert [p["seed"] for p in seeds] == [43, 44]
    assert len(ablations) == 9
    assert {p["name"] for p in ablations} == set(cfg["ablations"])
    assert all(p["run_id"].startswith("ablate_") for p in ablations)
    assert all(p["seed"] == 42 for p in ablations)


def test_extra_plan_covers_length_and_encoder_comparisons():
    plan = extra_plan()
    by_name = {p["name"]: p for p in plan}
    assert by_name["max_len_512"]["overrides"]["max_len"] == 512
    assert by_name["minilm_frozen"]["overrides"]["freeze_layers"] >= 6
    assert "MiniLM" in by_name["minilm_frozen"]["overrides"]["encoder_path"]
