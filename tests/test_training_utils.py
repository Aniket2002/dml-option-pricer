import pandas as pd
import pytest
import torch

from data.bs_data_generator import generate_synthetic_data
from train.train_model import (
    TrainingConfig,
    build_model,
    frame_to_tensors,
    split_frame,
)


def test_split_is_deterministic_and_disjoint():
    frame = generate_synthetic_data(100, seed=11)
    first = split_frame(
        frame,
        seed=5,
        validation_fraction=0.15,
        test_fraction=0.15,
    )
    second = split_frame(
        frame,
        seed=5,
        validation_fraction=0.15,
        test_fraction=0.15,
    )

    assert [len(part) for part in first] == [70, 15, 15]
    assert all(left.equals(right) for left, right in zip(first, second, strict=True))


def test_model_scaler_is_fitted_from_training_features():
    frame = generate_synthetic_data(50, seed=3)
    features, *_ = frame_to_tensors(frame)
    model = build_model(features, hidden_dims=(16, 8))

    assert torch.allclose(model.input_mean, features.mean(dim=0))
    assert torch.allclose(
        model.input_scale,
        features.std(dim=0, unbiased=False).clamp_min(1e-6),
    )


def test_training_config_rejects_invalid_split():
    config = TrainingConfig(validation_fraction=0.6, test_fraction=0.5)
    with pytest.raises(ValueError, match="sum to less than one"):
        config.validate()


def test_frame_to_tensors_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        frame_to_tensors(pd.DataFrame({"S": [100.0]}))


def test_training_pipeline_smoke(tmp_path):
    from train.train_model import train_pipeline

    metadata = train_pipeline(
        TrainingConfig(
            samples=200,
            epochs=1,
            batch_size=64,
            learning_rate=1e-3,
            patience=1,
            hidden_dims=(8,),
        ),
        compare_baseline=True,
        output_dir=tmp_path,
    )

    assert (tmp_path / "dml_option_pricer.pt").exists()
    assert (tmp_path / "metrics.json").exists()
    benchmark_path = tmp_path / "benchmark_summary.md"
    assert benchmark_path.exists()
    benchmark_text = benchmark_path.read_text(encoding="utf-8")
    assert "Price-only baseline" in benchmark_text
    assert "held-out synthetic" in benchmark_text
    assert metadata["metrics"]["test_rows"] > 0
    assert metadata["baseline_metrics"] is not None
