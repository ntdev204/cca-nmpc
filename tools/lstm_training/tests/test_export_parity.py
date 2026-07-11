"""Export tests: ONNX round-trips and matches PyTorch within tolerance."""

from tools.lstm_training.model import LSTMPredictor, LSTMConfig
from tools.lstm_training.export import (
    export_onnx, check_parity, load_model, INPUT_NAME, OUTPUT_NAME,
)
from tools.lstm_training.train import train, TrainConfig


def test_export_signature_and_parity(dataset_dir):
    model = LSTMPredictor(LSTMConfig(hidden_size=16, horizon=12))
    onnx_path = export_onnx(
        model, dataset_dir / "m.onnx", L=8, stats_path=dataset_dir / "stats.json"
    )
    assert onnx_path.exists()
    # stats copied next to the model for the runtime
    assert (dataset_dir / "normalization_stats.json").exists()
    max_diff = check_parity(model, onnx_path, L=8)
    assert max_diff < 1e-4


def test_names_are_stable():
    assert INPUT_NAME == "input"
    assert OUTPUT_NAME == "output"


def test_train_then_export_end_to_end(dataset_dir):
    cfg = TrainConfig(epochs=2, patience=2, hidden=16, batch_size=8, seed=0)
    summary = train(dataset_dir / "train.npz", dataset_dir / "val.npz",
                    dataset_dir / "stats.json", dataset_dir / "models", cfg)
    model = load_model(summary["checkpoint"])
    onnx_path = export_onnx(model, dataset_dir / "models" / "p.onnx", L=8,
                            stats_path=dataset_dir / "stats.json")
    assert check_parity(model, onnx_path, L=8) < 1e-4
