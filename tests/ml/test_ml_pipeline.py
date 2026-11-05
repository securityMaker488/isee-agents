import os
import sys
import tempfile
import unittest
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ML_DATASETS_DIR = os.path.join(ROOT_DIR, "ml", "datasets")
ML_EXPERIMENTS_DIR = os.path.join(ROOT_DIR, "ml", "experiments")

for p in (ML_DATASETS_DIR, ML_EXPERIMENTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from prepare_dataset import anonymize_pii, prepare_dataset  # noqa: E402
from train_lora import (  # noqa: E402
    compute_lora_parameter_budget,
    run_verifiable_low_rank_adaptation,
)
from train_qlora import quantize_dequantize_nf4_blockwise  # noqa: E402


class TestMLTrainingPipeline(unittest.TestCase):
    def test_01_dataset_preparation_deduplication_and_zero_leakage(self) -> None:
        raw_jsonl = os.path.join(ML_DATASETS_DIR, "synthetic_agents.jsonl")
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = prepare_dataset(raw_jsonl, tmp_dir)
            self.assertEqual(manifest["data_leakage_detected"], 0)
            self.assertGreaterEqual(manifest["duplicates_removed"], 1)
            self.assertGreaterEqual(manifest["pii_redactions_applied"], 2)
            self.assertGreaterEqual(manifest["splits"]["train"], 1)
            self.assertGreaterEqual(manifest["splits"]["val"], 1)
            self.assertGreaterEqual(manifest["splits"]["test"], 1)

    def test_02_pii_anonymization_masks_sensitive_tokens(self) -> None:
        raw = "Contato: user@dominio.com.br, CPF 123.456.789-00 e chave sk-abcdef1234567890abcd."
        cleaned, count = anonymize_pii(raw)
        self.assertEqual(count, 3)
        self.assertIn("[EMAIL_REDACTED]", cleaned)
        self.assertIn("[CPF_REDACTED]", cleaned)
        self.assertIn("[SECRET_REDACTED]", cleaned)

    def test_03_lora_low_rank_adaptation_and_frozen_weights(self) -> None:
        budget = compute_lora_parameter_budget(
            d_model=4096, k_model=4096, num_layers=32, rank=16, lora_alpha=32
        )
        self.assertGreater(budget["parameter_reduction_percentage"], 99.0)

        splits_dir = os.path.join(ML_DATASETS_DIR, "splits")
        with tempfile.TemporaryDirectory() as tmp_out:
            report = run_verifiable_low_rank_adaptation(
                train_jsonl_path=os.path.join(splits_dir, "train.jsonl"),
                val_jsonl_path=os.path.join(splits_dir, "val.jsonl"),
                output_dir=tmp_out,
                rank=16,
                lora_alpha=32,
                epochs=40,
            )
            self.assertTrue(report["frozen_base_weights_verified_unchanged"])
            self.assertLess(report["final_train_loss"], report["initial_train_loss"])
            self.assertTrue(
                os.path.exists(os.path.join(tmp_out, "lora_adapter_weights.npz"))
            )

    def test_04_qlora_nf4_and_double_quantization_compression(self) -> None:
        rng = np.random.default_rng(123)
        weights = rng.normal(0.0, 0.02, size=(256, 256)).astype(np.float32)
        stats = quantize_dequantize_nf4_blockwise(weights, block_size=64)
        self.assertEqual(stats["nf4_levels"], 16)
        self.assertTrue(stats["double_quantization_enabled"])
        self.assertGreater(stats["compression_vs_fp16_pct"], 70.0)
        self.assertLess(stats["reconstruction_rmse"], 0.005)


if __name__ == "__main__":
    unittest.main()
