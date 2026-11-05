"""
I.See Agents — Base vs. Specialized Adapter & RAG Inference Comparison
Developer: Júlio César | I.See Technologies
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from typing import Any, Dict, List
import numpy as np


def run_inference_comparison(
    adapter_dir: str,
    test_jsonl_path: str,
    output_path: str,
) -> Dict[str, Any]:
    weights_path = os.path.join(adapter_dir, "lora_adapter_weights.npz")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Adapter weights not found at {weights_path}. Run train_lora.py first."
        )

    data = np.load(weights_path)
    lora_A = data["lora_A"]
    lora_B = data["lora_B"]
    scaling = float(data["scaling"][0])
    dim = lora_A.shape[1]

    rng = np.random.default_rng(42)
    W0 = rng.normal(0.0, 0.05, size=(dim, dim)).astype(np.float32)

    comparisons: List[Dict[str, Any]] = []
    with open(test_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            x_vec = np.zeros(dim, dtype=np.float32)
            y_vec = np.zeros(dim, dtype=np.float32)
            for tok in (row.get("domain", "") + " " + row["instruction"]).lower().split():
                idx = int.from_bytes(hashlib.sha256(tok.encode("utf-8")).digest()[:4], "big") % dim
                x_vec[idx] += 1.0
            for tok in (row.get("domain", "") + " " + row["output"]).lower().split():
                idx = int.from_bytes(hashlib.sha256(tok.encode("utf-8")).digest()[:4], "big") % dim
                y_vec[idx] += 1.0
            x_vec /= np.linalg.norm(x_vec) + 1e-6
            y_vec /= np.linalg.norm(y_vec) + 1e-6

            base_pred = x_vec @ W0.T
            adapted_pred = base_pred + scaling * ((x_vec @ lora_A.T) @ lora_B.T)

            base_mse = float(np.mean((base_pred - y_vec) ** 2))
            adapted_mse = float(np.mean((adapted_pred - y_vec) ** 2))

            comparisons.append(
                {
                    "id": row.get("id"),
                    "domain": row.get("domain"),
                    "instruction": row.get("instruction"),
                    "base_model_mse": round(base_mse, 6),
                    "lora_adapted_mse": round(adapted_mse, 6),
                    "error_reduction_pct": round(
                        ((base_mse - adapted_mse) / max(base_mse, 1e-8)) * 100.0, 2
                    ),
                }
            )

    summary = {
        "compared_at": datetime.now(timezone.utc).isoformat(),
        "adapter_directory": adapter_dir,
        "test_examples_evaluated": len(comparisons),
        "comparisons": comparisons,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    adapter_dir = os.path.join(base_dir, "outputs", "isee-synthetic-lora-adapter")
    test_jsonl = os.path.join(base_dir, "datasets", "splits", "test.jsonl")
    out_file = os.path.join(base_dir, "outputs", "inference_comparison_report.json")
    report = run_inference_comparison(adapter_dir, test_jsonl, out_file)
    print(json.dumps(report, indent=2, ensure_ascii=False))
