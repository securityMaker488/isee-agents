"""
I.See Agents — Dataset Preparation, Cleaning, Deduplication, PII Anonymization & Split Pipeline
Developer: Júlio César | I.See Technologies
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import unicodedata
from typing import Any, Dict, List, Set, Tuple

PII_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    ("cpf", re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"), "[CPF_REDACTED]"),
    ("cnpj", re.compile(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b"), "[CNPJ_REDACTED]"),
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[EMAIL_REDACTED]",
    ),
    (
        "phone",
        re.compile(r"\b(?:\+55\s?)?(?:\(?\d{2}\)?\s?)?\d{4,5}-\d{4}\b"),
        "[PHONE_REDACTED]",
    ),
    (
        "api_key",
        re.compile(r"\b(?:sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{16,})\b"),
        "[SECRET_REDACTED]",
    ),
]


def normalize_text(value: str) -> str:
    norm = unicodedata.normalize("NFC", value or "")
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


def anonymize_pii(value: str) -> Tuple[str, int]:
    redactions = 0
    result = value
    for _, pattern, replacement in PII_PATTERNS:
        matches = len(pattern.findall(result))
        if matches > 0:
            redactions += matches
            result = pattern.sub(replacement, result)
    return result, redactions


def jaccard_similarity(text_a: str, text_b: str) -> float:
    set_a = set(re.findall(r"\w+", text_a.lower()))
    set_b = set(re.findall(r"\w+", text_b.lower()))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def prepare_dataset(
    input_path: str,
    output_dir: str,
    min_output_chars: int = 40,
    near_dup_threshold: float = 0.92,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    raw_records: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))

    cleaned_records: List[Dict[str, Any]] = []
    seen_hashes: Set[str] = set()
    duplicates_removed = 0
    low_quality_removed = 0
    total_pii_redactions = 0

    for item in raw_records:
        instr, r1 = anonymize_pii(normalize_text(item.get("instruction", "")))
        inp, r2 = anonymize_pii(normalize_text(item.get("input", "")))
        out, r3 = anonymize_pii(normalize_text(item.get("output", "")))
        total_pii_redactions += r1 + r2 + r3

        if len(instr) < 10 or len(out) < min_output_chars:
            low_quality_removed += 1
            continue

        fingerprint_source = f"{instr.lower()}||{inp.lower()}||{out.lower()}"
        digest = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            duplicates_removed += 1
            continue

        # Check near-duplicate via Jaccard token similarity
        is_near_dup = False
        for existing in cleaned_records:
            sim = jaccard_similarity(
                f"{instr} {out}",
                f"{existing['instruction']} {existing['output']}",
            )
            if sim >= near_dup_threshold:
                is_near_dup = True
                break
        if is_near_dup:
            duplicates_removed += 1
            continue

        seen_hashes.add(digest)

        messages = item.get("messages", [])
        clean_messages = []
        for m in messages:
            c_text, r_msg = anonymize_pii(normalize_text(m.get("content", "")))
            total_pii_redactions += r_msg
            clean_messages.append({"role": m.get("role", "user"), "content": c_text})

        cleaned_records.append(
            {
                "id": item.get("id"),
                "domain": item.get("domain", "general"),
                "instruction": instr,
                "input": inp,
                "output": out,
                "messages": clean_messages,
                "sha256": digest,
            }
        )

    # Deterministic split: 70% train, 15% val, 15% test (ensuring >= 1 item in each split for small datasets)
    n_total = len(cleaned_records)
    n_test = max(1, int(round(n_total * 0.15))) if n_total >= 3 else 0
    n_val = max(1, int(round(n_total * 0.15))) if n_total >= 3 else 0
    n_train = max(1, n_total - n_val - n_test)

    train_set = cleaned_records[:n_train]
    val_set = cleaned_records[n_train : n_train + n_val]
    test_set = cleaned_records[n_train + n_val :]

    # Verify zero data leakage across splits
    train_hashes = {r["sha256"] for r in train_set}
    val_hashes = {r["sha256"] for r in val_set}
    test_hashes = {r["sha256"] for r in test_set}
    leakage_count = (
        len(train_hashes & val_hashes)
        + len(train_hashes & test_hashes)
        + len(val_hashes & test_hashes)
    )
    if leakage_count > 0:
        raise RuntimeError(f"Data leakage detected across splits: {leakage_count}")

    for split_name, split_data in (
        ("train.jsonl", train_set),
        ("val.jsonl", val_set),
        ("test.jsonl", test_set),
    ):
        split_path = os.path.join(output_dir, split_name)
        with open(split_path, "w", encoding="utf-8") as sf:
            for row in split_data:
                sf.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "dataset_version": "2.0.0-synthetic",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source_file": os.path.basename(input_path),
        "raw_examples": len(raw_records),
        "cleaned_examples": n_total,
        "duplicates_removed": duplicates_removed,
        "low_quality_removed": low_quality_removed,
        "pii_redactions_applied": total_pii_redactions,
        "data_leakage_detected": leakage_count,
        "splits": {
            "train": len(train_set),
            "val": len(val_set),
            "test": len(test_set),
        },
    }

    manifest_path = os.path.join(output_dir, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare, anonymize, deduplicate and split synthetic dataset."
    )
    parser.add_argument(
        "--input",
        default=os.path.join(os.path.dirname(__file__), "synthetic_agents.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "splits"),
    )
    args = parser.parse_args()
    result = prepare_dataset(args.input, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
