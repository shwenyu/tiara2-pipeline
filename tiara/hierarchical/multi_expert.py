#!/usr/bin/env python3
"""Deterministic length-routed inference for the v2.3.2 + v2.4.0-B experts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tiara.hierarchical.data import fasta
from tiara.hierarchical.model import HierarchicalClassifier, probabilities
from tiara.hierarchical.schema import HierarchySchema
from tiara.src.transformations import TfidfWeighter
from tiara.training.featurize_cache import featurize_block


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expert_for_length(length_bp: int, threshold_bp: int) -> str:
    """Return the deterministic expert name; the boundary belongs to long."""
    return "short" if int(length_bp) < int(threshold_bp) else "long"


def load_expert(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    schema = HierarchySchema.from_dict(checkpoint["schema"])
    model = HierarchicalClassifier(**checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return checkpoint, schema, model


def classify(bundle, input_fasta, output, batch=512, device=None, min_len=1000, max_records=None):
    bundle_path = Path(bundle).resolve()
    manifest = json.loads(bundle_path.read_text())
    if manifest.get("format") != "tiara2-multi-expert-v1":
        raise ValueError("invalid multi-expert manifest")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else (bundle_path.parent / path).resolve()

    long_path = resolve(manifest["experts"]["long"]["checkpoint"])
    short_path = resolve(manifest["experts"]["short"]["checkpoint"])
    for name, path in (("long", long_path), ("short", short_path)):
        actual = sha256(path)
        expected = manifest["experts"][name]["sha256"]
        if actual != expected:
            raise ValueError(f"{name} checkpoint hash mismatch")
    long_checkpoint, long_schema, long_model = load_expert(long_path, dev)
    short_checkpoint, short_schema, short_model = load_expert(short_path, dev)
    if long_checkpoint["model"] != short_checkpoint["model"]:
        raise ValueError("expert model configurations differ")
    if long_schema.to_dict() != short_schema.to_dict():
        raise ValueError("expert schemas differ")
    tfidf = TfidfWeighter.load_params(str(resolve(manifest["tfidf"])))
    threshold = int(manifest["router"]["short_if_length_lt_bp"])
    thresholds = manifest.get("thresholds", {})
    temperatures = manifest.get("temperatures")

    def flush(records, writer):
        if not records:
            return
        short_items = []
        long_items = []
        for local_index, (header, sequence) in enumerate(records):
            target = short_items if expert_for_length(len(sequence), threshold) == "short" else long_items
            target.append((local_index, header, sequence))

        def predict(items, model):
            if not items:
                return {}
            matrix = featurize_block(
                [sequence for _, _, sequence in items],
                tfidf.k,
                np.asarray(tfidf.idfs, dtype=np.float32),
                4 ** tfidf.k,
            )
            with torch.inference_mode():
                scores = probabilities(model(torch.from_numpy(matrix).to(dev)), temperatures)
            result = {}
            for batch_index, (local_index, header, sequence) in enumerate(items):
                root_index = int(scores["root"][batch_index].argmax())
                root = long_schema.profile.root[root_index]
                root_probability = float(scores["root"][batch_index, root_index])
                leaf = root
                leaf_probability = root_probability
                branch = {"euk_nuclear": "euk", "prok": "prok", "organelle": "organelle"}.get(root)
                if branch:
                    leaf_index = int(scores[branch][batch_index].argmax())
                    leaf = long_schema.classes(branch)[leaf_index]
                    leaf_probability = float(scores[branch][batch_index, leaf_index])
                if root_probability < float(thresholds.get("root", 0)) or (
                    branch and leaf_probability < float(thresholds.get(branch, 0))
                ):
                    leaf = "unknown"
                result[local_index] = (
                    header,
                    len(sequence),
                    root,
                    leaf,
                    root_probability,
                    leaf_probability,
                )
            return result

        predictions = predict(short_items, short_model)
        predictions.update(predict(long_items, long_model))
        short_indices = {item[0] for item in short_items}
        for local_index in range(len(records)):
            header, length, root, leaf, root_p, leaf_p = predictions[local_index]
            writer.writerow([
                header,
                length,
                "short" if local_index in short_indices else "long",
                root,
                leaf,
                f"{root_p:.8f}",
                f"{leaf_p:.8f}",
            ])

    with Path(output).open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "record_id",
            "length_bp",
            "expert",
            "root",
            "leaf",
            "root_probability",
            "leaf_probability",
        ])
        records = []
        accepted = 0
        for header, sequence in fasta(Path(input_fasta)):
            if len(sequence) < min_len:
                continue
            records.append((header, sequence))
            accepted += 1
            if len(records) >= batch:
                flush(records, writer)
                records = []
            if max_records is not None and accepted >= max_records:
                break
        flush(records, writer)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--device")
    parser.add_argument("--min-len", type=int, default=1000)
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args(argv)
    classify(
        args.bundle,
        args.input,
        args.output,
        args.batch,
        args.device,
        args.min_len,
        args.max_records,
    )


if __name__ == "__main__":
    main()
