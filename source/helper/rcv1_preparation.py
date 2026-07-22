"""Prepare RCV1-103-H3 metadata without changing legacy WOS behavior.

The Hugging Face dataset stores human-readable label descriptions, while the
versioned RCV1 taxonomy uses topic codes.  This module joins those two
representations and materializes the two metadata artifacts that HGCLR needs:

* ``label_taxonomy.pkl``: official parent → child relations expressed as the
  dataset's stable integer label IDs;
* ``row_to_text_idx.pkl``: an explicit row-index → external-text-ID mapping
  for ranking/evaluation.  The two IDs are not interchangeable in RCV1.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAXONOMY_PATH = REPO_ROOT / "data" / "rcv1" / "rcv1.taxonomy"
DEFAULT_TOPIC_CODES_PATH = REPO_ROOT / "data" / "rcv1" / "rcv1_topic_codes.json"


def load_code_taxonomy(path: Path) -> dict[str, list[str]]:
    """Load the upstream tab-separated ``parent\tchild...`` RCV1 taxonomy."""
    taxonomy: dict[str, list[str]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = [part.strip() for part in line.split("\t") if part.strip()]
        if not parts:
            continue
        if len(parts) < 2:
            raise ValueError(f"{path}:{line_number} has no children")
        parent, *children = parts
        if parent in taxonomy:
            raise ValueError(f"duplicate taxonomy parent {parent!r} in {path}")
        taxonomy[parent] = children
    if "Root" not in taxonomy:
        raise ValueError(f"{path} does not define Root")
    return taxonomy


def _label_name_to_id(samples: Iterable[dict[str, Any]]) -> dict[str, int]:
    label_name_to_id: dict[str, int] = {}
    for row, sample in enumerate(samples):
        ids = sample["labels_ids"]
        names = sample["labels"]
        if len(ids) != len(names):
            raise ValueError(f"sample row {row} has mismatched labels_ids/labels")
        for label_id, label_name in zip(ids, names):
            existing = label_name_to_id.setdefault(label_name, label_id)
            if existing != label_id:
                raise ValueError(
                    f"label {label_name!r} maps to both {existing} and {label_id}"
                )
    return label_name_to_id


def build_label_taxonomy(
    samples: Iterable[dict[str, Any]],
    code_taxonomy: dict[str, list[str]],
    code_to_label: dict[str, str],
) -> dict[Any, list[int]]:
    """Translate the official RCV1 code tree into source dataset label IDs.

    Sample label order cannot be used for RCV1: a document may contain several
    sibling topics.  In contrast, WOS has exactly ``[parent, child]`` pairs,
    which is why the notebook's WOS rule is valid only there.
    """
    label_name_to_id = _label_name_to_id(samples)
    label_to_code = {label: code for code, label in code_to_label.items()}

    missing_codes = sorted(set(label_name_to_id) - set(label_to_code))
    if missing_codes:
        raise ValueError(f"dataset labels missing RCV1 topic codes: {missing_codes}")

    code_to_id: dict[str, int] = {}
    for label_name, label_id in label_name_to_id.items():
        code = label_to_code[label_name]
        if code in code_to_id:
            raise ValueError(f"two dataset labels resolve to RCV1 code {code!r}")
        code_to_id[code] = label_id

    taxonomy_codes = set(code_taxonomy)
    taxonomy_codes.update(child for children in code_taxonomy.values() for child in children)
    missing_taxonomy_codes = sorted(set(code_to_id) - taxonomy_codes)
    if missing_taxonomy_codes:
        raise ValueError(
            f"dataset topic codes absent from official taxonomy: {missing_taxonomy_codes}"
        )

    result: dict[Any, list[int]] = {"root": []}
    for parent_code, child_codes in code_taxonomy.items():
        if parent_code == "Root":
            destination = result["root"]
        elif parent_code not in code_to_id:
            # Match the notebook's documented fallback, while retaining a
            # valid graph if a future subset omits an internal category.
            destination = result["root"]
        else:
            destination = result.setdefault(code_to_id[parent_code], [])

        for child_code in child_codes:
            child_id = code_to_id.get(child_code)
            if child_id is None:
                continue
            if child_id not in destination:
                destination.append(child_id)

    attached = set(result["root"])
    for parent_id, child_ids in result.items():
        if parent_id != "root":
            attached.update(child_ids)
    expected_ids = set(label_name_to_id.values())
    if attached != expected_ids:
        raise ValueError(
            "official taxonomy did not attach every dataset label; "
            f"missing={sorted(expected_ids - attached)}, extra={sorted(attached - expected_ids)}"
        )
    return result


def build_slot_hierarchy(
    label_taxonomy: dict[Any, list[int]],
    source_id_to_label: dict[int, str],
    dynamic_label_dict: dict[str, int],
) -> dict[int, set[int]]:
    """Create HGCLR ``slot.pt`` edges from an official integer taxonomy.

    ``root`` is structural only and therefore never becomes a model node.
    The name-to-dynamic-ID lookup makes this safe even if future data files use
    source label IDs in a different order from the model's label dictionary.
    """
    hierarchy: defaultdict[int, set[int]] = defaultdict(set)
    for parent_source_id, child_source_ids in label_taxonomy.items():
        if parent_source_id == "root":
            continue
        try:
            parent_dynamic_id = dynamic_label_dict[source_id_to_label[parent_source_id]]
        except KeyError as error:
            raise ValueError(f"taxonomy parent cannot be resolved: {parent_source_id!r}") from error
        for child_source_id in child_source_ids:
            try:
                child_dynamic_id = dynamic_label_dict[source_id_to_label[child_source_id]]
            except KeyError as error:
                raise ValueError(f"taxonomy child cannot be resolved: {child_source_id!r}") from error
            hierarchy[parent_dynamic_id].add(child_dynamic_id)
    return dict(hierarchy)


def build_row_to_text_idx(samples: Iterable[dict[str, Any]]) -> list[int]:
    """Return the explicit positional-index → external-text-ID lookup table."""
    mapping: list[int] = []
    for row, sample in enumerate(samples):
        if sample["idx"] != row:
            raise ValueError(
                f"samples must be positional: row {row} declares idx={sample['idx']}"
            )
        mapping.append(sample["text_idx"])
    return mapping


def ranking_text_id(row_idx: int, row_to_text_idx: list[int] | None) -> int:
    """Use legacy row IDs unless an explicit external-ID mapping is provided."""
    return row_idx if row_to_text_idx is None else row_to_text_idx[row_idx]


def prepare_dataset(
    dataset_dir: Path,
    taxonomy_path: Path = DEFAULT_TAXONOMY_PATH,
    topic_codes_path: Path = DEFAULT_TOPIC_CODES_PATH,
) -> dict[str, int]:
    """Write RCV1-only metadata artifacts and return validation statistics."""
    with (dataset_dir / "samples.pkl").open("rb") as handle:
        samples = pickle.load(handle)
    if not isinstance(samples, list):
        raise ValueError("samples.pkl must contain a list")

    code_taxonomy = load_code_taxonomy(taxonomy_path)
    code_to_label = json.loads(topic_codes_path.read_text(encoding="utf-8"))
    label_taxonomy = build_label_taxonomy(samples, code_taxonomy, code_to_label)
    row_to_text_idx = build_row_to_text_idx(samples)

    with (dataset_dir / "label_taxonomy.pkl").open("wb") as handle:
        pickle.dump(label_taxonomy, handle, protocol=4)
    with (dataset_dir / "row_to_text_idx.pkl").open("wb") as handle:
        pickle.dump(row_to_text_idx, handle, protocol=4)

    edge_count = sum(len(children) for parent, children in label_taxonomy.items() if parent != "root")
    return {
        "samples": len(samples),
        "labels": len(_label_name_to_id(samples)),
        "root_labels": len(label_taxonomy["root"]),
        "non_root_edges": edge_count,
        "unique_text_ids": len(set(row_to_text_idx)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--taxonomy-path", type=Path, default=DEFAULT_TAXONOMY_PATH)
    parser.add_argument("--topic-codes-path", type=Path, default=DEFAULT_TOPIC_CODES_PATH)
    args = parser.parse_args()
    stats = prepare_dataset(args.data_dir, args.taxonomy_path, args.topic_codes_path)
    print("Prepared RCV1-103-H3 metadata:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
