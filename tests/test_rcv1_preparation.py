import unittest

from source.helper.rcv1_preparation import (
    build_label_taxonomy,
    build_row_to_text_idx,
    build_slot_hierarchy,
    ranking_text_id,
)


class RCV1PreparationTests(unittest.TestCase):
    def test_build_label_taxonomy_uses_the_reference_tree_not_label_order(self):
        samples = [
            {
                "labels_ids": [0, 1, 2],
                "labels": ["TOP", "SIBLING_A", "SIBLING_B"],
            }
        ]
        code_taxonomy = {
            "Root": ["T"],
            "T": ["A", "B"],
        }
        code_to_label = {
            "T": "TOP",
            "A": "SIBLING_A",
            "B": "SIBLING_B",
        }

        self.assertEqual(
            build_label_taxonomy(samples, code_taxonomy, code_to_label),
            {"root": [0], 0: [1, 2]},
        )

    def test_build_row_to_text_idx_preserves_duplicate_document_ids(self):
        samples = [
            {"idx": 0, "text_idx": 7},
            {"idx": 1, "text_idx": 7},
            {"idx": 2, "text_idx": 19},
        ]

        self.assertEqual(build_row_to_text_idx(samples), [7, 7, 19])

    def test_build_slot_hierarchy_uses_official_edges_and_omits_root(self):
        taxonomy = {"root": [0], 0: [1, 2]}
        source_id_to_label = {0: "TOP", 1: "SIBLING_A", 2: "SIBLING_B"}
        dynamic_label_dict = {"TOP": 0, "SIBLING_A": 1, "SIBLING_B": 2}

        self.assertEqual(
            build_slot_hierarchy(taxonomy, source_id_to_label, dynamic_label_dict),
            {0: {1, 2}},
        )

    def test_ranking_text_id_keeps_legacy_row_index_without_a_mapping(self):
        self.assertEqual(ranking_text_id(4, None), 4)

    def test_ranking_text_id_uses_text_idx_when_mapping_is_supplied(self):
        self.assertEqual(ranking_text_id(1, [7, 13]), 13)


if __name__ == "__main__":
    unittest.main()
