from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.dom_graph_builder import DOMGraphBuilder


class DOMGraphBuilderTests(unittest.TestCase):
    def test_dom_graph_builder_outputs_nodes_edges_candidates(self) -> None:
        graph = DOMGraphBuilder().build(_observation())
        self.assertGreaterEqual(len(graph["node_features"]), 2)
        self.assertGreaterEqual(len(graph["edge_index"]), 2)
        self.assertEqual(1, len(graph["candidate_node_indices"]))

    def test_dom_graph_builder_handles_missing_dom_fields(self) -> None:
        graph = DOMGraphBuilder().build({"candidate_elements": [{"text": "Submit"}]})
        self.assertGreaterEqual(graph["graph_stats"]["node_count"], 1)
        self.assertEqual(1, graph["graph_stats"]["candidate_count"])

    def test_dom_graph_builder_ignores_bug_label_metadata(self) -> None:
        base = _observation()
        labeled = _observation()
        labeled["candidate_elements"][0].update(
            {
                "data_bug_id": "site001-bug01",
                "known_bug": True,
                "catalog_bug_id_matches": ["site001-bug01"],
                "ground_truth": "x",
                "label": "x",
            }
        )
        self.assertEqual(DOMGraphBuilder().build(base)["node_features"], DOMGraphBuilder().build(labeled)["node_features"])

    def test_dom_graph_builder_stable_shapes(self) -> None:
        graph1 = DOMGraphBuilder().build(_observation())
        graph2 = DOMGraphBuilder().build(_observation())
        self.assertEqual(len(graph1["node_features"]), len(graph2["node_features"]))
        self.assertEqual(len(graph1["node_features"][0]), len(graph2["node_features"][0]))

    def test_dom_graph_builder_graph_stats_written(self) -> None:
        stats = DOMGraphBuilder().build(_observation())["graph_stats"]
        self.assertIn("node_count", stats)
        self.assertIn("edge_count", stats)
        self.assertIn("candidate_count", stats)
        self.assertIn("matched_candidate_count", stats)


def _observation() -> dict:
    return {
        "dom_object": {
            "tag": "div",
            "role": "main",
            "children": [
                {
                    "tag": "button",
                    "role": "button",
                    "text": "Submit",
                    "visible": True,
                    "enabled": True,
                    "clickable": True,
                    "bbox": {"x": 10, "y": 20, "width": 80, "height": 30},
                }
            ],
        },
        "candidate_elements": [
            {
                "text": "Submit",
                "role": "button",
                "tag": "button",
                "visible": True,
                "enabled": True,
                "clickable": True,
                "bbox": {"x": 10, "y": 20, "width": 80, "height": 30},
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
