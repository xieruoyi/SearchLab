"""Hand-computed binary NDCG examples and ranking tie checks."""
import math
import unittest

from experiment import ndcg, ranked, tokens


class MetricTests(unittest.TestCase):
    def test_perfect(self):
        self.assertEqual(ndcg(["a", "b"], {"a", "b"}), 1)

    def test_discount(self):
        self.assertAlmostEqual(ndcg(["x", "a"], {"a"}), 1 / math.log2(3))

    def test_full_qrels_denominator(self):
        self.assertAlmostEqual(ndcg(["a", "x"], {"a", "b"}), 1 / (1 + 1 / math.log2(3)))

    def test_unjudged_and_cutoff(self):
        self.assertEqual(ndcg([str(i) for i in range(11)], {"10"}), 0)

    def test_missing_candidate_relevance(self):
        self.assertEqual(ndcg(["x"], {"a"}), 0)

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            ndcg(["a", "a"], {"a"})

    def test_ties(self):
        self.assertEqual(ranked([1, 2, 2], ["a", "b", "c"], 3), [1, 2, 0])

    def test_normalization(self):
        self.assertEqual(tokens("ＡBC, café!"), ["abc", "café"])


if __name__ == "__main__":
    unittest.main()
