import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import LEXICON_COLS, LSA_COLS, PROCESSED_DIR
from src.feature_extractor import (
    assemble_feature_table,
    extract_lexicon_features,
    extract_lsa_features,
)
from src.weak_labeling import apply_weak_supervision_teacher


class FeatureExtractorTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.directory = Path(self.temp_directory.name)

    def test_lexicon_distinguishes_constructive_and_hostile_phrases(self):
        comments = pd.DataFrame(
            {"cleaned_content": ["cảm ơn hợp lý khách quan", "cút ngu", None, ""]}
        )
        output = self.directory / "lexicon.csv"

        features = extract_lexicon_features(comments, out_path=output)

        self.assertEqual(list(features.columns), LEXICON_COLS)
        self.assertEqual(features.shape, (4, 8))
        self.assertGreater(features.loc[0, "lex_constructive"], 0)
        self.assertGreater(features.loc[0, "lex_politeness"], 0)
        self.assertEqual(features.loc[0, "lex_aggression"], 0)
        self.assertGreater(features.loc[1, "lex_aggression"], 0)
        self.assertGreater(features.loc[1, "lex_insult"], 0)
        self.assertTrue((features.iloc[2:] == 0).all().all())
        self.assertTrue(np.isfinite(features.to_numpy()).all())
        pd.testing.assert_frame_equal(pd.read_csv(output), features)

    def test_small_corpus_still_has_thirty_lsa_columns(self):
        comments = pd.DataFrame(
            {
                "cleaned_content": [
                    "cảm ơn bạn đã chia sẻ",
                    "cảm ơn bạn rất lịch sự",
                    "bình luận thiếu tôn trọng",
                    "bình luận thật hợp lý",
                ]
            }
        )

        features = extract_lsa_features(comments, out_path=self.directory / "lsa.csv")

        self.assertEqual(features.shape, (4, 30))
        self.assertEqual(list(features.columns), LSA_COLS)
        self.assertTrue(np.isfinite(features.to_numpy()).all())

    def test_real_comments_round_trip_and_feed_weak_labeling(self):
        comments = pd.read_csv(PROCESSED_DIR / "processed_comments.csv").head(96)
        comments_file = self.directory / "comments.csv"
        lexicon_file = self.directory / "lexicon.csv"
        lsa_file = self.directory / "lsa.csv"
        comments.to_csv(comments_file, index=False)

        aligned, table = assemble_feature_table(comments_file, lexicon_file, lsa_file)

        self.assertTrue(aligned)
        self.assertEqual(table.shape, (96, 57))
        self.assertTrue(table["comment_id"].equals(comments["comment_id"]))
        self.assertTrue(table.columns.is_unique)
        self.assertTrue(np.isfinite(table[LEXICON_COLS + LSA_COLS].to_numpy()).all())

        labeled = apply_weak_supervision_teacher(table)

        self.assertEqual(len(labeled), len(comments))
        self.assertTrue(labeled["comment_id"].equals(comments["comment_id"]))
        self.assertTrue(labeled["label"].isin(["LABEL_0", "LABEL_1"]).all())
        self.assertTrue(labeled["ai_confidence"].between(0, 1).all())
        _, reloaded = assemble_feature_table(comments_file, lexicon_file, lsa_file)
        pd.testing.assert_frame_equal(reloaded, table)

    def test_row_mismatch_is_rejected_before_concatenating_features(self):
        comments_file = self.directory / "comments.csv"
        lexicon_file = self.directory / "lexicon.csv"
        lsa_file = self.directory / "lsa.csv"
        pd.DataFrame({"comment_id": ["a", "b"]}).to_csv(comments_file, index=False)
        pd.DataFrame(np.zeros((1, 8)), columns=LEXICON_COLS).to_csv(lexicon_file, index=False)
        pd.DataFrame(np.zeros((2, 30)), columns=LSA_COLS).to_csv(lsa_file, index=False)

        with self.assertRaisesRegex(ValueError, "Row count mismatch"):
            assemble_feature_table(comments_file, lexicon_file, lsa_file)


if __name__ == "__main__":
    unittest.main()
