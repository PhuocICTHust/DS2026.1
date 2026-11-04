import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import LEXICON_COLS, LSA_COLS, N_LSA_DIMS, PROCESSED_DIR, RANDOM_STATE

PSYCH_LEXICONS = {
    "lex_aggression": [
        "cút", "câm", "chết", "giết", "đánh", "cắn", "xé", "đập", "tẩy chay",
        "biến", "xéo", "điên", "khùng", "hủy diệt", "tấn công", "chửi",
    ],
    "lex_profanity": [
        "đáng mắng", "không hề", "vô cùng lớn", "vô lý", "mẹ", "vãi",
        "cặn bã", "rác rưởi", "súc vật", "chó đàn", "bẩn", "tởm", "phế vật",
    ],
    "lex_insult": [
        "ngu", "ngốc", "dốt", "mất não", "ngáo", "vô học", "hạ đẳng",
        "mất dạy", "đạo đức giả", "rẻ tiền", "hãm", "ích kỷ", "hẹp hòi", "kém hiểu biết",
    ],
    "lex_politeness": [
        "cảm ơn", "xin", "dạ", "vâng", "chúc", "mong", "lịch sự",
        "văn minh", "tôn trọng", "bái phục", "ngưỡng mộ", "nhẹ nhàng",
    ],
    "lex_constructive": [
        "hợp lý", "khách quan", "đa chiều", "bình tĩnh", "lắng nghe", "giải pháp",
        "ý thức", "chất lượng", "chuẩn xác", "nỗ lực", "cải thiện", "kinh nghiệm", "tự học",
    ],
    "lex_sarcasm": [
        "thông minh nhất vũ trụ", "đỉnh cao", "bái phục", "giới hạn dưới",
        "tự hào khoe", "ngụy biện", "văn vở", "câu view", "câu tương tác", "đi vào lòng đất",
    ],
    "lex_politics_social": [
        "cộng đồng", "xã hội", "quy định", "xử phạt", "giáo dục", "báo",
        "thông tin", "tranh luận", "quan điểm", "phán xét", "dư luận",
    ],
    "lex_emotional_intensity": [
        "quá", "rất", "cực", "thật sự", "hoàn toàn", "nhất", "luôn",
        "kinh tởm", "hoang mang", "căng", "chưởng", "đét",
    ],
}


def extract_lexicon_features(comments_df, out_path=PROCESSED_DIR / "lexicon_features.csv"):
    texts = comments_df["cleaned_content"].fillna("").astype(str).tolist()
    rows = []
    for text in texts:
        words = text.split()
        denom = max(len(words), 1)
        row_scores = {}
        for col_name in LEXICON_COLS:
            phrases = PSYCH_LEXICONS[col_name]
            hits = sum(1 for p in phrases if p in text)
            row_scores[col_name] = round(hits / denom * 5.0, 4)
        rows.append(row_scores)

    lex_df = pd.DataFrame(rows, columns=LEXICON_COLS)
    lex_df.to_csv(out_path, index=False, encoding="utf-8")
    return lex_df


def extract_lsa_features(
    comments_df,
    n_components=N_LSA_DIMS,
    out_path=PROCESSED_DIR / "lsa_features.csv",
):
    texts = comments_df["cleaned_content"].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=1500,
        min_df=2,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    actual_components = min(n_components, tfidf_matrix.shape[0], tfidf_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=actual_components, random_state=RANDOM_STATE)
    lsa_matrix = svd.fit_transform(tfidf_matrix)

    if actual_components < n_components:
        pad = np.zeros((lsa_matrix.shape[0], n_components - actual_components))
        lsa_matrix = np.hstack([lsa_matrix, pad])

    lsa_df = pd.DataFrame(np.round(lsa_matrix, 5), columns=LSA_COLS)
    lsa_df.to_csv(out_path, index=False, encoding="utf-8")
    return lsa_df


def assemble_feature_table(
    comments_path=PROCESSED_DIR / "processed_comments.csv",
    lexicon_path=PROCESSED_DIR / "lexicon_features.csv",
    lsa_path=PROCESSED_DIR / "lsa_features.csv",
):
    comments = pd.read_csv(comments_path)
    if not lexicon_path.exists():
        extract_lexicon_features(comments, lexicon_path)
    if not lsa_path.exists():
        extract_lsa_features(comments, N_LSA_DIMS, lsa_path)

    lexicon_df = pd.read_csv(lexicon_path)
    lsa_df = pd.read_csv(lsa_path)

    aligned = (len(comments) == len(lexicon_df) == len(lsa_df))
    if not aligned:
        raise ValueError("Row count mismatch")

    feature_table = pd.concat([comments, lexicon_df, lsa_df], axis=1)
    return aligned, feature_table


if __name__ == "__main__":
    comments_df = pd.read_csv(PROCESSED_DIR / "processed_comments.csv")
    extract_lexicon_features(comments_df)
    extract_lsa_features(comments_df)
    aligned, feat_table = assemble_feature_table()
    print(aligned, feat_table.shape)
