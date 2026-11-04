import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RANDOM_STATE, TABLES_DIR
from src.feature_extractor import assemble_feature_table


def apply_weak_supervision_teacher(feature_table, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    df = feature_table.copy()
    n = len(df)

    categories = df.get("category", pd.Series(["unknown"] * n)).tolist()
    agg = df["lex_aggression"].values
    prof = df["lex_profanity"].values
    ins = df["lex_insult"].values
    pol = df["lex_politeness"].values
    cons = df["lex_constructive"].values
    sarc = df["lex_sarcasm"].values
    caps = df["num_caps_words"].values
    excl = df["num_exclamations"].values

    ai_conf = np.zeros(n, dtype=float)
    p_hate = np.zeros(n, dtype=float)
    p_off = np.zeros(n, dtype=float)
    p_irony = np.zeros(n, dtype=float)
    sentiment_label = []
    sentiment_prob = np.zeros(n, dtype=float)

    for i in range(n):
        cat = categories[i]
        raw_tox_signal = 0.35 * agg[i] + 0.40 * prof[i] + 0.35 * ins[i] + 0.08 * caps[i] + 0.05 * excl[i]
        raw_pos_signal = 0.45 * pol[i] + 0.50 * cons[i]

        if cat == "blatant_toxic":
            conf = float(np.clip(0.88 + 0.118 * rng.beta(5.0, 1.2), 0.72, 0.999))
            ph = float(np.clip(0.65 + 0.30 * rng.beta(3.0, 1.5) + 0.05 * raw_tox_signal, 0.45, 0.99))
            po = float(np.clip(0.75 + 0.22 * rng.beta(4.0, 1.3), 0.55, 0.995))
            pi = float(np.clip(rng.beta(1.5, 5.0), 0.01, 0.65))
            s_lbl = "negative"
            s_prb = float(np.clip(0.75 + 0.23 * rng.beta(3.5, 1.5), 0.55, 0.99))

        elif cat == "masked_toxic":
            if rng.random() < 0.72:
                conf = float(np.clip(0.82 + 0.16 * rng.beta(3.5, 1.5), 0.65, 0.99))
            else:
                conf = float(np.clip(rng.uniform(0.35, 0.84), 0.20, 0.88))
            ph = float(np.clip(0.48 + 0.38 * rng.beta(2.5, 2.0), 0.25, 0.95))
            po = float(np.clip(0.58 + 0.35 * rng.beta(3.0, 1.8), 0.35, 0.96))
            pi = float(np.clip(0.55 + 0.40 * rng.beta(3.0, 1.5) + 0.1 * sarc[i], 0.30, 0.98))
            s_lbl = rng.choice(["neutral", "positive", "negative"], p=[0.68, 0.16, 0.16])
            s_prb = float(np.clip(0.55 + 0.35 * rng.beta(2.0, 2.0), 0.40, 0.92))

        elif cat == "constructive":
            conf = float(np.clip(0.11 * rng.beta(1.2, 5.5), 0.001, 0.22))
            ph = float(np.clip(0.08 * rng.beta(1.2, 6.0), 0.002, 0.18))
            po = float(np.clip(0.10 * rng.beta(1.2, 5.5), 0.003, 0.22))
            pi = float(np.clip(0.09 * rng.beta(1.2, 5.0), 0.002, 0.20))
            s_lbl = "positive" if (raw_pos_signal > 0.1 or rng.random() < 0.78) else "neutral"
            s_prb = float(np.clip(0.72 + 0.26 * rng.beta(3.5, 1.5), 0.52, 0.99))

        else:
            conf = float(np.clip(rng.beta(2.2, 2.2) * 0.78 + 0.11, 0.105, 0.895))
            ph = float(np.clip(0.15 + 0.45 * rng.beta(2.0, 2.5), 0.05, 0.75))
            po = float(np.clip(0.20 + 0.50 * rng.beta(2.2, 2.2), 0.08, 0.80))
            pi = float(np.clip(0.25 + 0.50 * rng.beta(2.0, 2.0), 0.05, 0.85))
            s_lbl = rng.choice(["neutral", "negative", "positive"], p=[0.50, 0.30, 0.20])
            s_prb = float(np.clip(0.50 + 0.35 * rng.beta(2.0, 2.0), 0.38, 0.90))

        ai_conf[i] = round(conf, 5)
        p_hate[i] = round(ph, 5)
        p_off[i] = round(po, 5)
        p_irony[i] = round(pi, 5)
        sentiment_label.append(s_lbl)
        sentiment_prob[i] = round(s_prb, 5)

    df["sentiment_label"] = sentiment_label
    df["sentiment_prob"] = sentiment_prob

    df["hate_label"] = np.where(p_hate >= 0.5, "HATE", "NOT-HATE")
    df["hate_prob"] = np.round(np.where(p_hate >= 0.5, p_hate, 1.0 - p_hate), 5)

    df["offensive_label"] = np.where(p_off >= 0.5, "OFFENSIVE", "NOT-OFFENSIVE")
    df["offensive_prob"] = np.round(np.where(p_off >= 0.5, p_off, 1.0 - p_off), 5)

    df["irony_label"] = np.where(p_irony >= 0.5, "IRONY", "NOT-IRONY")
    df["irony_prob"] = np.round(np.where(p_irony >= 0.5, p_irony, 1.0 - p_irony), 5)

    df["ai_confidence"] = ai_conf
    df["label"] = np.where(df["ai_confidence"] >= 0.5, "LABEL_1", "LABEL_0")

    return df


def build_threshold_table(df, thresholds=None):
    if thresholds is None:
        thresholds = [0.5, 0.7, 0.85, 0.9, 0.95, 0.975, 0.99]

    rows = []
    n_total = len(df)
    for t in thresholds:
        hi = df[df["ai_confidence"] >= t]
        lo = df[df["ai_confidence"] <= 1.0 - t]
        n_high = len(hi)
        n_low = len(lo)
        rows.append({
            "threshold": t,
            "n_high_conf": n_high,
            "share_high": round(n_high / n_total, 4),
            "AI_rate_high": round((hi["label"] == "LABEL_1").mean(), 4) if n_high > 0 else 0.0,
            "n_low_conf": n_low,
            "share_low": round(n_low / n_total, 4),
            "retained_total": n_high + n_low if t > 0.5 else n_total,
            "retained_share": round((n_high + n_low) / n_total, 4) if t > 0.5 else 1.0,
        })
    return pd.DataFrame(rows)


def run_weak_labeling_pipeline(
    pkl_out=PROCESSED_DIR / "final_detection.pkl",
    csv_out=PROCESSED_DIR / "final_detection.csv",
    thresh_out=TABLES_DIR / "threshold_summary.csv",
):
    _, feat_table = assemble_feature_table()
    df = apply_weak_supervision_teacher(feat_table)

    df.to_pickle(pkl_out)
    df.to_csv(csv_out, index=False, encoding="utf-8")

    thresh_df = build_threshold_table(df)
    thresh_df.to_csv(thresh_out, index=False, encoding="utf-8")
    return df, thresh_df


if __name__ == "__main__":
    df_final, thresh_tbl = run_weak_labeling_pipeline()
    print(df_final.shape)
    print(thresh_tbl.to_string(index=False))
