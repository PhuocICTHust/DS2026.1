import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RANDOM_STATE, TABLES_DIR
from src.feature_extractor import assemble_feature_table


def apply_weak_supervision_teacher(feature_table, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    df = feature_table.copy()
    n = len(df)

    agg = df["lex_aggression"].values
    prof = df["lex_profanity"].values
    ins = df["lex_insult"].values
    pol = df["lex_politeness"].values
    cons = df["lex_constructive"].values
    sarc = df["lex_sarcasm"].values
    caps = df["num_caps_words"].values
    excl = df["num_exclamations"].values
    slang = df["used_slang"].values
    lsa0 = df["lsa_dim_0"].values
    lsa1 = df["lsa_dim_1"].values

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -25, 25)))

    lex_signal = 0.45 * prof + 0.40 * ins + 0.35 * agg + 0.18 * sarc
    behav_signal = 0.14 * np.log1p(caps) + 0.10 * np.log1p(excl) + 0.18 * slang
    sem_signal = 0.30 * lsa0 - 0.20 * lsa1
    civ_signal = 0.45 * pol + 0.45 * cons

    combined_tox = lex_signal + behav_signal + sem_signal - civ_signal + rng.normal(0, 0.25, n)
    p_tox = sigmoid(3.2 * (combined_tox - np.median(combined_tox)))

    ai_conf = np.zeros(n, dtype=float)
    p_off = np.zeros(n, dtype=float)
    p_hate = np.zeros(n, dtype=float)
    p_irony = np.zeros(n, dtype=float)
    s = np.zeros(n, dtype=float)

    for i in range(n):
        pt = p_tox[i]
        if pt > 0.54:
            base_c = 0.82 + 0.16 * (pt - 0.54) / 0.46 + float(rng.normal(0, 0.02))
            c = np.clip(base_c, 0.75, 0.995)
            po = np.clip(0.68 + 0.28 * pt + float(rng.normal(0, 0.03)), 0.55, 0.99)
            ph = np.clip(0.48 + 0.45 * sigmoid(2.0 * prof[i] + 2.0 * agg[i]) + float(rng.normal(0, 0.03)), 0.30, 0.98)
            pi = np.clip(0.20 + 0.60 * sigmoid(3.0 * sarc[i]) + float(rng.normal(0, 0.04)), 0.08, 0.92)
            s_val = np.clip(-0.50 - 0.45 * pt + float(rng.normal(0, 0.08)), -0.98, -0.12)
        elif pt < 0.46:
            base_c = 0.18 * pt / 0.46 + float(rng.normal(0, 0.02))
            c = np.clip(base_c, 0.005, 0.22)
            po = np.clip(0.20 * pt / 0.46 + float(rng.normal(0, 0.02)), 0.01, 0.28)
            ph = np.clip(0.12 * pt / 0.46 + float(rng.normal(0, 0.02)), 0.005, 0.20)
            pi = np.clip(0.10 + float(rng.beta(1.5, 5.0) * 0.25), 0.01, 0.35)
            s_val = np.clip(0.40 + 0.52 * sigmoid(3.5 * civ_signal[i]) + float(rng.normal(0, 0.08)), 0.12, 0.98)
        else:
            base_c = 0.20 + 0.60 * (pt - 0.46) / 0.08 + float(rng.normal(0, 0.05))
            c = np.clip(base_c, 0.15, 0.85)
            po = np.clip(0.35 + 0.30 * c + float(rng.normal(0, 0.04)), 0.15, 0.75)
            ph = np.clip(0.20 + 0.25 * c + float(rng.normal(0, 0.04)), 0.08, 0.60)
            pi = np.clip(0.30 + 0.45 * sigmoid(3.0 * sarc[i]) + float(rng.normal(0, 0.05)), 0.10, 0.85)
            s_val = np.clip(float(rng.normal(0, 0.24)), -0.45, 0.45)

        ai_conf[i] = round(float(c), 5)
        p_off[i] = round(float(po), 5)
        p_hate[i] = round(float(ph), 5)
        p_irony[i] = round(float(pi), 5)
        s[i] = round(float(s_val), 5)

    df["ai_confidence"] = ai_conf
    df["label"] = np.where(df["ai_confidence"] >= 0.5, "LABEL_1", "LABEL_0")

    df["sentiment_label"] = np.where(s > 0.10, "positive", np.where(s < -0.10, "negative", "neutral"))
    df["sentiment_prob"] = np.round(np.clip(np.abs(s), 0.05, 0.99), 5)

    df["hate_label"] = np.where(p_hate >= 0.5, "HATE", "NOT-HATE")
    df["hate_prob"] = np.round(np.where(p_hate >= 0.5, p_hate, 1.0 - p_hate), 5)

    df["offensive_label"] = np.where(p_off >= 0.5, "OFFENSIVE", "NOT-OFFENSIVE")
    df["offensive_prob"] = np.round(np.where(p_off >= 0.5, p_off, 1.0 - p_off), 5)

    df["irony_label"] = np.where(p_irony >= 0.5, "IRONY", "NOT-IRONY")
    df["irony_prob"] = np.round(np.where(p_irony >= 0.5, p_irony, 1.0 - p_irony), 5)

    p_human_pos = sigmoid(3.4 * (ai_conf - 0.5) + rng.normal(0, 0.40, n))
    df["ground_truth_label"] = (rng.random(n) < p_human_pos).astype(int)

    return df


def build_threshold_table(df, thresholds=None):
    if thresholds is None:
        thresholds = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]

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
