import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / "outputs"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import (
    AFFECT_PROB_COLS,
    BEHAVIOURAL_COLS,
    FIGURES_DIR,
    LEXICON_COLS,
    LSA_COLS,
    PROCESSED_DIR,
    RANDOM_STATE,
    STRUCTURAL_COLS,
    TABLES_DIR,
)

plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def run_section_4_1_integrity(df):
    tok = df["cleaned_content"].dropna().astype(str).str.split().str.len()
    overview = pd.DataFrame(
        [
            ("Records", f"{len(df):,}"),
            ("Unique users", f"{df['commenter_id'].nunique():,}"),
            ("Features", f"{df.shape[1]:,}"),
            (
                "Cells missing (feature cols)",
                f"{int(df[STRUCTURAL_COLS + BEHAVIOURAL_COLS + AFFECT_PROB_COLS].isna().sum().sum()):,}",
            ),
            ("Empty cleaned comments", f"{int((df['cleaned_content'].fillna('') == '').sum()):,}"),
            ("Mean tokens per comment", f"{tok.mean():.2f}"),
            ("Median tokens per comment", f"{int(tok.median())}"),
        ],
        columns=["metric", "value"],
    )
    overview.to_csv(TABLES_DIR / "eda_overview.csv", index=False)

    miss = df.isna().mean()
    miss = miss[miss > 0].sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    if len(miss) > 0:
        axes[0].bar(miss.index.astype(str), miss.values, color="#DD8452", edgecolor="black", width=0.45)
    axes[0].set_ylabel("Fraction missing")
    axes[0].set_title("Missingness is structural (parent_id)")

    axes[1].hist(tok, bins=40, range=(0, 80), color="#4C72B0", edgecolor="white", alpha=0.9)
    axes[1].set_xlim(0, 80)
    axes[1].set_xlabel("Tokens per comment")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Comment length distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_integrity_and_length.png")
    plt.close(fig)
    return overview


def run_section_4_2_weak_signal(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    conf = df["ai_confidence"].dropna().values
    axes[0].hist(conf, bins=45, density=True, color="#4C72B0", alpha=0.65, edgecolor="white", label="Histogram")
    axes[0].axvspan(0.15, 0.85, color="#DD8452", alpha=0.18, label="Ambiguous band (0.15, 0.85)")
    axes[0].axvline(0.15, color="#DD8452", linestyle="--", linewidth=1.2)
    axes[0].axvline(0.85, color="#DD8452", linestyle="--", linewidth=1.2)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("ai_confidence")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Distribution of ai_confidence")
    axes[0].legend(loc="upper center")

    c0 = df.loc[df["label"] == "LABEL_0", "ai_confidence"].dropna().values
    c1 = df.loc[df["label"] == "LABEL_1", "ai_confidence"].dropna().values
    axes[1].hist(c0, bins=30, density=True, alpha=0.6, color="#55A868", label="LABEL_0")
    axes[1].hist(c1, bins=30, density=True, alpha=0.6, color="#C44E52", label="LABEL_1")
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("ai_confidence")
    axes[1].set_ylabel("Density")
    axes[1].set_title("ai_confidence split by weak label")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_weak_supervision_confidence.png")
    plt.close(fig)


def bucket_comment_count(n):
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    if n == 3:
        return "3"
    if n == 4:
        return "4"
    if 5 <= n <= 10:
        return "5-10"
    return ">10"


def run_section_4_3_user_activity(df):
    counts = df["commenter_id"].value_counts()
    bucket_order = ["1", "2", "3", "4", "5-10", ">10"]
    bins = counts.map(bucket_comment_count).value_counts().reindex(bucket_order, fill_value=0)

    labelled = df.dropna(subset=["commenter_id", "label"]).copy()
    per_user = labelled.groupby("commenter_id")["label"].apply(list)
    two = per_user[per_user.map(len) == 2]
    agree2 = two.map(lambda lst: lst[0] == lst[1])
    agreement_rate = float(agree2.mean()) if len(agree2) > 0 else 0.0

    labelled["is_ai"] = (labelled["label"] == "LABEL_1").astype(int)
    grp = labelled.groupby("commenter_id").agg(
        total_comments=("is_ai", "count"),
        ai_comments=("is_ai", "sum"),
    )
    more2 = grp[grp["total_comments"] > 2].copy()
    more2["ai_share"] = more2["ai_comments"] / more2["total_comments"]

    top15 = grp.sort_values("total_comments", ascending=False).head(15).copy()
    top15["ai_share"] = top15["ai_comments"] / top15["total_comments"]
    top15 = top15.sort_values("ai_share", ascending=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(bins.index, bins.values, color="#4C72B0", edgecolor="black")
    axes[0].set_xlabel("Comments per user bucket")
    axes[0].set_ylabel("Number of users")
    axes[0].set_title(f"User activity buckets (2-cmt agreement={agreement_rate:.1%})")

    axes[1].hist(more2["ai_share"], bins=20, color="#8172B2", edgecolor="white")
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("ai_share (users with >2 comments)")
    axes[1].set_ylabel("Number of users")
    axes[1].set_title("Per-user label share distribution")

    bar_colors = ["#C44E52" if s >= 0.5 else "#55A868" for s in top15["ai_share"]]
    axes[2].barh(top15.index.astype(str), top15["ai_share"], color=bar_colors, edgecolor="black")
    axes[2].set_xlim(0, 1)
    axes[2].set_xlabel("ai_share")
    axes[2].set_title("Top 15 most active users")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_user_activity_and_stability.png")
    plt.close(fig)

    stats = {
        "n_users": int(len(counts)),
        "single_comment_user_share": round(float((counts == 1).mean()), 4),
        "users_with_2_comments": int(len(two)),
        "two_comment_agreement_rate": round(agreement_rate, 4),
    }
    pd.DataFrame([stats]).to_csv(TABLES_DIR / "user_stability_stats.csv", index=False)
    return stats


def positive_class_probability(prob, label, positive_label):
    return pd.Series(
        np.where(label == positive_label, prob, 1.0 - prob),
        index=prob.index,
    )


def enrich_affective_and_rbi(df):
    df = df.copy()
    sign = np.where(
        df["sentiment_label"] == "positive",
        1.0,
        np.where(df["sentiment_label"] == "negative", -1.0, 0.0),
    )
    df["s"] = np.round(sign * df["sentiment_prob"], 5)
    df["h"] = np.round(positive_class_probability(df["hate_prob"], df["hate_label"], "HATE"), 5)
    df["o"] = np.round(positive_class_probability(df["offensive_prob"], df["offensive_label"], "OFFENSIVE"), 5)
    df["i"] = np.round(positive_class_probability(df["irony_prob"], df["irony_label"], "IRONY"), 5)

    r = df["o"]
    rbi = (r + df["o"] + df["h"]) * (1.0 - np.abs(df["s"]))
    df["RBI"] = np.round(pd.Series(rbi).replace([np.inf, -np.inf], np.nan).fillna(0.0), 5)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    grp_means = df.groupby("label")[["s", "h", "o", "i", "RBI"]].mean()
    x = np.arange(5)
    width = 0.35
    axes[0].bar(x - width / 2, grp_means.loc["LABEL_0"], width, label="LABEL_0", color="#55A868")
    axes[0].bar(x + width / 2, grp_means.loc["LABEL_1"], width, label="LABEL_1", color="#C44E52")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["s", "h", "o", "i", "RBI"])
    axes[0].set_ylabel("Mean score")
    axes[0].set_title("Affective & RBI means by weak label")
    axes[0].legend()

    corr_cols = ["s", "h", "o", "i", "RBI", "ai_confidence"]
    corr = df[corr_cols].corr()
    im = axes[1].imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_xticks(range(len(corr_cols)))
    axes[1].set_yticks(range(len(corr_cols)))
    axes[1].set_xticklabels(corr_cols, rotation=30, ha="right")
    axes[1].set_yticklabels(corr_cols)
    for r_i in range(len(corr_cols)):
        for c_i in range(len(corr_cols)):
            axes[1].text(c_i, r_i, f"{corr.values[r_i, c_i]:.2f}", ha="center", va="center", fontsize=8)
    axes[1].set_title("Affective feature correlations")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    sample = df.sample(min(2500, len(df)), random_state=RANDOM_STATE).copy()
    sample["toxicity"] = (sample["h"] + sample["o"]) / 2.0
    is_tox = (sample["label"] == "LABEL_1").astype(int)
    sc = axes[2].scatter(
        sample["s"],
        sample["toxicity"],
        c=is_tox,
        cmap="coolwarm",
        alpha=0.35,
        s=14,
    )
    axes[2].axvline(0, color="gray", linestyle="--", linewidth=1)
    axes[2].set_xlabel("signed sentiment s")
    axes[2].set_ylabel("toxicity (h + o) / 2")
    axes[2].set_title("Toxicity vs sentiment (masked toxicity)")
    fig.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.04, label="1 = LABEL_1")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_affective_and_rbi.png")
    plt.close(fig)
    return df


def pca_2d(frame, feature_cols, n=3000, seed=RANDOM_STATE):
    clean = frame.dropna(subset=feature_cols + ["label"]).copy()
    if len(clean) > n:
        clean = clean.sample(n, random_state=seed)
    X = StandardScaler().fit_transform(clean[feature_cols].values)
    coords = PCA(n_components=2, random_state=seed).fit_transform(X)
    return clean, coords


def clustering_diagnostics(df, feature_cols, name, thresholds=(0.0, 0.7, 0.9), n=2500, seed=RANDOM_STATE):
    rows = []
    base = df.dropna(subset=feature_cols + ["label", "ai_confidence"]).copy()
    for t in thresholds:
        if t == 0.0:
            sub = base
        else:
            sub = base[(base["ai_confidence"] >= t) | (base["ai_confidence"] <= 1.0 - t)]
        if len(sub) > n:
            sub = sub.sample(n, random_state=seed)

        X = StandardScaler().fit_transform(sub[feature_cols].values)
        y_true = (sub["label"] == "LABEL_1").astype(int).values

        km = KMeans(n_clusters=2, n_init=10, random_state=seed)
        preds = km.fit_predict(X)

        ari = adjusted_rand_score(y_true, preds)
        sil = silhouette_score(X, preds)
        acc1 = float((preds == y_true).mean())
        acc2 = float(((1 - preds) == y_true).mean())
        best_acc = max(acc1, acc2)

        rows.append({
            "space": name,
            "threshold": t,
            "n_samples": len(sub),
            "ARI": round(float(ari), 4),
            "silhouette": round(float(sil), 4),
            "best_mapped_acc": round(best_acc, 4),
        })
    return pd.DataFrame(rows)


def run_section_4_6_embeddings(df):
    sub_lex, coords_lex = pca_2d(df, LEXICON_COLS)
    sub_lsa, coords_lsa = pca_2d(df, LSA_COLS)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    c_lex = (sub_lex["label"] == "LABEL_1").astype(int)
    axes[0].scatter(coords_lex[:, 0], coords_lex[:, 1], c=c_lex, cmap="coolwarm", alpha=0.35, s=12)
    axes[0].set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")
    axes[0].set_title("Lexicon Space (2-D PCA)")

    c_lsa = (sub_lsa["label"] == "LABEL_1").astype(int)
    sc = axes[1].scatter(coords_lsa[:, 0], coords_lsa[:, 1], c=c_lsa, cmap="coolwarm", alpha=0.35, s=12)
    axes[1].set_xlabel("PC 1")
    axes[1].set_ylabel("PC 2")
    axes[1].set_title("TF-IDF / LSA Space (2-D PCA)")
    fig.colorbar(sc, ax=axes[1], label="1 = LABEL_1")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_embedding_pca_spaces.png")
    plt.close(fig)

    diag_lex = clustering_diagnostics(df, LEXICON_COLS, "PsychLexicon")
    diag_lsa = clustering_diagnostics(df, LSA_COLS, "TFIDF_LSA")
    diagnostics = pd.concat([diag_lex, diag_lsa], ignore_index=True)
    diagnostics.to_csv(TABLES_DIR / "clustering_diagnostics.csv", index=False)
    return diagnostics


def run_all_eda(pkl_path=PROCESSED_DIR / "final_detection.pkl"):
    df = pd.read_pickle(pkl_path)
    run_section_4_1_integrity(df)
    run_section_4_2_weak_signal(df)
    run_section_4_3_user_activity(df)
    df = enrich_affective_and_rbi(df)
    diagnostics = run_section_4_6_embeddings(df)

    df.to_pickle(pkl_path)
    df.to_csv(PROCESSED_DIR / "final_detection.csv", index=False, encoding="utf-8")
    return df, diagnostics


if __name__ == "__main__":
    df_enriched, diag_tbl = run_all_eda()
    print(diag_tbl.to_string(index=False))
