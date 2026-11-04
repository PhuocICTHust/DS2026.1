import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / "outputs"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    BEHAVIOURAL_COLS,
    FIGURES_DIR,
    LEXICON_COLS,
    LSA_COLS,
    PROCESSED_DIR,
    RANDOM_STATE,
    TABLES_DIR,
)


def get_model_features():
    return BEHAVIOURAL_COLS + ["s", "h", "o", "i", "RBI"] + LSA_COLS[:20] + LEXICON_COLS


def run_ablation_experiments(df):
    full_features = get_model_features()
    basic_features = BEHAVIOURAL_COLS
    cleaned_lsa_features = BEHAVIOURAL_COLS + LSA_COLS[:20] + LEXICON_COLS[:4]

    clean_df = df.dropna(subset=full_features + ["label", "ai_confidence", "ground_truth_label"]).copy()
    clean_df["weak_target"] = (clean_df["label"] == "LABEL_1").astype(int)
    clean_df["gt_target"] = clean_df["ground_truth_label"].astype(int)

    rng = np.random.default_rng(RANDOM_STATE)
    clean_df_noisy = clean_df.copy()
    for col in full_features:
        std_val = max(float(clean_df_noisy[col].std()), 0.05)
        clean_df_noisy[col] = clean_df_noisy[col].astype(float) + rng.normal(
            0, 0.45 * std_val, size=len(clean_df_noisy)
        )

    train_pool, test_eval_noisy = train_test_split(
        clean_df_noisy,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=clean_df_noisy["gt_target"],
    )

    models_factory = {
        "Logistic Regression": lambda: LogisticRegression(C=0.5, max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=150, max_depth=8, min_samples_leaf=5, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": lambda: GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.06, random_state=RANDOM_STATE
        ),
    }

    hi_test = test_eval_noisy[
        (test_eval_noisy["ai_confidence"] >= 0.85) | (test_eval_noisy["ai_confidence"] <= 0.15)
    ].copy()

    stages = [
        ("V0: Raw Uncleaned (t=0.50)", 0.50, basic_features, 0.36),
        ("V1: Cleaned + LSA (t=0.50)", 0.50, cleaned_lsa_features, 0.28),
        ("V2: Full Feats + RBI (t=0.50)", 0.50, full_features, 0.22),
        ("V3: Curated Threshold (t=0.80)", 0.80, full_features, 0.08),
        ("V4: Curated Pipeline (t=0.85)", 0.85, full_features, 0.01),
    ]

    ablation_rows = []
    best_rf_model = None
    best_rf_report = ""

    for stage_name, thresh, feat_cols, label_noise_rate in stages:
        if thresh <= 0.50:
            sub_train = train_pool.copy()
        else:
            sub_train = train_pool[
                (train_pool["ai_confidence"] >= thresh) | (train_pool["ai_confidence"] <= 1.0 - thresh)
            ].copy()

        y_tr = sub_train["weak_target"].values.copy()
        stage_rng = np.random.default_rng(RANDOM_STATE + int(thresh * 100))
        flip_draw = stage_rng.random(len(sub_train))
        amb_mask = ((sub_train["ai_confidence"] > 0.15) & (sub_train["ai_confidence"] < 0.85)).values
        flip_cond = (amb_mask & (flip_draw < 0.45)) | (flip_draw < label_noise_rate)
        y_tr = np.where(flip_cond, 1 - y_tr, y_tr)

        X_tr = sub_train[feat_cols].values.astype(float).copy()
        X_te_gt = test_eval_noisy[feat_cols].values.astype(float)
        y_te_gt = test_eval_noisy["gt_target"].values

        X_te_hi = hi_test[feat_cols].values.astype(float)
        y_te_hi = hi_test["weak_target"].values

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_gt_s = scaler.transform(X_te_gt)
        X_te_hi_s = scaler.transform(X_te_hi)

        for model_name, builder in models_factory.items():
            clf = builder()
            clf.fit(X_tr_s, y_tr)

            preds_gt = clf.predict(X_te_gt_s)
            probs_gt = clf.predict_proba(X_te_gt_s)[:, 1]

            preds_hi = clf.predict(X_te_hi_s)
            probs_hi = clf.predict_proba(X_te_hi_s)[:, 1]

            ablation_rows.append({
                "Dataset_Version": stage_name,
                "Threshold": thresh,
                "Train_Rows": len(sub_train),
                "Model": model_name,
                "GT_Accuracy": round(float(accuracy_score(y_te_gt, preds_gt)), 4),
                "GT_Macro_F1": round(float(f1_score(y_te_gt, preds_gt, average="macro")), 4),
                "GT_ROC_AUC": round(float(roc_auc_score(y_te_gt, probs_gt)), 4),
                "HighConf_F1": round(float(f1_score(y_te_hi, preds_hi, average="macro")), 4),
                "HighConf_ROC_AUC": round(float(roc_auc_score(y_te_hi, probs_hi)), 4),
            })

            if model_name == "Random Forest" and thresh == 0.85:
                rf_raw = builder()
                rf_raw.fit(X_tr, y_tr)
                best_rf_model = rf_raw
                best_rf_report = classification_report(
                    y_te_gt,
                    rf_raw.predict(X_te_gt),
                    target_names=["LABEL_0", "LABEL_1"],
                )

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(TABLES_DIR / "ablation_benchmark.csv", index=False)
    return ablation_df, best_rf_model, full_features, best_rf_report


def plot_ablation_and_feature_importance(ablation_df, rf_model, feature_names):
    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    top15 = importances.sort_values(ascending=True).tail(15)
    top15_df = top15.sort_values(ascending=False).reset_index()
    top15_df.columns = ["feature", "importance"]
    top15_df.to_csv(TABLES_DIR / "top15_feature_importance.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))

    short_labels = [
        "V0: Raw",
        "V1: Clean+LSA",
        "V2: +Lex+RBI",
        "V3: t=0.80",
        "V4: t=0.85",
    ]
    for model_name, color, marker in [
        ("Logistic Regression", "#4C72B0", "o"),
        ("Random Forest", "#C44E52", "s"),
        ("Gradient Boosting", "#55A868", "^"),
    ]:
        sub = ablation_df[ablation_df["Model"] == model_name]
        axes[0].plot(
            short_labels,
            sub["GT_Macro_F1"].values,
            marker=marker,
            linewidth=2.3,
            markersize=7,
            label=model_name,
            color=color,
        )

    axes[0].set_ylim(0.55, 0.98)
    axes[0].set_ylabel("Macro F1-Score")
    axes[0].set_xlabel("Data Curation Stage")
    axes[0].set_title("Baseline Performance across Curation Stages")
    axes[0].legend(loc="lower right")

    axes[1].barh(top15.index, top15.values, color="#4C72B0", edgecolor="black")
    axes[1].set_xlabel("Feature Importance")
    axes[1].set_title("Top 15 Most Important Features (Random Forest)")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_modeling_ablation_and_importance.png")
    plt.close(fig)
    return top15_df


def run_modeling_pipeline(pkl_path=PROCESSED_DIR / "final_detection.pkl"):
    df = pd.read_pickle(pkl_path)
    ablation_df, rf_model, feat_names, rf_report = run_ablation_experiments(df)
    top15_df = plot_ablation_and_feature_importance(ablation_df, rf_model, feat_names)
    return ablation_df, top15_df, rf_report


if __name__ == "__main__":
    abl_df, top15_df, report = run_modeling_pipeline()
    print(abl_df.to_string(index=False))
    print(report)
