from src.crawler import flatten_trees_to_csv, generate_seed_social_trees
from src.text_cleaner import demo_clean, process_all_tables
from src.feature_extractor import assemble_feature_table, extract_lexicon_features, extract_lsa_features
from src.weak_labeling import run_weak_labeling_pipeline
from src.eda_analysis import run_all_eda
from src.baseline_models import run_modeling_pipeline


def main():
    generate_seed_social_trees()
    flatten_trees_to_csv()
    demo_clean()
    comments_df, _ = process_all_tables()
    extract_lexicon_features(comments_df)
    extract_lsa_features(comments_df)
    assemble_feature_table()
    run_weak_labeling_pipeline()
    run_all_eda()
    abl_df, _, _ = run_modeling_pipeline()
    print(abl_df.to_string(index=False))


if __name__ == "__main__":
    main()
