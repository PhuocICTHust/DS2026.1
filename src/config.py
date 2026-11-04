from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_JSON_DIR = DATA_DIR / "raw_json"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

for d in [RAW_JSON_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

STRUCTURAL_COLS = [
    "commenter_id",
    "comment_id",
    "parent_id",
    "post_id",
    "topic",
    "likes",
    "comment_content",
    "cleaned_content",
]

BEHAVIOURAL_COLS = [
    "num_emojis",
    "num_text_emojis",
    "num_caps_words",
    "num_unicode_chars",
    "contains_media",
    "contains_link",
    "num_tagged_people",
    "used_slang",
    "num_exclamations",
]

AFFECT_PROB_COLS = [
    "sentiment_prob",
    "irony_prob",
    "hate_prob",
    "offensive_prob",
]

AFFECT_LABEL_COLS = [
    "sentiment_label",
    "irony_label",
    "hate_label",
    "offensive_label",
]

LEXICON_COLS = [
    "lex_aggression",
    "lex_profanity",
    "lex_insult",
    "lex_politeness",
    "lex_constructive",
    "lex_sarcasm",
    "lex_politics_social",
    "lex_emotional_intensity",
]

LABEL_COLS = [
    "label",
    "ai_confidence",
    "ground_truth_label",
]

N_LSA_DIMS = 30
LSA_COLS = [f"lsa_dim_{i}" for i in range(N_LSA_DIMS)]
