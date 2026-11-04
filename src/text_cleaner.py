import re
import unicodedata
import pandas as pd

from src.config import BEHAVIOURAL_COLS, PROCESSED_DIR

URL_RE = re.compile(r"https?://\S+")
MEDIA_URL_RE = re.compile(r"https?://\S*(?:pic|photo|img|video|media|youtu\.be|tiktok)\S*", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
REPEAT_RE = re.compile(r"(.)\1{2,}")
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
NUMBER_RE = re.compile(r"\b\d+\b")
SPACE_RE = re.compile(r"\s+")

UNICODE_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

TEXT_EMOJIS = [":)", ":(", "=))", ":))", ":v", "^^", "-_-", ":3", "<3", ":/"]

CONTRACTIONS = {
    "can't": "can not",
    "won't": "will not",
    "don't": "do not",
    "i'm": "i am",
    "you're": "you are",
    "it's": "it is",
    "that's": "that is",
    "k0": "không",
    "hong": "không",
    "hông": "không",
}

SLANG_MAP = {
    "ko": "không",
    "k": "không",
    "khum": "không",
    "đc": "được",
    "dc": "được",
    "vcl": "vô cùng lớn",
    "vl": "vô lý",
    "đm": "đáng mắng",
    "dm": "đáng mắng",
    "đéo": "không hề",
    "vãi": "rất",
    "omg": "oh my god",
    "lol": "laugh out loud",
    "btw": "by the way",
    "tbh": "to be honest",
    "idk": "i do not know",
    "mn": "mọi người",
    "ae": "anh em",
    "ng": "người",
    "bt": "bình thường",
    "cx": "cũng",
    "vs": "với",
    "mik": "mình",
    "mk": "mình",
}

ABBREVIATIONS = {
    "vn": "việt nam",
    "hn": "hà nội",
    "hcm": "hồ chí minh",
    "đn": "đà nẵng",
    "ca": "california",
    "ny": "new york",
    "us": "united states",
    "uk": "united kingdom",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "is", "are", "was", "were", "to", "of", "in",
    "thì", "là", "mà", "ở", "những", "các", "cái", "chiếc", "nhé", "nha", "ạ", "á", "ơi",
}

SAMPLE_RAW_TEXT = "@sonpham-yf1vd về hỏi xem ông già bạn có xem không rồi hãng phát Ngôn😂😂 trẻ con nào mỗi mùa kiếm về cho Giải Chục Tỷ bảng"


def demo_clean(text=SAMPLE_RAW_TEXT, author="sonpham-yf1vd"):
    stages = []

    def add(label, value):
        stages.append({"stage": label, "text": value})

    t = unicodedata.normalize("NFC", str(text))
    add("0. raw", t)

    for k, v in CONTRACTIONS.items():
        t = re.sub(rf"\b{re.escape(k)}\b", v, t, flags=re.IGNORECASE)
    add("1. contractions", t)

    t = re.sub(r"\bRT\b", " ", t)
    t = re.sub(rf"@{re.escape(author)}\b", " ", t, flags=re.IGNORECASE)
    add("2. mentions", t)

    t = MEDIA_URL_RE.sub(" media ", t)
    t = URL_RE.sub(" link ", t)
    t = MENTION_RE.sub(" tag ", t)
    add("3. urls_and_tags", t)

    t = UNICODE_EMOJI_RE.sub(" ", t)
    for emo in TEXT_EMOJIS:
        t = t.replace(emo, " ")
    add("4. emojis", t)

    t = HASHTAG_RE.sub(lambda m: CAMEL_RE.sub(" ", m.group(1)), t)
    add("5. hashtags", t)

    t = REPEAT_RE.sub(r"\1", t)
    t = YEAR_RE.sub(" year ", t)
    t = NUMBER_RE.sub(" number ", t)
    add("6. repeats_and_numbers", t)

    t = t.lower()
    tokens = re.findall(r"\w+|[^\w\s]", t, flags=re.UNICODE)
    expanded_tokens = []
    for tok in tokens:
        if tok in ABBREVIATIONS:
            expanded_tokens.append(ABBREVIATIONS[tok])
        elif tok in SLANG_MAP:
            expanded_tokens.append(SLANG_MAP[tok])
        else:
            expanded_tokens.append(tok)
    t = " ".join(expanded_tokens)
    add("7. slang_and_abbreviations", t)

    words = [w for w in re.findall(r"\w+", t, flags=re.UNICODE) if w not in STOPWORDS]
    t = SPACE_RE.sub(" ", " ".join(words)).strip()
    add("8. final", t)

    return pd.DataFrame(stages)


def clean_and_extract_behavioural(text, author=""):
    raw = unicodedata.normalize("NFC", str(text or ""))

    unicode_emojis = UNICODE_EMOJI_RE.findall(raw)
    num_emojis = sum(len(e) for e in unicode_emojis)
    num_text_emojis = sum(raw.count(te) for te in TEXT_EMOJIS)
    raw_words = raw.split()
    num_caps_words = sum(1 for w in raw_words if len(w) >= 2 and w.isupper())
    num_unicode_chars = sum(1 for ch in raw if ord(ch) > 127)
    contains_media = int(bool(MEDIA_URL_RE.search(raw)))
    contains_link = int(bool(URL_RE.search(raw)))
    mentions = MENTION_RE.findall(raw)
    num_tagged_people = len(mentions)
    num_exclamations = raw.count("!") + raw.count("?")

    lower_words = set(re.findall(r"\w+", raw.lower(), flags=re.UNICODE))
    used_slang = int(bool(lower_words.intersection(SLANG_MAP.keys())))

    t = raw
    for k, v in CONTRACTIONS.items():
        t = re.sub(rf"\b{re.escape(k)}\b", v, t, flags=re.IGNORECASE)
    t = re.sub(r"\bRT\b", " ", t)
    if author:
        t = re.sub(rf"@{re.escape(author)}\b", " ", t, flags=re.IGNORECASE)
    t = MEDIA_URL_RE.sub(" media ", t)
    t = URL_RE.sub(" link ", t)
    t = MENTION_RE.sub(" tag ", t)
    t = UNICODE_EMOJI_RE.sub(" ", t)
    for emo in TEXT_EMOJIS:
        t = t.replace(emo, " ")
    t = HASHTAG_RE.sub(lambda m: CAMEL_RE.sub(" ", m.group(1)), t)
    t = REPEAT_RE.sub(r"\1", t)
    t = YEAR_RE.sub(" year ", t)
    t = NUMBER_RE.sub(" number ", t)
    t = t.lower()

    tokens = re.findall(r"\w+", t, flags=re.UNICODE)
    cleaned_words = []
    for tok in tokens:
        if tok in ABBREVIATIONS:
            cleaned_words.extend(ABBREVIATIONS[tok].split())
        elif tok in SLANG_MAP:
            cleaned_words.extend(SLANG_MAP[tok].split())
        elif tok not in STOPWORDS:
            cleaned_words.append(tok)

    cleaned_text = " ".join(cleaned_words).strip()

    behav = {
        "num_emojis": num_emojis,
        "num_text_emojis": num_text_emojis,
        "num_caps_words": num_caps_words,
        "num_unicode_chars": num_unicode_chars,
        "contains_media": contains_media,
        "contains_link": contains_link,
        "num_tagged_people": num_tagged_people,
        "used_slang": used_slang,
        "num_exclamations": num_exclamations,
    }
    return cleaned_text, behav


def process_all_tables(
    raw_comments_path=PROCESSED_DIR / "raw_comments.csv",
    raw_posts_path=PROCESSED_DIR / "posts.csv",
    out_comments_path=PROCESSED_DIR / "processed_comments.csv",
    out_posts_path=PROCESSED_DIR / "processed_posts.csv",
):
    comments = pd.read_csv(raw_comments_path)
    posts = pd.read_csv(raw_posts_path)

    cleaned_texts = []
    behav_rows = []
    for text in comments["comment_content"].fillna(""):
        c_text, b_dict = clean_and_extract_behavioural(text)
        cleaned_texts.append(c_text)
        behav_rows.append(b_dict)

    comments["cleaned_content"] = cleaned_texts
    behav_df = pd.DataFrame(behav_rows)
    for col in BEHAVIOURAL_COLS:
        comments[col] = behav_df[col]

    posts["cleaned_post_content"] = [
        clean_and_extract_behavioural(t)[0] for t in posts["post_content"].fillna("")
    ]

    comments.to_csv(out_comments_path, index=False, encoding="utf-8")
    posts.to_csv(out_posts_path, index=False, encoding="utf-8")
    return comments, posts


if __name__ == "__main__":
    c_df, p_df = process_all_tables()
    print(c_df.shape, p_df.shape)
