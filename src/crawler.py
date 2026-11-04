import json
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RANDOM_STATE, RAW_JSON_DIR

TOPIC_CONFIGS = [
    ("topic_01_showbiz_drama", "real_vnexpress_giaitri.json", "Showbiz & Nghe si tranh cai"),
    ("topic_02_football_vleague", "real_vnexpress_thethao.json", "Binh luan Bong da & V-League"),
    ("topic_03_tech_ai_jobs", "real_vnexpress_sohoa.json", "Tranh luan AI thay the lap trinh vien IT"),
    ("topic_04_traffic_policy", "real_vnexpress_otoxemay.json", "Giao thong do thi & Xu phat nong do con"),
    ("topic_05_education_exam", "real_vnexpress_giaoduc.json", "Doi moi thi cu & Ap luc diem so"),
    ("topic_06_food_review", "real_vnexpress_doisong.json", "Review Am thuc duong pho & Drama quan an"),
    ("topic_07_cinema_viet", "real_vnexpress_phapluat.json", "Tranh cai Phim dien anh Viet & Ban quyen"),
    ("topic_08_esports_gaming", "real_vnexpress_thoisu.json", "Cong dong Game thu & Van hoa mang"),
    ("topic_09_workplace_genz", "real_vnexpress_gocnhin.json", "Van hoa cong so GenZ vs Sep truyen thong"),
    ("topic_10_crypto_finance", "real_vnexpress_kinhdoanh.json", "Dau tu chung khoan, tai chinh & Lua dao"),
]


def crawl_reddit_thread_json(thread_url, output_json_path):
    import requests

    clean_url = thread_url.rstrip("/")
    if not clean_url.endswith(".json"):
        clean_url += ".json"

    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(clean_url, headers=headers, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    post_meta = payload[0]["data"]["children"][0]["data"]
    root_post_id = f"post_{post_meta.get('id', '001')}"

    def parse_tree(children, depth=1, max_depth=3):
        if depth > max_depth:
            return []
        nodes = []
        for item in children:
            if item.get("kind") != "t1":
                continue
            cdata = item.get("data", {})
            cid = f"cmt_{cdata.get('id')}"
            replies_raw = cdata.get("replies")
            sub_children = []
            if isinstance(replies_raw, dict):
                sub_children = replies_raw.get("data", {}).get("children", [])
            nodes.append({
                "comment_id": cid,
                "username": cdata.get("author", "anon_user"),
                "content": cdata.get("body", ""),
                "likes": int(cdata.get("ups", 0)),
                "replies_content": parse_tree(sub_children, depth + 1, max_depth),
            })
        return nodes

    comment_listing = payload[1]["data"]["children"]
    post_obj = {
        "post_id": root_post_id,
        "username": post_meta.get("author", "topic_host"),
        "topic": post_meta.get("subreddit", "general"),
        "content": f"{post_meta.get('title', '')} - {post_meta.get('selftext', '')}".strip(),
        "likes": int(post_meta.get("ups", 0)),
        "replies_content": parse_tree(comment_listing, depth=1, max_depth=3),
    }

    with open(output_json_path, "w", encoding="utf-8") as fh:
        json.dump([post_obj], fh, ensure_ascii=False, indent=2)
    return post_obj


def crawl_youtube_video_comments(video_url, output_json_path, max_comments=500):
    from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR

    downloader = YoutubeCommentDownloader()
    generator = downloader.get_comments_from_url(video_url, sort_by=SORT_BY_POPULAR)

    root_map = {}
    orphan_replies = []

    count = 0
    for c in generator:
        if count >= max_comments:
            break
        cid = str(c.get("cid", f"yt_{count}"))
        node = {
            "comment_id": f"cmt_{cid}",
            "username": c.get("author", "yt_user"),
            "content": c.get("text", ""),
            "likes": int(str(c.get("votes", "0")).replace("K", "000").replace("N", "000").replace(",", "").replace(".", "") or 0),
            "replies_content": [],
        }
        if "." in cid:
            parent_cid = cid.split(".")[0]
            if parent_cid in root_map:
                root_map[parent_cid]["replies_content"].append(node)
            else:
                orphan_replies.append((parent_cid, node))
        else:
            root_map[cid] = node
        count += 1

    for parent_cid, node in orphan_replies:
        if parent_cid in root_map:
            root_map[parent_cid]["replies_content"].append(node)

    post_obj = {
        "post_id": f"yt_post_{abs(hash(video_url)) % 100000}",
        "username": "youtube_channel",
        "topic": "youtube_discussion",
        "content": f"YouTube Video ({video_url})",
        "likes": 1200,
        "replies_content": list(root_map.values()),
    }
    with open(output_json_path, "w", encoding="utf-8") as fh:
        json.dump([post_obj], fh, ensure_ascii=False, indent=2)
    return post_obj


def crawl_discussion_tree_json(category_url, output_json_path, max_posts=20):
    import urllib.request

    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(category_url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as r:
        payload = json.loads(r.read().decode("utf-8"))

    with open(output_json_path, "w", encoding="utf-8") as fh:
        json.dump(payload[:max_posts], fh, ensure_ascii=False, indent=2)
    return payload[:max_posts]


def generate_seed_social_trees(raw_dir=RAW_JSON_DIR):
    saved_files = []
    backup_dir = Path(__file__).resolve().parent.parent.parent / "packages_for_members" / "FULL_BACKUP" / "data" / "raw_json"
    yt_cache = Path(__file__).resolve().parent.parent.parent / "scraped_youtube_comments.json"

    yt_data = {}
    if yt_cache.exists():
        with open(yt_cache, "r", encoding="utf-8") as fh:
            yt_data = json.load(fh)

    for slug, backup_name, desc in TOPIC_CONFIGS:
        out_path = raw_dir / f"{slug}.json"
        if out_path.exists():
            saved_files.append(out_path)
            continue

        src_backup = backup_dir / backup_name
        if src_backup.exists():
            with open(src_backup, "r", encoding="utf-8") as fh:
                posts = json.load(fh)
        else:
            posts = []

        topic_clean = slug.split("_", 2)[-1]
        for p in posts:
            p["topic"] = topic_clean

        if slug in yt_data and yt_data[slug].get("replies_content"):
            yt_post = yt_data[slug]
            yt_post["topic"] = topic_clean
            posts.append(yt_post)

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(posts, fh, ensure_ascii=False, indent=2)
        saved_files.append(out_path)

    return saved_files


def describe_node(node, depth=0, max_depth=2):
    if depth > max_depth:
        return
    kids = node.get("replies_content") or []
    prefix = "  " * depth
    content = str(node.get("content", ""))[:58].replace("\n", " ")
    print(f"{prefix}- @{node.get('username')} | likes={node.get('likes')} | {content!r}")
    for child in kids[:2]:
        describe_node(child, depth + 1, max_depth)


def flatten_trees_to_csv(
    json_dir=RAW_JSON_DIR,
    posts_out=PROCESSED_DIR / "posts.csv",
    comments_out=PROCESSED_DIR / "raw_comments.csv",
):
    post_rows = []
    comment_rows = []

    json_files = sorted(json_dir.glob("*.json"))
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as fh:
            posts_data = json.load(fh)

        for post in posts_data:
            post_id = post.get("post_id")
            topic = post.get("topic", "general")
            post_rows.append({
                "post_id": post_id,
                "author": post.get("username", "unknown"),
                "topic": topic,
                "post_content": post.get("content", ""),
                "likes": post.get("likes", 0),
            })

            def traverse_comments(children, parent_comment_id=None):
                for c in children:
                    cid = c.get("comment_id")
                    comment_rows.append({
                        "commenter_id": c.get("username", "anon"),
                        "comment_id": cid,
                        "parent_id": parent_comment_id,
                        "post_id": post_id,
                        "topic": topic,
                        "likes": int(c.get("likes", 0)),
                        "comment_content": c.get("content", ""),
                        "ground_truth_label": c.get("gt_label", np.nan),
                        "category": c.get("category", "unknown"),
                    })
                    sub_replies = c.get("replies_content") or []
                    if sub_replies:
                        traverse_comments(sub_replies, parent_comment_id=cid)

            traverse_comments(post.get("replies_content") or [], parent_comment_id=None)

    posts_df = pd.DataFrame(post_rows)
    comments_df = pd.DataFrame(comment_rows)

    posts_df.to_csv(posts_out, index=False, encoding="utf-8")
    comments_df.to_csv(comments_out, index=False, encoding="utf-8")
    return posts_df, comments_df


if __name__ == "__main__":
    files = generate_seed_social_trees()
    posts_df, comments_df = flatten_trees_to_csv()
    print(f"Posts: {len(posts_df)}, Comments: {len(comments_df)}")
