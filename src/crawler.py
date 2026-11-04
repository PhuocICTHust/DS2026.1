import json
import random
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RANDOM_STATE, RAW_JSON_DIR


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
            "likes": int(str(c.get("votes", "0")).replace("K", "000").replace(".", "") or 0),
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


TOPICS = [
    ("showbiz_drama", "Threads Showbiz & Nghe si tranh cai phat ngon"),
    ("football_vleague", "Binh luan Bong da & Doi tuyen Quoc gia"),
    ("tech_ai_jobs", "Tranh luan AI thay the lap trinh vien & Sinh vien IT"),
    ("traffic_policy", "Thao luan Giao thong do thi & Xu phat nong do con"),
    ("education_exam", "Doi moi thi cu, hoc phi dai hoc & Ap luc diem so"),
    ("food_review", "Review Am thuc duong pho & Drama quan an TikTok/Threads"),
    ("cinema_viet", "Tranh cai Phim dien anh Viet doanh thu tram ty"),
    ("esports_gaming", "Cong dong Game thu & Drama giai dau Esports"),
    ("workplace_genz", "Tranh luan Van hoa cong so GenZ vs Sep truyen thong"),
    ("crypto_finance", "Dau tu chung khoan, tai chinh ca nhan & Lua ga"),
]

CONSTRUCTIVE_TEMPLATES = [
    "Mình thấy bài viết phân tích khá hợp lý và khách quan. Cảm ơn tác giả đã chia sẻ góc nhìn đa chiều nhé! 👍",
    "Quan điểm cá nhân của mình là vấn đề nào cũng có hai mặt, chúng ta nên bình tĩnh lắng nghe trước khi phán xét.",
    "Hoàn toàn đồng ý với ý số 2. Thực tế ở {loc} mình cũng gặp trường hợp tương tự, giải quyết nhẹ nhàng là ổn.",
    "Cảm ơn bạn đã tổng hợp thông tin chi tiết nhé, đọc dễ hiểu hẳn. Chúc kênh ngày càng phát triển! ❤️",
    "Mình nghĩ giải pháp lâu dài vẫn là nâng cao ý thức cộng đồng chứ không chỉ dựa vào quy định xử phạt.",
    "Bài phân tích rất có tâm, số liệu rõ ràng. Mong có thêm nhiều bài viết chất lượng như thế này trên Threads.",
    "Đúng rồi bạn ơi, hôm qua mình cũng vừa đọc báo chính thống nói về vụ này, thông tin hoàn toàn chuẩn xác.",
    "Thật sự rất ngưỡng mộ sự nỗ lực của ekip, dù còn vài điểm nhỏ cần cải thiện nhưng tổng thể quá tốt rồi 👏",
    "Theo kinh nghiệm 3 năm đi làm của mình thì kỹ năng tự học quan trọng hơn bằng cấp rất nhiều, các bạn sinh viên cứ yên tâm nhé.",
    "Góc nhìn rất văn minh và lịch sự. Giá như ai lên mạng xã hội cũng tranh luận đàng hoàng thế này thì tốt biết mấy!",
]

BLATANT_TOXIC_TEMPLATES = [
    "Đúng là lũ NGU NGỐC không biết gì cũng bày đặt lên mạng dạy đời!! Cút mẹ chúng mày đi lũ rác rưởi 😡🤬",
    "Thằng viết bài này bị ngáo đá hay mất não vậy??? Nói chuyện ngu vcl đéo chịu nổi luôn á!!",
    "Bọn bẩn tính vô học này suốt ngày chỉ biết cắn bậy thôi, đồ súc vật hạ đẳng cặn bã xã hội 🤮🤬",
    "Câm cái mồm lại đi đồ dốt nát!! Đã ngu còn tỏ ra nguy hiểm, nhìn cái mặt trên link https://t.co/xyz123 là thấy hãm tài rồi!",
    "Lũ fan cuồng mù quáng điên khùng, ai trái ý là nhảy vào cắn xé như lũ chó đàn, kinh tởm thật sự!!",
    "Đm thằng @user_tag mày bị điên à?? Ngu hết phần thiên hạ, biến khỏi mạng xã hội giùm cái đồ phế vật!!",
    "Toàn một lũ đạo đức giả, mở miệng ra là văn vở nhưng sống bẩn vãi chưởng, nhìn mà phát tởm 🤮",
    "Thứ rác rưởi câu view rẻ tiền!! Tẩy chay chết mẹ cái kênh mất dạy này đi anh em ơi #TayChay #RacRuoi",
]

MASKED_TOXIC_TEMPLATES = [
    "Dạ vâng bạn thì thông minh nhất vũ trụ rồi ạ, người thường có não không ai tư duy đỉnh cao như bạn đâu, xin bái phục.",
    "Đọc bình luận của bạn xong mình mới hiểu tại sao giáo dục cơ bản lại quan trọng đến vậy. Chúc bạn sớm tìm lại được nhận thức.",
    "Không sao đâu bạn, thiếu kiến thức không phải là cái tội, cái tội là tự hào khoe sự kém hiểu biết đó lên mạng công cộng thôi.",
    "Cảm ơn bạn đã dũng cảm chứng minh cho mọi người thấy giới hạn dưới của trí tuệ con người nằm ở đâu nhé, thật đáng ngưỡng mộ.",
    "Chắc bố mẹ bạn ở nhà vất vả lắm mới nuôi lớn được một người có tư duy ngược đời và ích kỷ một cách tự nhiên như vậy.",
    "Lời nói của bạn nghe thì có vẻ lịch sự đấy nhưng thực chất là sự ngụy biện rẻ tiền để che đậy bản chất hẹp hòi và độc hại.",
]

AMBIGUOUS_TEMPLATES = [
    "Ủa alo vụ này căng đét vậy hả mọi người?? Hóng biến từ tối qua tới giờ chưa thấy hồi kết luôn á 😂🍿",
    "Nghe cũng hơi cấn cấn sao á, ko biết thực hư thế nào chứ trên mạng giờ toàn drama câu tương tác thôi",
    "Vụ này nói đi cũng phải nói lại, cả hai bên đều có cái sai chứ ko ai vừa đâu, hóng tập tiếp theo lol btw",
    "Trời ơi tin chuẩn chưa anh em?? Thấy thiên hạ đồn ầm ầm từ sáng trên https://threads.net/t/123 mà hoang mang quá trời!",
    "Thôi giải tán đi ngủ đi mấy má ơi, cãi nhau từ sáng tới đêm ko mệt hả, việc nhà thì nhác việc chú bác thì siêng =))",
    "Căng cực căng cực!! Pha này xử lý hơi đi vào lòng đất rồi, để xem ngày mai lên bài xin lỗi kiểu gì đây 🤔",
]

LOCATIONS = ["hn", "hcm", "đn", "vn", "us", "uk"]


def generate_seed_social_trees(num_topics=10, posts_per_topic=12, seed=RANDOM_STATE):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    users = []
    for u_idx in range(1, 2201):
        r = rng.random()
        if r < 0.72:
            persona = "normal"
            weight = np_rng.pareto(2.5) + 0.5
        elif r < 0.87:
            persona = "toxic"
            weight = np_rng.pareto(1.4) + 1.2
        else:
            persona = "ambiguous"
            weight = np_rng.pareto(2.0) + 0.8
        users.append({
            "user_id": f"user_{u_idx:04d}",
            "persona": persona,
            "weight": float(weight),
        })

    weights = np.array([u["weight"] for u in users])
    probs = weights / weights.sum()

    def sample_comment(user, is_reply=False):
        persona = user["persona"]
        p = rng.random()
        if persona == "normal":
            if p < 0.85:
                cat = "constructive"
            elif p < 0.96:
                cat = "ambiguous"
            else:
                cat = "masked_toxic"
        elif persona == "toxic":
            if p < 0.60:
                cat = "blatant_toxic"
            elif p < 0.88:
                cat = "masked_toxic"
            else:
                cat = "ambiguous"
        else:
            if p < 0.55:
                cat = "ambiguous"
            elif p < 0.80:
                cat = "constructive"
            else:
                cat = "blatant_toxic"

        loc = rng.choice(LOCATIONS)
        if cat == "constructive":
            text = rng.choice(CONSTRUCTIVE_TEMPLATES).format(loc=loc)
            gt_label = 0
        elif cat == "blatant_toxic":
            text = rng.choice(BLATANT_TOXIC_TEMPLATES).format(loc=loc)
            gt_label = 1
        elif cat == "masked_toxic":
            text = rng.choice(MASKED_TOXIC_TEMPLATES).format(loc=loc)
            gt_label = 1
        else:
            text = rng.choice(AMBIGUOUS_TEMPLATES).format(loc=loc)
            gt_label = 1 if rng.random() < 0.35 else 0

        if is_reply and rng.random() < 0.65:
            text = f"@author_reply {text}"
        if rng.random() < 0.25:
            text += f" #{rng.choice(['TrendingVN', 'ThreadsVN', 'GocNhin', 'Debate2026', 'TinNong'])}"
        if rng.random() < 0.20:
            text += f" năm {rng.choice([2024, 2025, 2026])}"
        if rng.random() < 0.12 and cat == "blatant_toxic":
            text = text.upper()

        return {"text": text, "gt_label": gt_label, "category": cat}

    saved_files = []
    global_cmt_counter = 10000

    for t_idx in range(min(num_topics, len(TOPICS))):
        topic_slug, topic_title = TOPICS[t_idx]
        topic_posts = []

        for p_idx in range(1, posts_per_topic + 1):
            post_id = f"post_{t_idx + 1:02d}_{p_idx:03d}"
            host_username = f"influencer_{t_idx + 1:02d}"
            n_root_comments = rng.randint(18, 32)
            root_comments = []

            for _ in range(n_root_comments):
                global_cmt_counter += 1
                u = users[int(np_rng.choice(len(users), p=probs))]
                sampled = sample_comment(u, is_reply=False)
                root_cid = f"cmt_{global_cmt_counter}"

                n_replies = rng.choices([0, 1, 2, 3, 4], weights=[0.45, 0.25, 0.15, 0.10, 0.05])[0]
                replies_lvl1 = []
                for _ in range(n_replies):
                    global_cmt_counter += 1
                    u_rep = users[int(np_rng.choice(len(users), p=probs))]
                    sampled_rep = sample_comment(u_rep, is_reply=True)
                    rep_cid = f"cmt_{global_cmt_counter}"

                    n_sub = rng.choices([0, 1, 2], weights=[0.70, 0.20, 0.10])[0]
                    replies_lvl2 = []
                    for _ in range(n_sub):
                        global_cmt_counter += 1
                        u_sub = users[int(np_rng.choice(len(users), p=probs))]
                        sampled_sub = sample_comment(u_sub, is_reply=True)
                        replies_lvl2.append({
                            "comment_id": f"cmt_{global_cmt_counter}",
                            "username": u_sub["user_id"],
                            "content": sampled_sub["text"],
                            "likes": int(np_rng.poisson(3)),
                            "gt_label": sampled_sub["gt_label"],
                            "category": sampled_sub["category"],
                            "replies_content": [],
                        })

                    replies_lvl1.append({
                        "comment_id": rep_cid,
                        "username": u_rep["user_id"],
                        "content": sampled_rep["text"],
                        "likes": int(np_rng.poisson(6)),
                        "gt_label": sampled_rep["gt_label"],
                        "category": sampled_rep["category"],
                        "replies_content": replies_lvl2,
                    })

                root_comments.append({
                    "comment_id": root_cid,
                    "username": u["user_id"],
                    "content": sampled["text"],
                    "likes": int(np_rng.poisson(14)),
                    "gt_label": sampled["gt_label"],
                    "category": sampled["category"],
                    "replies_content": replies_lvl1,
                })

            topic_posts.append({
                "post_id": post_id,
                "username": host_username,
                "topic": topic_slug,
                "content": f"[{topic_title}] Bài thảo luận số #{p_idx}: Quan điểm của mọi người về vấn đề này như thế nào?",
                "likes": int(np_rng.integers(150, 2500)),
                "replies_content": root_comments,
            })

        out_path = RAW_JSON_DIR / f"topic_{t_idx + 1:02d}_{topic_slug}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(topic_posts, fh, ensure_ascii=False, indent=2)
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
