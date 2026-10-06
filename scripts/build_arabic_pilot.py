"""Build 11 attributed Arabic document collections for a chunking pilot.

The collections combine Arabic Wikipedia articles. They approximate the CLAIR
domain mix and token lengths, but are not equivalent to the original documents.
Article text is licensed CC BY-SA; see the generated manifest for attribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import requests
import tiktoken


API = "https://ar.wikipedia.org/w/api.php"
USER_AGENT = "AdaptiveChunkingResearch/0.1 (Arabic corpus pilot; https://github.com/Ahmed05v/chunking_methods)"
ENC = tiktoken.get_encoding("o200k_base")
DOCS = [
    ("tech_01_ai", "technology", 5257, ["ذكاء اصطناعي", "تعلم الآلة", "شبكة عصبونية اصطناعية"]),
    ("tech_02_cyber", "technology", 5257, ["أمن الحاسوب", "أمن المعلومات", "تشفير"]),
    ("tech_03_digital", "technology", 5257, ["تحول رقمي", "حوسبة سحابية", "إنترنت الأشياء"]),
    ("legal_01_privacy", "legal", 30895, ["حماية البيانات", "خصوصية المعلومات", "قانون حماية البيانات", "بيانات شخصية"]),
    ("legal_02_constitution", "legal", 30895, ["دستور", "قانون دستوري", "سلطة قضائية", "حقوق الإنسان"]),
    ("legal_03_civil", "legal", 30895, ["قانون مدني", "عقد", "مسؤولية مدنية", "ملكية فكرية"]),
    ("legal_04_criminal", "legal", 30895, ["قانون جنائي", "جريمة", "إجراءات جنائية", "عقوبة"]),
    ("legal_05_international", "legal", 30895, ["قانون دولي", "معاهدة", "محكمة العدل الدولية", "حقوق اللاجئين"]),
    ("social_01_education", "social", 79862, ["تعليم", "تعليم في الوطن العربي", "محو الأمية", "مدرسة"]),
    ("social_02_health", "social", 79862, ["صحة عامة", "رعاية صحية", "تغذية", "وباء"]),
    ("social_03_development", "social", 79862, ["تنمية مستدامة", "فقر", "هجرة بشرية", "بطالة"]),
]


def api_get(session: requests.Session, **params: str | int) -> dict:
    for attempt in range(5):
        response = session.get(API, params={"format": "json", **params}, timeout=45)
        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            time.sleep(2 ** attempt)
            continue
        return data
    raise RuntimeError(f"Wikipedia API failed: {params}")


def search(session: requests.Session, query: str, cache_dir: Path) -> list[dict]:
    key = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    path = cache_dir / f"search_{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = api_get(session, action="query", list="search", srsearch=query, srlimit=50, srnamespace=0)
    hits = data["query"]["search"]
    path.write_text(json.dumps(hits, ensure_ascii=False), encoding="utf-8")
    return hits


def get_article(session: requests.Session, pageid: int, cache_dir: Path) -> dict:
    path = cache_dir / f"{pageid}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = api_get(session, action="query", prop="extracts|info|revisions", pageids=pageid,
                   explaintext=1, inprop="url", rvprop="ids|timestamp", rvlimit=1)
    page = data["query"]["pages"][str(pageid)]
    result = {
        "pageid": pageid,
        "title": page.get("title", ""),
        "url": page.get("fullurl", ""),
        "revision": page.get("lastrevid"),
        "extract": page.get("extract", ""),
    }
    path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def clean_extract(text: str) -> str:
    text = re.sub(r"(?m)^={2,6}\s*(.*?)\s*={2,6}\s*$", r"### \1", text)
    text = re.split(r"(?m)^### (?:المراجع|وصلات خارجية|انظر أيضًا|مراجع|مصادر)\s*$", text)[0]
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def tokens(text: str) -> int:
    return len(ENC.encode(text))


def build_doc(name: str, domain: str, target: int, queries: list[str],
              session: requests.Session, cache_dir: Path, used: set[int]) -> tuple[dict, dict]:
    candidates = []
    seen = set()
    for query in queries:
        for hit in search(session, query, cache_dir):
            pid = hit["pageid"]
            if pid not in used and pid not in seen and hit.get("wordcount", 0) >= 350:
                seen.add(pid)
                candidates.append(hit)
    paragraphs = [f"# مجموعة عربية تجريبية: {name}\n\n"]
    sources = []
    size = tokens(paragraphs[0])
    for hit in candidates:
        if size >= target:
            break
        article = get_article(session, hit["pageid"], cache_dir)
        raw = clean_extract(article["extract"])
        if sum("\u0600" <= c <= "\u06ff" for c in raw) < 1000:
            continue
        section = f"## {article['title']}\n\n{raw}\n\n"
        if tokens(section) < 500:
            continue
        remaining = target - size
        if remaining <= 1000:
            break
        # Keep entire articles where possible; shorten only the final article at
        # a paragraph boundary to keep document lengths close to the target.
        if tokens(section) > remaining * 1.15 and remaining > 1000:
            parts = re.split(r"\n\n+", raw)
            chosen = []
            for part in parts:
                trial = f"## {article['title']}\n\n" + "\n\n".join(chosen + [part]) + "\n\n"
                if tokens(trial) > remaining:
                    break
                chosen.append(part)
            if chosen:
                section = f"## {article['title']}\n\n" + "\n\n".join(chosen) + "\n\n"
            else:
                # Some articles have a very long opening paragraph. Take an
                # excerpt ending at a sentence or word boundary.
                lo, hi = 0, len(raw)
                prefix = f"## {article['title']}\n\n"
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if tokens(prefix + raw[:mid] + "\n\n") <= remaining:
                        lo = mid
                    else:
                        hi = mid - 1
                end = max(raw.rfind(". ", 0, lo), raw.rfind(".\n", 0, lo),
                          raw.rfind(".\n\n", 0, lo), raw.rfind(". ", 0, lo))
                if end < lo * 0.7:
                    end = raw.rfind(" ", 0, lo)
                section = prefix + raw[:max(end, 0)].strip() + "\n\n"
        paragraphs.append(section)
        size += tokens(section)
        used.add(hit["pageid"])
        sources.append({k: article[k] for k in ("pageid", "title", "url", "revision")})
    if size < target * 0.7:
        raise RuntimeError(f"{name}: only {size} of {target} tokens; add more search topics")

    text = "".join(paragraphs)
    # Synthetic pages are contiguous ~2,000-token spans ending at paragraph
    # boundaries. The distinction from original PDF pages is recorded below.
    page_parts = re.findall(r".*?(?:\n\n|$)", text, flags=re.DOTALL)
    pages = []
    current = ""
    for part in page_parts:
        if not part:
            continue
        if current and tokens(current + part) > 2000:
            pages.append(current)
            current = ""
        current += part
    if current:
        pages.append(current)
    assert "".join(pages) == text

    split_points = sorted({m.end() for m in re.finditer(r"\n\n+", text) if 0 < m.end() < len(text)})
    parsed = {
        "document_name": name,
        "pages": {str(i + 1): page for i, page in enumerate(pages)},
        "full_text": text,
        "split_points": split_points,
        "titles": [{"title": name, "start": 0, "level": 1, "end": len(text)}],
    }
    manifest = {
        "name": name, "domain": domain, "target_tokens": target,
        "tokens_o200k_base": tokens(text), "synthetic_pages": len(pages),
        "source_articles": sources, "source_count": len(sources),
        "arabic_character_fraction": round(sum("\u0600" <= c <= "\u06ff" for c in text) / max(len(text), 1), 3),
    }
    return parsed, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/arabic_pilot"))
    args = parser.parse_args()
    root = args.output_dir
    parsed_dir = root / "adi_parsed"
    cache_dir = root / "source_cache"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    used: set[int] = set()
    manifest = []
    for name, domain, target, queries in DOCS:
        parsed, entry = build_doc(name, domain, target, queries, session, cache_dir, used)
        (parsed_dir / f"{name}.json").write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
        manifest.append(entry)
        print(f"{name}: {entry['tokens_o200k_base']:,} tokens, {entry['source_count']} articles, {entry['synthetic_pages']} pages", flush=True)
    (root / "manifest.json").write_text(json.dumps({
        "description": "Eleven Arabic Wikipedia article collections for a chunking pilot; not natural single-source documents.",
        "license": "Arabic Wikipedia text is CC BY-SA 4.0; attribution URLs and revision IDs are listed per document.",
        "page_note": "Pages are synthetic ~2,000-token paragraph groups; split_points are paragraph boundaries.",
        "documents": manifest,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
