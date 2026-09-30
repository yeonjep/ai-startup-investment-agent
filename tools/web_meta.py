# 인용한 웹 페이지의 실제 발행일을 페이지 메타데이터에서 읽는다 (REFERENCE의 YYYY-MM-DD 표기용).
# 찾지 못하면 None을 반환하며 날짜를 추정해 만들지 않는다.

import json
import re
from datetime import date
from functools import lru_cache

import requests

_META_KEYS = (
    "article:published_time", "og:article:published_time", "datepublished", "pubdate",
    "publishdate", "date", "dc.date", "dc.date.issued", "article:published", "sailthru.date",
)
_ISO = re.compile(r"(20\d{2})-(\d{2})-(\d{2})")
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; investment-report-bot/1.0)"}


def _valid(text: str | None) -> str | None:
    match = _ISO.search(text or "")
    if not match:
        return None
    try:
        found = date(*(int(g) for g in match.groups()))
    except ValueError:
        return None
    return found.isoformat() if found <= date.today() else None


@lru_cache(maxsize=256)
def fetch_published_date(url: str, timeout: float = 6.0) -> str | None:
    try:
        response = requests.get(url, headers=_HEADERS, timeout=timeout)
        if response.status_code != 200 or "html" not in response.headers.get("content-type", "html"):
            return None
        html = response.text[:400_000]
    except Exception:  # noqa: BLE001 - 네트워크 실패는 날짜 미확인으로 남긴다
        return None

    for tag in re.findall(r"<meta[^>]+>", html, flags=re.I):
        key = re.search(r'(?:property|name|itemprop)=["\']([^"\']+)["\']', tag, flags=re.I)
        value = re.search(r'content=["\']([^"\']+)["\']', tag, flags=re.I)
        if key and value and key.group(1).lower() in _META_KEYS:
            found = _valid(value.group(1))
            if found:
                return found
    for block in re.findall(r'<script[^>]+ld\+json[^>]*>(.*?)</script>', html, flags=re.I | re.S):
        try:
            data = json.loads(block)
        except ValueError:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if isinstance(item, dict):
                found = _valid(str(item.get("datePublished") or ""))
                if found:
                    return found
    time_tag = re.search(r"<time[^>]+datetime=[\"']([^\"']+)[\"']", html, flags=re.I)
    return _valid(time_tag.group(1)) if time_tag else None
