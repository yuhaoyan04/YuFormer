from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

DECISION_KEEP = "keep"
DECISION_REVIEW = "review"
DECISION_DROP = "drop"

MIN_CHARS = 16
MAX_REVIEW_CHARS = 2_000_000

REPLACEMENT_CHAR_RATIO = 1e-4
CONTROL_CHAR_RATIO = 1e-3
HTML_TAG_RATIO = 0.05
HTML_ENTITY_RATIO = 0.01
BASE64_BLOB_CHARS = 8192
WEB_LINE_DUP_RATIO = 0.55
LANG_LOW_CONF = 0.3
MOJIBAKE_BIGRAM_RATIO = 5e-4

_SECRET_KEY_RE = re.compile(
    r"(?:[aA][pP][iI][_-]?[kK][eE][yY]|[sS][eE][cC][rR][eE][tT]|[pP][aA][sS]{2}[wW][oO][rR][dD]|"
    r"[aA][cC]{2}[eE][sS]{2}[_-]?[tT][oO][kK][eE][nN])[\"']?\s*[:=][\"']?"
)
_TAG_RE = re.compile(r"<[^<>\n]{1,200}>")
_SCRIPT_STYLE_RE = re.compile(r"(?is)<\s*(script|style)[\s>]")
_ENTITY_RE = re.compile(r"&(?:[a-zA-Z]{2,32}|#\d{1,8}|#x[0-9a-fA-F]{1,8});")
_DATA_URI_RE = re.compile(
    r"data:[a-zA-Z0-9!#$&\-^_.]+/[a-zA-Z0-9!#$&\-^_.]+;base64,([A-Za-z0-9+/=\s]{%d,})"
    % BASE64_BLOB_CHARS
)
_BASE64_RUN_RE = re.compile(r"[A-Za-z0-9+/]{%d,}={0,2}" % BASE64_BLOB_CHARS)

# Latin-1-as-UTF-8 mojibake 双字符签名（Ã©=é 等）；对应语言正文不产生这些双字符
_MOJIBAKE_BIGRAMS = (
    "Ã©", "Ã¨", "Ãª", "Ã«", "Ã¯", "Ã®", "Ã§", "Ã¨", "Ã¹", "Ã»", "Ã´", "Ã¶",
    "Ã¤", "Ã¥", "Ã¦", "Ã¸", "Ã±", "ÃŸ", "Ã‡", "Ã‰", "Ãˆ", "ÃŠ", "Ã…", "Ã„",
    "Ãœ", "Ã–", "Ã˜", "Ã‘", "Ãš", "Ã“", "Ã", "Ã­", "Ã¡", "Ã³", "Ãº", "Ã±",
    "Â«", "Â»", "Â°", "Â§", "Â©", "Â®", "Â±", "Âµ", "Ã€", "Ã ", "Ã¢", "Ã¤",
    # emoji/标点的 UTF-8→Windows-1252 双编码形态（ðŸ˜Š / â€“ / â€™ 系）
    "ðŸ", "â€", "â€™", "â€“", "â€œ", "â€\x9d", "â€¦", "â€¹", "â€º",
)
_MOJIBAKE_SET = frozenset(_MOJIBAKE_BIGRAMS)
_MOJIBAKE_RE = re.compile("|".join(re.escape(b) for b in sorted(_MOJIBAKE_SET)))

_DELETE_CODES = list(range(0x00, 0x09)) + [0x0B, 0x0C] + list(range(0x0E, 0x20)) + [0x7F]
_DELETE_TABLE = [chr(i) for i in range(256)]
for _c in _DELETE_CODES:
    _DELETE_TABLE[_c] = ""
_DELETE_TABLE = tuple(_DELETE_TABLE)
_WS_RUN_RE = re.compile(r"[^\S\n]+")
_TRAIL_SPACE_RE = re.compile(r" +$", flags=re.M)
_BLANK_RUN_RE = re.compile(r"\n{3,}")

_VENDOR_PATH_RE = re.compile(
    r"(node_modules|/vendor/|\.min\.js(?:\?|#|$|\")|\.map(?:\?|#|$|\"))"
)

_PATH_META_KEYS = (
    "path",
    "fpath",
    "max_stars_repo_path",
    "repo_path",
    "file_path",
    "filename",
)

_LID_MODEL_ENV = "YUFORMER_LID_MODEL"


@dataclass(slots=True)
class Verdict:
    decision: str = DECISION_KEEP
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def add(self, decision: str, reason: str) -> None:
        if decision == DECISION_DROP:
            self.decision = DECISION_DROP
        elif decision == DECISION_REVIEW and self.decision == DECISION_KEEP:
            self.decision = DECISION_REVIEW
        if reason not in self.reasons:
            self.reasons.append(reason)


def normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).translate(_DELETE_TABLE)
    if "  " in t or "\t" in t:
        t = _WS_RUN_RE.sub(" ", t)
    if " \n" in t or t.endswith(" "):
        t = _TRAIL_SPACE_RE.sub("", t)
    if "\n\n\n" in t:
        t = _BLANK_RUN_RE.sub("\n\n", t)
    return t.strip()


def check_integrity(text: str) -> tuple[str, str | None]:
    if len(text) < MIN_CHARS:
        return DECISION_DROP, "empty_or_short"
    if len(text) > MAX_REVIEW_CHARS:
        return DECISION_REVIEW, "too_long"
    return DECISION_KEEP, None


def check_garbled(text: str) -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    n = len(text)
    if not n:
        return hits
    if text.count("\ufffd") / n > REPLACEMENT_CHAR_RATIO:
        hits.append((DECISION_DROP, "garbled_replacement_char"))
    controls = n - len(text.translate(_DELETE_TABLE))
    if controls / n > CONTROL_CHAR_RATIO:
        hits.append((DECISION_DROP, "garbled_control_chars"))
    if "\u00c3" in text or "\u00c2" in text or "\u00f0\u0178" in text or "\u00e2\u20ac" in text:
        mojibake = len(_MOJIBAKE_RE.findall(text))
        if mojibake and mojibake / n > MOJIBAKE_BIGRAM_RATIO:
            hits.append((DECISION_DROP, "garbled_latin1_mojibake"))
    return hits


def check_html(text: str, domain: str = "web") -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    n = len(text)
    if not n:
        return hits
    if "<" in text:
        tag_chars = sum(len(m.group(0)) for m in _TAG_RE.finditer(text))
        if tag_chars / n > HTML_TAG_RATIO:
            hits.append((DECISION_DROP, "html_tag_ratio"))
        if _SCRIPT_STYLE_RE.search(text):
            hits.append((DECISION_DROP, "html_script_style_block"))
    if "&" in text:
        entity_chars = sum(len(m.group(0)) for m in _ENTITY_RE.finditer(text))  # 字符占比，与标签规则单位一致
        if entity_chars / n > HTML_ENTITY_RATIO:
            hits.append((DECISION_DROP, "html_entity_flood"))
    if n >= BASE64_BLOB_CHARS:
        run = _BASE64_RUN_RE.search(text)
        # DNA 单字母表豁免；非 web 域降级 review（academic/code 合法长串不误杀）
        dna_like = run is not None and set(run.group(0)) <= set("ACGTNacgtn")
        if not dna_like and (run is not None or (";base64," in text and _DATA_URI_RE.search(text))):
            dec = DECISION_DROP if domain == "web" else DECISION_REVIEW
            hits.append((dec, "base64_blob"))
    return hits


_SECRET_HINT = ("assword", "ecret", "oken", "key")
_SECRET_SPECIFIC_RE = re.compile(
    r"AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)
_PLACEHOLDER_RE = re.compile(r"YOUR|XXX|example|placeholder|changeme", re.IGNORECASE)


def check_secrets(text: str) -> tuple[str, str] | None:
    if _SECRET_SPECIFIC_RE.search(text):
        return DECISION_REVIEW, "secret_specific"
    if not any(h in text for h in _SECRET_HINT):
        return None
    m = _SECRET_KEY_RE.search(text)
    if m is None:
        return None
    value = text[m.end():].lstrip()
    if len(value) < 12:
        return None
    head = value[:24]
    if _PLACEHOLDER_RE.search(head):
        return None  # 教程占位符豁免
    quoted = value[0] in "\"'"
    mixed = (any(c.isdigit() for c in head) and any(c.isupper() for c in head)
             and any(c.islower() for c in head))
    if quoted or mixed:
        return DECISION_REVIEW, "secret_pattern"
    return None
    m = _SECRET_KEY_RE.search(text)
    if m is None:
        return None
    value = text[m.end() :].lstrip()
    if len(value) >= 12 and not value[0].isspace():
        return DECISION_REVIEW, "secret_pattern"
    return None


def check_web_line_dup(text: str) -> tuple[str, str] | None:
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    if len(lines) < 20:
        return None
    dup = 1.0 - len(set(lines)) / len(lines)
    if dup > WEB_LINE_DUP_RATIO:
        return DECISION_DROP, "web_line_dup"
    return None


def extract_path(meta: str) -> str | None:
    import json

    try:
        obj = json.loads(meta)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    for key in _PATH_META_KEYS:
        val = obj.get(key)
        if isinstance(val, str) and val:
            return val
    return None


def check_code_path(path: str | None) -> tuple[str, str] | None:
    if path is None:
        return None
    if _VENDOR_PATH_RE.search(path):
        return DECISION_DROP, "code_vendor_path"
    return None


class LangDetector:
    def __init__(self, model: Any):
        self._model = model

    def predict(self, text: str) -> tuple[str, float]:
        snippet = text[:4000].replace("\n", " ").replace("\r", " ").strip()
        if not snippet:
            return "unk", 0.0
        try:
            labels, probs = self._model.predict(snippet, k=1)
            return labels[0].replace("__label__", ""), float(probs[0])
        except ValueError:
            # numpy>=2 与 fasttext 0.9.x 包装器不兼容（np.array(probs, copy=False)）：
            # E-16 事故根因——凡需语言检测的分片第一行即崩。改走底层 f.predict 绕过。
            pairs = self._model.f.predict(snippet, 1, 0.0, "strict")
            if not pairs:
                return "unk", 0.0
            prob, label = pairs[0]
            return label.replace("__label__", ""), float(prob)


_lang_detector: LangDetector | None | bool = False


def get_lang_detector() -> LangDetector | None:
    global _lang_detector
    if _lang_detector is False:
        path = os.environ.get(_LID_MODEL_ENV, "")
        if path and os.path.isfile(path):
            try:
                import fasttext

                _lang_detector = LangDetector(fasttext.load_model(path))
            except Exception:
                _lang_detector = None
        else:
            _lang_detector = None
    return _lang_detector  # type: ignore[return-value]


def reset_lang_detector() -> None:
    global _lang_detector
    _lang_detector = False


_QA_INLINE_TAG_RE = re.compile(
    r"</?(?:p|br|hr|b|i|em|strong|code|pre|blockquote|ul|ol|li|a|sup|sub|h[1-6]"
    r"|table|thead|tbody|tr|td|th|img|span|div|strike|dl|dt|dd)\b[^<>\n]{0,300}>"
)


def clean_inline_html(text: str) -> str:
    """qa 域正文残留的 inline HTML 清理（stack-exchange 实测 `<p>`/`<a>` 残留）：
    strip 常见问答标签 + HTML 实体 unescape。仅用于低标签密度文本——高密度
    HTML 源文件由 check_html 的 html_tag_ratio drop 路径处置，与本函数互补。"""
    t = _QA_INLINE_TAG_RE.sub(" ", text)
    if "&" in t:
        import html as _html

        t = _html.unescape(t)
    return t


def clean_for_domain(text: str, domain: str) -> str:
    """域专属文本清洗（检测与写出都作用于清洗后文本）。qa 域启用 inline HTML
    清理；其他域原样返回——`&amp;` 在 code/math 域是语义内容，不可全局 unescape。"""
    if domain == "qa":
        return clean_inline_html(text)
    return text


def evaluate(text: str, domain: str, meta: str, lang: str) -> tuple[Verdict, str]:
    v = Verdict()
    v.metrics["raw_chars"] = float(len(text))
    d, r = check_integrity(text)
    if r:
        v.add(d, r)
    for d, r in check_garbled(text):
        v.add(d, r)
    for d, r in check_html(text, domain):
        v.add(d, r)
    sr = check_secrets(text)
    if sr:
        v.add(*sr)
    norm = normalize_text(text)
    v.metrics["norm_chars"] = float(len(norm))
    v.metrics["lines"] = float(norm.count("\n") + 1)

    if domain == "web":
        wd = check_web_line_dup(norm)
        if wd:
            v.add(*wd)
    elif domain == "code":
        cp = check_code_path(extract_path(meta))
        if cp:
            v.add(*cp)

    if not lang:
        det = get_lang_detector()
        if det is None:
            v.metrics["lang_skipped"] = 1.0
            lang = "unk"
        else:
            lang, prob = det.predict(norm)
            v.metrics["lang_prob"] = prob
            if prob < LANG_LOW_CONF:
                v.add(DECISION_REVIEW, "lang_low_conf")
    return v, lang
