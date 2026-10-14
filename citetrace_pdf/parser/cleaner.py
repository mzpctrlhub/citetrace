from __future__ import annotations

import re
import unicodedata

# dvips Type-3 bitmap fonts often encode digits 1-8 as ASCII control chars 0x01-0x08
# and TeX ligatures (ff, fi, fl, ffi, ffl) as 0x0b-0x0f.
_DVIPS_BASE_MAP = {
    0x0B: "ff",
    0x0C: "fi",
    0x0E: "ffi",
    0x0F: "ffl",
}
_DVIPS_FULL_MAP = {i: ord("0") + i for i in range(1, 10)}
_DVIPS_FULL_MAP.update(_DVIPS_BASE_MAP)
_DVIPS_FULL_MAP[0x0D] = "fl"

# TeX CMEX10 (Computer Modern Math Extension) & Adobe Symbol PUA delimiter slots
# Note: Never map 0x09 (\t), 0x0A (\n), or 0x0D (\r)!
_CMEX_DELIMITER_MAP = {
    0x00: "(",
    0x01: ")",
    0x02: "[",
    0x03: "]",
    0x04: "⌊",
    0x05: "⌋",
    0x06: "⌈",
    0x07: "⌉",
    0x08: "{",
    0x10: "(",
    0x11: ")",
    0x12: "(",
    0x13: ")",
    0x14: "[",
    0x15: "]",
    0x1A: "{",
    0x1B: "}",
    # Adobe Symbol / Wingdings Private Use Area (PUA) math brackets & operators
    0xF8EB: "(",
    0xF8EC: "|",
    0xF8ED: "(",
    0xF8EE: "[",
    0xF8EF: "[",
    0xF8F0: "[",
    0xF8F6: ")",
    0xF8F7: "|",
    0xF8F8: ")",
    0xF8F9: "]",
    0xF8FA: "]",
    0xF8FB: "]",
    0xF0E5: "∑",
    0xF03D: "=",
}

_STRIKETHROUGH_SQRT = re.compile(r"_?\s*~~\s*(?:<u>)?\s*[√\ufffd]\s*(?:</u>)?\s*~~\s*_?")
_STRIKETHROUGH_UNDERSCORE = re.compile(r"(\w|</(?:sup|sub)>)\s*~~([a-zA-Z0-9])~~\s*")
_UNDERLINE_FRAC_ONE = re.compile(r"<u>1</u>\s*(_?\s*√\s*[_a-zA-Z0-9]+|\d+)")
_UNDERLINE_TAG = re.compile(r"</?u>")
_ADJACENT_SUPSUB = re.compile(r"</(sup|sub)>\s*<\1>")
_STACKED_FRAC_SQRT = re.compile(r"(?:\n[ \t]*1[ \t]*\n+[ \t]*|\b1[ \t]*\n+[ \t]*)(√)")

_MATH_ITALIC_DECIMAL = re.compile(r"(\d)\s*_([.,])_\s*(\d)")
_SPACED_DECIMAL = re.compile(r"(?<!\.)\b(\d+)\s+\.\s+(\d+)\b(?!\.)")
_MATH_ITALIC_PUNCT = re.compile(r"_([.,:;+\-*/=<>()\[\]{}±×÷∗·]+)_")
_ITALIC_PURE_NUMBER = re.compile(r"(?<![a-zA-Z0-9])_(\d+(?:\.\d+)?)_(?![a-zA-Z0-9])")
_FLAT_NEG_POW10 = re.compile(r"\b10[−-](\d+)\b")
_FLAT_DOT_POW10 = re.compile(r"(\d+(?:\.\d+)?\s*[·×∗]\s*10)(\d{2})\b")
_DVIPS_OPEN_QUOTE = re.compile(r"(?<![\\a-zA-Z])\\([A-Za-z]{2,}\b)")
_EMPTY_MARKDOWN_EMPH = re.compile(r"\*{4,}|(?<![\w*])\*\*[ \t\r\n]*\*\*(?![\w*])|(?<![\w_])__[ \t\r\n]*__(?![\w_])")
_CTRL_AND_PUA_STRIP = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ue000-\uf8ff\ufffd\ufffe\uffff]")

_COMPOUND_PREFIXES = {
    "self", "non", "pre", "post", "co", "inter", "intra", "semi", "auto",
    "over", "under", "cross", "counter", "source", "sequence", "position",
    "attention", "encoder", "feed", "word", "byte", "point", "long", "short",
    "real", "back", "state", "end", "high", "low", "fine", "large", "small",
    "single", "two", "three", "four", "one", "zero", "first", "second",
    "full", "half", "all", "well", "free", "open", "closed", "hard", "soft",
    "deep", "shallow", "fast", "slow", "task", "class", "domain", "time",
    "data", "model", "layer", "step", "tree", "phrase", "sentence", "character",
}
_COMPOUND_SUFFIXES = (
    "based", "wise", "aligned", "free", "like", "aware", "driven", "oriented",
    "specific", "independent", "dependent", "level", "scale", "dimensional",
    "layered", "headed", "gated", "connected", "weighted", "target", "source",
    "term", "time", "step", "task", "piece", "pair", "decoder", "encoder",
    "forward", "attention", "regressive", "supervised", "fitting", "tuning",
)


def clean_latex_math_italics(text: str) -> str:
    """Restore decimals, CMEX/PUA delimiters, strikethrough square roots/underscores, and math punctuation."""
    if not text:
        return ""
    text = text.translate(_CMEX_DELIMITER_MAP)
    if "~~" in text:
        text = _STRIKETHROUGH_SQRT.sub("√", text)
        text = _STRIKETHROUGH_UNDERSCORE.sub(r"\1_\2", text)
        text = _STRIKETHROUGH_UNDERSCORE.sub(r"\1_\2", text)
    if "<" in text:
        text = _UNDERLINE_FRAC_ONE.sub(r"1/\1", text)
        text = _UNDERLINE_TAG.sub("", text)
        text = _ADJACENT_SUPSUB.sub("", text)
    if "√" in text:
        text = _STACKED_FRAC_SQRT.sub(r" 1/\1", text)
    if "_" in text:
        text = _MATH_ITALIC_DECIMAL.sub(r"\1\2\3", text)
        text = _MATH_ITALIC_PUNCT.sub(r"\1", text)
        text = _ITALIC_PURE_NUMBER.sub(r"\1", text)
    text = _SPACED_DECIMAL.sub(r"\1.\2", text)
    if "10" in text:
        text = _FLAT_NEG_POW10.sub(r"10^-\1", text)
        text = _FLAT_DOT_POW10.sub(r"\1^\2", text)
    text = _EMPTY_MARKDOWN_EMPH.sub("", text)
    text = _CTRL_AND_PUA_STRIP.sub("", text)
    return text


def clean_dvips_text(text: str) -> str:
    """Translate dvips Type-3 control-character digits (0x01-0x09 -> 1-9), ligatures, quotes, and umlauts."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n")
    text = text.translate(_DVIPS_FULL_MAP)
    text = re.sub(r"\x7f\s*([aouAOU])", r"\1", text)
    text = _DVIPS_OPEN_QUOTE.sub(r'"\1', text)
    return normalize_text(text)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.translate(_DVIPS_BASE_MAP)
    text = (text.replace("\ufb00", "ff").replace("\ufb01", "fi").replace("\ufb02", "fl")
            .replace("\ufb03", "ffi").replace("\ufb04", "ffl").replace("\u00a0", " "))
    text = clean_latex_math_italics(text)
    return unicodedata.normalize("NFKC", text).strip()


class TextCleaner:
    HYPHENATED_LINE = re.compile(r"([a-zA-Z0-9]{2,})-\s*\n[\s\n]*([a-zA-Z0-9]{2,})")
    INLINE_HYPHEN = re.compile(r"([a-zA-Z0-9]{2,})-\s+([a-z]{2,})")
    _PROSE_TAIL_RE = re.compile(r"[a-z]{2,},?$")
    _PROSE_HEAD_RE = re.compile(r"^_?[a-z]{2,}\b(?!\s*=)")

    @classmethod
    def _join_hyphen(cls, m: re.Match) -> str:
        left, right = m.group(1), m.group(2)
        if not right[0].islower():
            return f"{left}-{right}"
        l_low, r_low = left.lower(), right.lower()
        if l_low in _COMPOUND_PREFIXES or r_low.startswith(_COMPOUND_SUFFIXES):
            return f"{left}-{right}"
        return f"{left}{right}"

    @classmethod
    def dehyphenate(cls, text: str) -> str:
        return cls.HYPHENATED_LINE.sub(cls._join_hyphen, text)

    @classmethod
    def stitch_broken_paragraphs(cls, paragraphs: list[str]) -> list[str]:
        if len(paragraphs) < 2:
            return paragraphs
        merged: list[str] = [paragraphs[0]]
        for p in paragraphs[1:]:
            prev = merged[-1]
            if (
                len(prev) >= 45
                and not prev.startswith(("#", "|", ">", "Table ", "Figure "))
                and not p.startswith(("#", "|", ">", "Table ", "Figure "))
                and cls._PROSE_TAIL_RE.search(prev)
                and cls._PROSE_HEAD_RE.match(p)
            ):
                merged[-1] = f"{prev} {p}"
            else:
                merged.append(p)
        return merged

    @classmethod
    def clean(cls, text: str) -> str:
        if not text:
            return ""
        text = clean_latex_math_italics(text)
        text = _DVIPS_OPEN_QUOTE.sub(r'"\1', text)
        text = cls.dehyphenate(text)
        text = cls.INLINE_HYPHEN.sub(cls._join_hyphen, text)
        text = re.sub(r"(?m)^\s*#{1,6}\s*$", "", text)
        text = re.sub(r"\n\s*\d+\s*\n", "\n\n", text)
        paragraphs = []
        for paragraph in text.split("\n\n"):
            paragraph = paragraph.strip()
            if paragraph:
                paragraphs.append(paragraph if paragraph.startswith("|") else re.sub(r"\s*\n\s*", " ", paragraph))
        paragraphs = cls.stitch_broken_paragraphs(paragraphs)
        return re.sub(r"[ \t]+", " ", "\n\n".join(paragraphs)).strip()


