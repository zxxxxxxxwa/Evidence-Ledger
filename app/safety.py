from __future__ import annotations

import re
from typing import Any


INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore_instructions",
        re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.I),
    ),
    ("system_prompt", re.compile(r"(reveal|print|show).{0,30}system\s+prompt", re.I | re.S)),
    ("role_override", re.compile(r"you\s+are\s+now|act\s+as\s+(the\s+)?system", re.I)),
    ("zh_ignore", re.compile(r"忽略.{0,12}(之前|此前|以上|前面).{0,12}(指令|要求|规则)")),
    ("zh_system", re.compile(r"(显示|泄露|输出).{0,12}(系统提示|系统指令|密钥)")),
)

PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
    ("phone", re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")),
    ("student_number", re.compile(r"(?:学号|student\s*id)\s*[:：]?\s*[A-Z0-9-]{5,}", re.I)),
)


def scan_text(text: str) -> list[dict[str, Any]]:

    flags: list[dict[str, Any]] = []
    for code, pattern in INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            flags.append(
                {
                    "code": f"PROMPT_INJECTION_{code.upper()}",
                    "severity": "high",
                    "offset": match.start(),
                }
            )
    for code, pattern in PII_PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            flags.append(
                {
                    "code": f"POSSIBLE_PII_{code.upper()}",
                    "severity": "warning",
                    "count": len(matches),
                }
            )
    return flags


def has_high_risk_flag(flags: list[dict[str, Any]]) -> bool:
    return any(item.get("severity") == "high" for item in flags)
