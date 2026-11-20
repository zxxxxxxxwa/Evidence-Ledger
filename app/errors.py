from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AppError(Exception):

    code: str
    status: int = 400
    message: str = ""
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        return self.message or self.code

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"error": {"code": self.code}}
        if self.details:
            payload["error"]["details"] = self.details
        return payload
