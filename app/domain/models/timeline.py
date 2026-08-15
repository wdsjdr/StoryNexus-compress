"""时序倒排索引契约：CSA 主干(A) 的事实三元组。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class FactTriple(BaseModel):
    """(主语, 谓语, 宾语) + 章号，保证时间顺序不被打乱。"""

    subject: str
    predicate: str
    object: str
    chapter_no: int = Field(gt=0)

    def text(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"
