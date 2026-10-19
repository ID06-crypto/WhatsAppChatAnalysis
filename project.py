"""Projects: one folder per imported chat, holding the original export and saved graph recipes.

projects/<name>/chat.txt      the WhatsApp export, copied in
projects/<name>/graphs.json   list of saved graph recipes (settings, not images)
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from whatsapp_analysis import Chat, parse_chat

PROJECTS_DIR = Path(__file__).parent / "projects"


def _slug(name: str) -> str:
    slug = re.sub(r"[^\w\- ]+", "", name).strip()
    if not slug:
        raise ValueError("Project name needs at least one letter or number.")
    return slug


class Project:
    def __init__(self, name: str):
        self.name = name
        self.path = PROJECTS_DIR / name
        if not (self.path / "chat.txt").exists():
            raise FileNotFoundError(f"No project named '{name}'")
        self._chat: Chat | None = None

    # -- lifecycle
    @classmethod
    def create(cls, name: str, chat_bytes: bytes, dayfirst: bool | None = None) -> "Project":
        """Create a project from an uploaded export. Validates the file before keeping it."""
        name = _slug(name)
        path = PROJECTS_DIR / name
        if path.exists():
            raise FileExistsError(f"A project called '{name}' already exists.")
        path.mkdir(parents=True)
        try:
            (path / "chat.txt").write_bytes(chat_bytes)
            parse_chat(path / "chat.txt", dayfirst=dayfirst)  # raises if it isn't a chat export
        except Exception:
            shutil.rmtree(path, ignore_errors=True)
            raise
        (path / "meta.json").write_text(json.dumps({"dayfirst": dayfirst}))
        return cls(name)

    @staticmethod
    def list_all() -> list[str]:
        if not PROJECTS_DIR.exists():
            return []
        return sorted(p.name for p in PROJECTS_DIR.iterdir() if (p / "chat.txt").exists())

    def delete(self) -> None:
        shutil.rmtree(self.path)

    # -- data
    @property
    def chat(self) -> Chat:
        if self._chat is None:
            meta = self.path / "meta.json"
            dayfirst = json.loads(meta.read_text()).get("dayfirst") if meta.exists() else None
            self._chat = parse_chat(self.path / "chat.txt", dayfirst=dayfirst)
        return self._chat

    # -- saved graphs
    @property
    def graphs(self) -> list[dict]:
        f = self.path / "graphs.json"
        return json.loads(f.read_text()) if f.exists() else []

    def save_graph(self, recipe: dict) -> None:
        """Add a recipe, replacing any existing graph with the same name."""
        graphs = [g for g in self.graphs if g["name"] != recipe["name"]] + [recipe]
        (self.path / "graphs.json").write_text(json.dumps(graphs, indent=2))

    def delete_graph(self, name: str) -> None:
        graphs = [g for g in self.graphs if g["name"] != name]
        (self.path / "graphs.json").write_text(json.dumps(graphs, indent=2))
