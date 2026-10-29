"""Application settings.

All paths are resolved relative to the project root so the backend works no matter
which directory Uvicorn or pytest is launched from. Values can be overridden with
environment variables (see ``.env.example``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines) so we do not need python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(PROJECT_ROOT / ".env")
# joblib/loky otherwise shells out to count physical cores, which fails in some Windows sandboxes.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))


def _path(env_key: str, default: Path) -> Path:
    value = os.environ.get(env_key)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass
class Settings:
    app_name: str = "PneumoCare API"
    version: str = "1.0.0"
    data_dir: Path = field(default_factory=lambda: _path("PNEUMOCARE_DATA_DIR", PROJECT_ROOT / "data"))
    models_dir: Path = field(default_factory=lambda: _path("PNEUMOCARE_MODELS_DIR", PROJECT_ROOT / "models"))
    reports_dir: Path = field(default_factory=lambda: _path("PNEUMOCARE_REPORTS_DIR", PROJECT_ROOT / "reports"))
    uploads_dir: Path = field(default_factory=lambda: _path("PNEUMOCARE_UPLOADS_DIR",
                                                           PROJECT_ROOT / "data" / "uploads"))
    database_url: str = field(default_factory=lambda: os.environ.get("PNEUMOCARE_DATABASE_URL", ""))
    cors_origins: list[str] = field(
        default_factory=lambda: os.environ.get(
            "PNEUMOCARE_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")
    )
    max_upload_mb: float = float(os.environ.get("PNEUMOCARE_MAX_UPLOAD_MB", "10"))
    device: str = os.environ.get("PNEUMOCARE_DEVICE", "cpu")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def sample_dir(self) -> Path:
        return self.data_dir / "sample"

    @property
    def resolved_database_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_dir / 'pneumocare.db').as_posix()}"

    def ensure_dirs(self) -> None:
        for path in (self.raw_dir, self.processed_dir, self.sample_dir, self.uploads_dir, self.models_dir,
                     self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
