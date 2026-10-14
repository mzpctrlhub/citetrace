from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MetadataCache:
    ARTIFACT_SCHEMA_VERSION = 2
    PARSER_VERSION = "2.1.0"
    CHUNKER_VERSION = "1.0.0"
    TITLE_EXTRACTOR_VERSION = "1.1.0"
    REFERENCE_EXTRACTOR_VERSION = "1.0.0"

    def __init__(
        self,
        parser_version: str = PARSER_VERSION,
        chunker_version: str = CHUNKER_VERSION,
        artifact_schema_version: int = ARTIFACT_SCHEMA_VERSION,
        title_extractor_version: str = TITLE_EXTRACTOR_VERSION,
        reference_extractor_version: str = REFERENCE_EXTRACTOR_VERSION,
    ) -> None:
        self.parser_version = parser_version
        self.chunker_version = chunker_version
        self.artifact_schema_version = artifact_schema_version
        self.title_extractor_version = title_extractor_version
        self.reference_extractor_version = reference_extractor_version

    @property
    def version_fields(self) -> dict[str, Any]:
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "parser_version": self.parser_version,
            "chunker_version": self.chunker_version,
            "title_extractor_version": self.title_extractor_version,
            "reference_extractor_version": self.reference_extractor_version,
        }

    def read_valid(self, metadata_path: Path) -> dict[str, Any] | None:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            metadata.get("artifact_schema_version") != self.artifact_schema_version
            or metadata.get("parser_version") != self.parser_version
            or not metadata.get("paper_title")
            or not metadata.get("outline")
        ):
            return None
        return metadata
