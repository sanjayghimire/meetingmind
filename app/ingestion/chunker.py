import re
from dataclasses import dataclass, field
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.config import get_settings

settings = get_settings()


@dataclass
class Chunk:
    text: str
    source_id: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(idx={self.chunk_index}, '{preview}...')"


class MeetingChunker:

    SPEAKER_PATTERN = re.compile(
        r"(?:\[[\d:]+\]\s*)?([A-Za-z][A-Za-z\s]{1,30}):\s*(.*)"
    )

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, text: str, source_id: str,
              metadata: dict = None) -> List[Chunk]:
        meta = metadata or {}
        if self._looks_like_transcript(text):
            return self._chunk_transcript(text, source_id, meta)
        return self._chunk_plain(text, source_id, meta)

    def _looks_like_transcript(self, text: str) -> bool:
        matches = self.SPEAKER_PATTERN.findall(text[:2000])
        return len(matches) >= 3

    def _chunk_transcript(self, text: str, source_id: str,
                          meta: dict) -> List[Chunk]:
        lines = text.strip().split("\n")
        turns = []
        for line in lines:
            m = self.SPEAKER_PATTERN.match(line.strip())
            if m:
                turns.append({
                    "speaker": m.group(1).strip(),
                    "text": m.group(2).strip()
                })
            elif turns:
                turns[-1]["text"] += " " + line.strip()

        chunks = []
        current_text = ""
        current_speakers = set()
        idx = 0

        for turn in turns:
            addition = f"{turn['speaker']}: {turn['text']}\n"
            if (current_text and
                    len(current_text) + len(addition) > settings.chunk_size):
                chunks.append(Chunk(
                    text=current_text.strip(),
                    source_id=source_id,
                    chunk_index=idx,
                    metadata={**meta, "speakers": list(current_speakers)},
                ))
                idx += 1
                current_text = addition
                current_speakers = {turn["speaker"]}
            else:
                current_text += addition
                current_speakers.add(turn["speaker"])

        if current_text.strip():
            chunks.append(Chunk(
                text=current_text.strip(),
                source_id=source_id,
                chunk_index=idx,
                metadata={**meta, "speakers": list(current_speakers)},
            ))

        return chunks

    def _chunk_plain(self, text: str, source_id: str,
                     meta: dict) -> List[Chunk]:
        raw_chunks = self.splitter.split_text(text)
        return [
            Chunk(text=c, source_id=source_id,
                  chunk_index=i, metadata=meta)
            for i, c in enumerate(raw_chunks)
            if c.strip()
        ]