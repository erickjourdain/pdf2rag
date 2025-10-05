import re
from typing import List, Dict

class MarkdownPreprocessor:
    def __init__(self, chunk_size: int = 1000, overlap: int = 100, max_heading_level: int = 3):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_heading_level = max_heading_level

    # -------------------------
    #  Parsing & nettoyage
    # -------------------------
    def parse_and_clean(self, markdown: str, source: str = "unknown") -> Dict:
        def strip_frontmatter(md: str) -> str:
            return re.sub(r"^---[\s\S]*?---\n+", "", md, flags=re.MULTILINE)

        def strip_code_fences(md: str) -> str:
            md = re.sub(r"```[\s\S]*?```", "", md)  # blocs de code
            md = re.sub(r"`([^`]+)`", r"\1", md)   # inline code
            return md

        def strip_html(md: str) -> str:
            return re.sub(r"<[^>]+>", " ", md)

        def normalize_links(md: str) -> str:
            return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", md)

        def collapse_spaces(md: str) -> str:
            md = md.replace("\u00A0", " ")
            md = re.sub(r"[ \t]+", " ", md)
            md = re.sub(r"\n{3,}", "\n\n", md)
            return md.strip()

        def convert_tables(md: str) -> str:
            lines = md.splitlines()
            out = []
            i = 0

            def is_table_row(s: str) -> bool:
                return "|" in s and s.count("|") >= 2

            while i < len(lines):
                line = lines[i]
                if (is_table_row(line) and i + 1 < len(lines)
                    and re.match(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", lines[i+1])):
                    # Début d'un tableau
                    table_lines = [line]
                    i += 1  # skip separator
                    table_lines.append(lines[i])
                    i += 1
                    while i < len(lines) and is_table_row(lines[i]):
                        table_lines.append(lines[i])
                        i += 1
                    # Conversion en TSV
                    tsv = []
                    for L in table_lines:
                        row = L.strip().lstrip("|").rstrip("|")
                        row = re.split(r"\s*\|\s*", row)
                        tsv.append("\t".join(row))
                    out.append("\nTableau:\n" + "\n".join(tsv) + "\n")
                    continue
                out.append(line)
                i += 1
            return "\n".join(out)

        md = markdown or ""
        md = strip_frontmatter(md)
        md = strip_code_fences(md)
        md = strip_html(md)
        md = normalize_links(md)
        md = convert_tables(md)
        md = collapse_spaces(md)

        return {"cleaned": md, "source": source}

    # -------------------------
    #  Chunking
    # -------------------------
    def chunk_with_headings(self, text: str, source: str = "unknown") -> List[Dict]:
        header_re = re.compile(r"^(#{1,6})\s+(.*)$")

        def split_by_headings(md: str):
            lines = md.splitlines()
            sections = []
            current_title = "root"
            current_lines: List[str] = []

            def flush():
                txt = "\n".join(current_lines).strip()
                if txt:
                    sections.append({"title": current_title, "text": txt})

            for line in lines:
                m = header_re.match(line.strip())
                if m and len(m.group(1)) <= self.max_heading_level:
                    flush()
                    current_title = m.group(2).strip()
                    current_lines = []
                else:
                    current_lines.append(line)
            flush()
            return sections

        def chunk_text(t: str) -> List[str]:
            chunks: List[str] = []
            start = 0
            n = len(t)

            size, ov = self.chunk_size, self.overlap
            if size <= 0:
                size = 1000
            if ov < 0:
                ov = 0
            if ov >= size:
                ov = max(0, size // 5)

            while start < n:
                end = min(start + size, n)
                slice_ = t[start:end]

                # couper à la fin d'une phrase si possible
                last_dot = slice_.rfind(". ")
                if last_dot > int(size * 0.6) and end < n:
                    end = start + last_dot + 1
                    slice_ = t[start:end]

                slice_ = slice_.strip()
                if slice_:
                    chunks.append(slice_)

                if end >= n:
                    break
                start = max(0, end - ov)
            return chunks

        sections = split_by_headings(text or "")
        out: List[Dict] = []
        global_index = 0

        for sec in sections:
            parts = chunk_text(sec["text"])
            for i, chunk in enumerate(parts):
                out.append({
                    "text": chunk,
                    "meta": {
                        "section": sec["title"],
                        "chunk_index": i,
                        "global_index": global_index,
                        "source": source,
                    },
                })
                global_index += 1
        return out

    # -------------------------
    #  Pipeline complet
    # -------------------------
    def process(self, markdown: str, source: str = "unknown") -> List[Dict]:
        """Parse + clean + chunk en un seul appel"""
        cleaned = self.parse_and_clean(markdown, source)
        return self.chunk_with_headings(cleaned["cleaned"], source)

