from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

from logging import getLogger
logger = getLogger(__name__)

EXT_PATTERN = {
    "PYTHON": ["*.py"],
    "DOCUMENT": ["*.md", "*.mdx"],
}

@dataclass
class DocumentFile:
    path: str
    content: str


class DocumentPreprocessingAdapter:
    """
    Adapter for document preprocessing using LangChain.
    """
    def __init__(self) -> None:
        pass

    def preprocess_documents(
        self,
        path: str,
        language: str = None,
        chunk_size: int = 800,
        chunk_overlap: int = 200
    ) -> None:
        """
        Preprocess documents from the given path
        """
        if language is None:
            language = "DOCUMENT"
        language = language.upper()
        if language not in EXT_PATTERN.keys():
            raise ValueError(f"Unsupported language for document preprocessing: {language}")

        # Load documents from the given files path
        documents = self._load_documents(path, language)
        # get text splitter
        splitter = self._get_splitter(language, chunk_size, chunk_overlap)
        # Chunk documents
        chunks = self._chunk_documents(documents, splitter)
        return chunks

    def _load_documents(self, path: str, language: str = None) -> List[DocumentFile]:
        root = Path(path)
        doc_files: List[DocumentFile] = []

        patterns = EXT_PATTERN[language]

        for pattern in patterns:
            for path in root.glob(pattern):
                if any(part.startswith(".") for part in path.parts):
                    continue

                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    text = path.read_text(encoding="utf-8", errors="ignore")

                rel_path = str(path.relative_to(root))
                doc_files.append(DocumentFile(path=rel_path, content=text))

        logger.info(f"{len(doc_files)} python files loaded")

        return doc_files

    def _get_splitter(
        self,
        language: str = None,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
    ) -> RecursiveCharacterTextSplitter:
        """
        Get text splitter based on language
        """
        if language == "DOCUMENT":
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        splitter = RecursiveCharacterTextSplitter.from_language(
            language=getattr(Language, language),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        return splitter

    def _chunk_documents(
        self,
        document_files: List[DocumentFile],
        splitter: RecursiveCharacterTextSplitter,
    ) -> List[Dict[str, Any]]:
        """
        Create chunks from Document of Python code files
        """
        chunks: List[Dict[str, Any]] = []

        for code_file in document_files:
            split_texts = splitter.split_text(code_file.content)

            for idx, chunk_text in enumerate(split_texts):
                chunk = {
                    "id": f"{code_file.path}::chunk-{idx}",
                    "text": chunk_text,
                    "metadata": {
                        "source": code_file.path,
                        "chunk_index": idx,
                    },
                }
                chunks.append(chunk)

        logger.info(f"{len(chunks)} chunks created from documents")

        return chunks
