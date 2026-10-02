from pathlib import Path
from typing import List, Any

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    JSONLoader
)

from langchain_community.document_loaders.excel import UnstructuredExcelLoader


def Load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from data directory
    and convert them into LangChain Document structures.

    Supported:
    PDF, TXT, CSV, EXCEL, WORD, JSON
    """

    # Use project root data folder
    data_path = Path(data_dir).resolve()
    documents = []

    # ---------------- PDF FILES ----------------
    pdf_files = list(data_path.glob("**/*.pdf"))

    print(
        f"[DEBUG] Found {len(pdf_files)} PDF files: "
        f"{[str(f) for f in pdf_files]}"
    )

    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF: {pdf_file}")

        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()

            print(
                f"[DEBUG] Loaded {len(loaded)} PDF documents "
                f"from {pdf_file}"
            )

            documents.extend(loaded)

        except Exception as e:
            print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")

    # ---------------- TEXT FILES ----------------
    text_files = list(data_path.glob("**/*.txt"))

    print(
        f"[DEBUG] Found {len(text_files)} TEXT files: "
        f"{[str(f) for f in text_files]}"
    )

    for text_file in text_files:
        print(f"[DEBUG] Loading TEXT: {text_file}")

        try:
            loader = TextLoader(str(text_file))
            loaded = loader.load()

            print(
                f"[DEBUG] Loaded {len(loaded)} TEXT documents "
                f"from {text_file}"
            )

            documents.extend(loaded)

        except Exception as e:
            print(f"[ERROR] Failed to load TEXT file {text_file}: {e}")

    # Return all loaded documents
    return documents