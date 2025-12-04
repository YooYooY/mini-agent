from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader

import pytesseract
import pdf2image

from src.core.config import TESSERACT_CMD


class PDFParser:
    """
    PDF 文档解析器
    - 先用 PyPDFLoader（文字版 PDF）
    - 不行则 fallback 到 Tesseract OCR（扫描版 PDF）
    """

    def __init__(self):
        # pytesseract 会自动使用本机的 tesseract
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    def parse(self, file_path: Path) -> List[Document]:
        # 1) 尝试用 PyPDFLoader
        docs = self._try_load_text_pdf(file_path)
        if docs:
            return docs

        # 2) fallback 到 OCR
        return self._ocr_pdf(file_path)

    def _try_load_text_pdf(self, file_path: Path):
        try:
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()

            if any(d.page_content.strip() for d in docs):
                print("使用 PyPDFLoader（文字版 PDF）")
                return docs

            return []
        except:
            return []

    def _ocr_pdf(self, file_path: Path) -> List[Document]:
        print("使用 Tesseract OCR 解析扫描 PDF...")

        pages = pdf2image.convert_from_path(str(file_path))
        full_text = ""

        for img in pages:
            # OCR 识别中文需要指定语言 chi_sim
            text = pytesseract.image_to_string(img, lang="chi_sim")
            full_text += text + "\n"

        if not full_text.strip():
            raise ValueError("OCR 未识别到有效文本")

        return [Document(page_content=full_text)]
