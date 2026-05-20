from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pdfplumber
from loguru import logger

from config import settings


@dataclass
class PageText:
    """Texte extrait d'une page physique (numérotation 1-based)."""

    page_number: int
    text: str
    extraction_mode: str  # "pdfplumber" | "ocr_tesseract"


class PDFExtractor:
    """
    Extrait le texte page par page d'un PDF.

    Stratégie :
    1. pdfplumber d'abord (rapide, fidèle pour les PDFs text-natif)
    2. fallback OCR (pytesseract + pdf2image) page par page si pdfplumber
       renvoie moins de `OCR_FALLBACK_MIN_CHARS` caractères pour la page.

    Les dépendances OCR sont importées paresseusement : si pytesseract/pdf2image
    ne sont pas installés, le fallback est désactivé et un warning est loggué.
    """

    def __init__(
        self,
        force_ocr: bool = False,
        ocr_lang: Optional[str] = None,
        ocr_dpi: Optional[int] = None,
        ocr_fallback_min_chars: Optional[int] = None,
    ) -> None:
        self.force_ocr = force_ocr
        self.ocr_lang = ocr_lang or settings.OCR_LANG
        self.ocr_dpi = ocr_dpi or settings.OCR_DPI
        self.ocr_fallback_min_chars = (
            ocr_fallback_min_chars
            if ocr_fallback_min_chars is not None
            else settings.OCR_FALLBACK_MIN_CHARS
        )

        self._ocr_ready: Optional[bool] = None  # détection paresseuse

    def _ensure_ocr(self) -> bool:
        """Vérifie que pytesseract + pdf2image sont importables (lazy)."""
        if self._ocr_ready is not None:
            return self._ocr_ready
        try:
            import pytesseract  # noqa: F401
            from pdf2image import convert_from_path  # noqa: F401
            from PIL import Image  # noqa: F401

            if settings.TESSERACT_CMD:
                import pytesseract as _pt

                _pt.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
            if settings.TESSDATA_DIR:
                # Tesseract lit TESSDATA_PREFIX au démarrage. On le pose dans
                # l'environnement du process pour qu'il pointe sur notre dossier
                # user-local (évite les soucis de droits sur Program Files).
                import os

                os.environ["TESSDATA_PREFIX"] = settings.TESSDATA_DIR
            self._ocr_ready = True
        except ImportError as ex:
            logger.warning(
                f"[PDFExtractor] OCR désactivé (modules manquants : {ex}). "
                "Installer le groupe `experimental` : `uv sync --group experimental`."
            )
            self._ocr_ready = False
        return self._ocr_ready

    def _extract_with_pdfplumber(self, pdf_path: Path) -> list[PageText]:
        out: list[PageText] = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                out.append(
                    PageText(page_number=i, text=text, extraction_mode="pdfplumber")
                )
        return out

    def _extract_page_with_ocr(self, pdf_path: Path, page_number: int) -> str:
        from pdf2image import convert_from_path
        from PIL import Image  # noqa: F401
        import pytesseract

        kwargs: dict = {"dpi": self.ocr_dpi, "first_page": page_number, "last_page": page_number}
        if settings.POPPLER_PATH:
            kwargs["poppler_path"] = settings.POPPLER_PATH
        images = convert_from_path(str(pdf_path), **kwargs)
        if not images:
            return ""
        text = pytesseract.image_to_string(images[0], lang=self.ocr_lang)
        # Recolle les mots coupés en fin de ligne par un trait d'union
        return text.replace("-\n", "")

    def extract_pages(self, pdf_path: Path) -> list[PageText]:
        """Retourne la liste des pages (numérotation physique 1-based)."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)

        if self.force_ocr:
            return self._extract_all_with_ocr(pdf_path)

        pages = self._extract_with_pdfplumber(pdf_path)
        ocr_ready = None

        for idx, page in enumerate(pages):
            if len(page.text.strip()) >= self.ocr_fallback_min_chars:
                continue
            # Texte trop court → tentative OCR
            if ocr_ready is None:
                ocr_ready = self._ensure_ocr()
            if not ocr_ready:
                logger.debug(
                    f"[PDFExtractor] {pdf_path.name} p.{page.page_number}: "
                    f"texte court ({len(page.text)} ch) mais OCR indisponible."
                )
                continue
            try:
                ocr_text = self._extract_page_with_ocr(pdf_path, page.page_number)
                if len(ocr_text.strip()) > len(page.text.strip()):
                    pages[idx] = PageText(
                        page_number=page.page_number,
                        text=ocr_text,
                        extraction_mode="ocr_tesseract",
                    )
                    logger.info(
                        f"[PDFExtractor] {pdf_path.name} p.{page.page_number}: OCR utilisé"
                    )
            except Exception as ex:
                logger.warning(
                    f"[PDFExtractor] {pdf_path.name} p.{page.page_number}: OCR échoué ({ex})"
                )

        return pages

    def _extract_all_with_ocr(self, pdf_path: Path) -> list[PageText]:
        if not self._ensure_ocr():
            raise RuntimeError(
                "OCR forcé mais pytesseract/pdf2image indisponibles. "
                "Installer le groupe `experimental`."
            )
        from pdf2image import convert_from_path
        import pytesseract

        kwargs: dict = {"dpi": self.ocr_dpi}
        if settings.POPPLER_PATH:
            kwargs["poppler_path"] = settings.POPPLER_PATH

        images = convert_from_path(str(pdf_path), **kwargs)
        out: list[PageText] = []
        for i, img in enumerate(images, start=1):
            text = pytesseract.image_to_string(img, lang=self.ocr_lang).replace("-\n", "")
            out.append(PageText(page_number=i, text=text, extraction_mode="ocr_tesseract"))
        return out
