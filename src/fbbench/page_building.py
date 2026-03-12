from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from concurrent.futures import ProcessPoolExecutor, as_completed

import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
from tqdm import tqdm
from PIL import Image

from fbbench.utils.io_utils import ensure_dir


@dataclass
class PageInfo:
    page_uid: str
    doc_name: str
    page_id: int
    page_image_path: str
    page_text: str


def render_pdf_to_images(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 150,
    page_ids: Optional[List[int]] = None,
) -> List[Path]:
    """
    将单个 PDF 渲染为若干页图像，返回每一页图像路径列表。

    - 若 page_ids 为 None，则渲染整本 PDF。
    - 若提供了 page_ids，则只渲染给定页码。
    """
    ensure_dir(output_dir)
    doc = fitz.open(pdf_path)
    image_paths: List[Path] = []

    if page_ids is None:
        iter_ids = range(len(doc))
    else:
        # 去重并排序，避免重复渲染
        iter_ids = sorted(set(int(i) for i in page_ids))

    for page_id in iter_ids:
        if page_id < 0 or page_id >= len(doc):
            continue
        page = doc.load_page(page_id)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat)
        out_path = output_dir / f"{page_id}.png"
        pix.save(out_path.as_posix())
        image_paths.append(out_path)

    doc.close()
    return image_paths


def extract_text_with_pdfplumber(pdf_path: Path, page_id: int) -> str:
    """仅使用 pdfplumber 提取指定页文本，不再进行 OCR。"""
    with pdfplumber.open(pdf_path) as pdf:
        if page_id >= len(pdf.pages):
            return ""
        page = pdf.pages[page_id]
        text = page.extract_text() or ""
    return text


def build_pages_for_pdf(
    pdf_path: Path,
    doc_name: str,
    image_root: Path,
    min_char_threshold: int = 30,
    dpi: int = 150,
    ocr_lang: str = "eng",
    page_ids: Optional[List[int]] = None,
) -> List[PageInfo]:
    """
    对单个 PDF 构建 PageInfo。

    - 若 page_ids 为 None，则构建该文档的所有页面；
    - 若提供了 page_ids，则仅构建这些证据页。
    """
    pdf_image_dir = image_root / doc_name
    image_paths = render_pdf_to_images(pdf_path, pdf_image_dir, dpi=dpi, page_ids=page_ids)

    pages: List[PageInfo] = []
    if page_ids is None:
        # 渲染了所有页面，按 0..N-1 的顺序对应
        for page_id, image_path in enumerate(image_paths):
            text = extract_text_with_pdfplumber(pdf_path, page_id)
            page_uid = f"{doc_name}_{page_id}"
            pages.append(
                PageInfo(
                    page_uid=page_uid,
                    doc_name=doc_name,
                    page_id=page_id,
                    page_image_path=str(image_path),
                    page_text=text,
                )
            )
    else:
        # 只渲染了指定页面，按 page_ids 与 image_paths 对应
        norm_ids = sorted(set(int(i) for i in page_ids))
        for page_id, image_path in zip(norm_ids, image_paths):
            text = extract_text_with_pdfplumber(pdf_path, page_id)
            page_uid = f"{doc_name}_{page_id}"
            pages.append(
                PageInfo(
                    page_uid=page_uid,
                    doc_name=doc_name,
                    page_id=page_id,
                    page_image_path=str(image_path),
                    page_text=text,
                )
            )
    return pages


def _build_and_save_single_pdf(
    doc_name: str,
    pdf_path: Path,
    image_root: Path,
    per_doc_dir: Path,
    min_char_threshold: int,
    dpi: int,
    pages_to_keep: Optional[List[int]] = None,
) -> Path:
    """构建单个 PDF 的页面并增量写入到独立 parquet 文件中。"""
    pages = build_pages_for_pdf(
        pdf_path=pdf_path,
        doc_name=doc_name,
        image_root=image_root,
        min_char_threshold=min_char_threshold,
        dpi=dpi,
        page_ids=pages_to_keep,
    )
    df = pd.DataFrame(
        [
            {
                "page_uid": p.page_uid,
                "doc_name": p.doc_name,
                "page_id": p.page_id,
                "page_image_path": p.page_image_path,
                "page_text": p.page_text,
            }
            for p in pages
        ]
    )
    ensure_dir(per_doc_dir)
    out_path = per_doc_dir / f"{doc_name}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def build_pages_for_all_pdfs(
    pdf_dir: Path,
    image_root: Path,
    min_char_threshold: int = 30,
    dpi: int = 150,
    doc_name_filter: Optional[List[str]] = None,
    num_workers: int = 1,
    per_doc_dir: Optional[Path] = None,
    evidence_pages_by_doc: Optional[Dict[str, List[int]]] = None,
) -> pd.DataFrame:
    """
    对目录下所有 PDF 构建页面表。

    - pdf_dir: 包含所有 PDF 文件的目录。
    - image_root: 页面图像输出根目录。
    - doc_name_filter: 若提供，则只处理给定 doc_name 列表。
    - per_doc_dir: 增量写入的单文档 parquet 目录；若为 None，则默认使用 image_root 上级的 pages_by_doc 目录。
    """
    if per_doc_dir is None:
        per_doc_dir = image_root.parent / "pages_by_doc"
    ensure_dir(per_doc_dir)

    pdf_paths: List[Tuple[str, Path]] = []
    for pdf_path in pdf_dir.glob("*.pdf"):
        doc_name = pdf_path.stem
        if doc_name_filter is not None and doc_name not in doc_name_filter:
            continue
        pdf_paths.append((doc_name, pdf_path))

    # 断点续跑：跳过已经存在 parquet 的文档
    done_docs = {p.stem for p in per_doc_dir.glob("*.parquet")}
    todo_paths = [(doc_name, path) for doc_name, path in pdf_paths if doc_name not in done_docs]
    sorted_paths = sorted(todo_paths)

    if sorted_paths:
        if num_workers is None or num_workers <= 1:
            for doc_name, pdf_path in tqdm(sorted_paths, desc="Building pages (serial)"):
                pages_to_keep = None
                if evidence_pages_by_doc is not None:
                    pages_to_keep = evidence_pages_by_doc.get(doc_name)
                _build_and_save_single_pdf(
                    doc_name=doc_name,
                    pdf_path=pdf_path,
                    image_root=image_root,
                    per_doc_dir=per_doc_dir,
                    min_char_threshold=min_char_threshold,
                    dpi=dpi,
                    pages_to_keep=pages_to_keep,
                )
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        _build_and_save_single_pdf,
                        doc_name,
                        pdf_path,
                        image_root,
                        per_doc_dir,
                        min_char_threshold,
                        dpi,
                        evidence_pages_by_doc.get(doc_name) if evidence_pages_by_doc is not None else None,
                    ): (doc_name, pdf_path)
                    for doc_name, pdf_path in sorted_paths
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc="Building pages (parallel)"):
                    _ = future.result()

    # 合并所有 per-doc parquet 为一个 DataFrame
    per_doc_files = sorted(per_doc_dir.glob("*.parquet"))
    if not per_doc_files:
        return pd.DataFrame(
            columns=["page_uid", "doc_name", "page_id", "page_image_path", "page_text"]
        )
    dfs = [pd.read_parquet(p) for p in per_doc_files]
    df = pd.concat(dfs, ignore_index=True)
    return df

