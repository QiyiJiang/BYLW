import argparse
from pathlib import Path
from typing import Dict, List, Set

from fbbench.page_building import build_pages_for_all_pdfs
from fbbench.utils.config_loader import load_yaml_with_env
from fbbench.utils.io_utils import ensure_dir, write_table, read_jsonl
from fbbench.utils.logging_utils import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Build page table: images + texts for evidence pages only.")
    parser.add_argument("--paths-config", type=str, default="config/paths.yaml")
    parser.add_argument("--pdf-dir", type=str, default=None, help="Override PDF directory.")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--min-char-threshold", type=int, default=30)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="并行处理的进程数（>1 启用多进程）",
    )
    args = parser.parse_args()

    logger = setup_logger("build_pages")

    paths_cfg = load_yaml_with_env(Path(args.paths_config))
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else Path(paths_cfg["pdf"]["pdf_dir"])
    image_root = Path(paths_cfg["pdf"]["page_image_dir"])
    pages_path = Path(paths_cfg["tables"]["pages"])
    ensure_dir(image_root)
    ensure_dir(pages_path.parent)
    per_doc_dir = pages_path.parent / "pages_by_doc"

    # 从 FinanceBench 开放问题集中抽取：
    # 1) 实际用到的 doc_name 列表（只处理这些 PDF）
    # 2) 每个 doc_name 对应的 evidence_page_num 列表（只构建这些证据页）
    fb_jsonl_path = Path("data/raw/financebench_open_source.jsonl")
    doc_names: Set[str] = set()
    evidence_pages_by_doc: Dict[str, Set[int]] = {}
    if fb_jsonl_path.exists():
        logger.info("Loading FinanceBench open-source annotations from %s", fb_jsonl_path)
        records = read_jsonl(fb_jsonl_path)
        for rec in records:
            base_doc = rec.get("doc_name")
            if base_doc:
                doc_names.add(base_doc)
            for ev in rec.get("evidence", []):
                if not isinstance(ev, dict):
                    continue
                ev_doc = ev.get("doc_name")
                page_num = ev.get("evidence_page_num")
                if ev_doc:
                    doc_names.add(ev_doc)
                if ev_doc and page_num is not None:
                    try:
                        page_idx = int(page_num)
                    except (TypeError, ValueError):
                        continue
                    evidence_pages_by_doc.setdefault(ev_doc, set()).add(page_idx)

    doc_name_filter: List[str] = sorted(doc_names)
    logger.info(
        "Building pages for PDFs in %s (PDF text only, no OCR, num_workers=%d, filtered docs=%d, per-doc dir=%s; evidence-only pages)",
        pdf_dir,
        args.num_workers,
        len(doc_name_filter),
        per_doc_dir,
    )

    df = build_pages_for_all_pdfs(
        pdf_dir=pdf_dir,
        image_root=image_root,
        min_char_threshold=args.min_char_threshold,
        dpi=args.dpi,
        num_workers=args.num_workers,
        doc_name_filter=doc_name_filter,
        per_doc_dir=per_doc_dir,
        evidence_pages_by_doc={k: sorted(v) for k, v in evidence_pages_by_doc.items()},
    )
    logger.info("Total evidence pages built: %d", len(df))

    write_table(df, pages_path)
    logger.info("Pages table saved to %s", pages_path)


if __name__ == "__main__":
    main()

