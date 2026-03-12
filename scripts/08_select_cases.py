import argparse
from pathlib import Path

from fbbench.analysis.cases_selection import select_cases
from fbbench.utils.config_loader import load_yaml_with_env
from fbbench.utils.io_utils import ensure_dir
from fbbench.utils.logging_utils import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Select typical cases for qualitative analysis.")
    parser.add_argument("--paths-config", type=str, default="config/paths.yaml")
    args = parser.parse_args()

    logger = setup_logger("select_cases")

    paths_cfg = load_yaml_with_env(Path(args.paths_config))

    questions_path = Path(paths_cfg["tables"]["questions"])
    gold_pages_path = Path(paths_cfg["tables"]["gold_pages"])
    retrieval_dir = Path(paths_cfg["results"]["retrieval_dir"])
    qa_dir = Path(paths_cfg["results"]["qa_dir"])
    analysis_dir = Path(paths_cfg["results"]["analysis_dir"])
    ensure_dir(analysis_dir)

    cases = select_cases(
        questions_path=questions_path,
        retrieval_dir=retrieval_dir,
        qa_dir=qa_dir,
        gold_pages_path=gold_pages_path,
    )

    for name, case_list in cases.items():
        out_md = analysis_dir / f"cases_{name.upper()}.md"
        lines = [f"# Cases {name.upper()}", ""]
        for case in case_list:
            lines.append(f"## financebench_id: {case['financebench_id']}")
            lines.append("")
            lines.append("Question:")
            lines.append(case["question"])
            lines.append("")
            lines.append("BM25 Top-5:")
            lines.append(f"{case['bm25_top5']}")
            lines.append("")
            lines.append("BGE Top-5:")
            lines.append(f"{case['bge_top5']}")
            lines.append("")
            lines.append("ColQwen Top-5:")
            lines.append(f"{case['colqwen_top5']}")
            lines.append("\n---\n")
        out_md.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Saved %s cases to %s", name, out_md)


if __name__ == "__main__":
    main()

