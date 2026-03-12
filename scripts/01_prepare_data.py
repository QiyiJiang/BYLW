import argparse
from pathlib import Path

from fbbench.data_loading import build_questions_table
from fbbench.utils.config_loader import load_yaml_with_env
from fbbench.utils.io_utils import ensure_dir, write_table
from fbbench.utils.logging_utils import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FinanceBench question table.")
    parser.add_argument("--paths-config", type=str, default="config/paths.yaml")
    parser.add_argument(
        "--open-source-jsonl",
        type=str,
        default="data/raw/financebench_open_source.jsonl",
    )
    parser.add_argument(
        "--doc-info-jsonl",
        type=str,
        default="data/raw/financebench_document_information.jsonl",
    )
    args = parser.parse_args()

    logger = setup_logger("prepare_data")

    paths_cfg = load_yaml_with_env(Path(args.paths_config))
    questions_path = Path(paths_cfg["tables"]["questions"])
    ensure_dir(questions_path.parent)

    open_source_path = Path(args.open_source_jsonl)
    doc_info_path = Path(args.doc_info_jsonl)

    logger.info("Building questions table from %s and %s", open_source_path, doc_info_path)
    df = build_questions_table(open_source_path, doc_info_path)
    logger.info("Questions loaded: %d", len(df))

    write_table(df, questions_path)
    logger.info("Questions table saved to %s", questions_path)


if __name__ == "__main__":
    main()

