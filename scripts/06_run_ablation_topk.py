import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from fbbench.qa.qa_pipeline import (
    QaConfig,
    load_qa_config,
    load_llm_client,
    run_qa_for_question,
    select_evidence_pages,
)
from fbbench.utils.config_loader import load_yaml_with_env
from fbbench.utils.io_utils import ensure_dir, read_table, write_table
from fbbench.utils.logging_utils import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-k ablation for ColQwen + Qwen.")
    parser.add_argument("--paths-config", type=str, default="config/paths.yaml")
    parser.add_argument("--api-config", type=str, default="config/api.yaml")
    parser.add_argument("--experiment-config", type=str, default="config/experiment.yaml")
    args = parser.parse_args()

    logger = setup_logger("ablation_topk")

    paths_cfg = load_yaml_with_env(Path(args.paths_config))
    exp_cfg = load_yaml_with_env(Path(args.experiment_config))

    questions_path = Path(paths_cfg["tables"]["questions"])
    pages_path = Path(paths_cfg["tables"]["pages"])
    retrieval_dir = Path(paths_cfg["results"]["retrieval_dir"])
    qa_dir = Path(paths_cfg["results"]["qa_dir"])
    ensure_dir(qa_dir)

    questions_df = read_table(questions_path)
    pages_df = read_table(pages_path)
    retr_df = read_table(retrieval_dir / "colqwen_raw_results.parquet")

    qa_cfg: QaConfig = load_qa_config(Path(args.experiment_config))
    ablation_topks = exp_cfg["ablation"]["colqwen_topk_variants"]

    llm = load_llm_client(Path(args.api_config))

    for top_k in ablation_topks:
        logger.info("Running ColQwen+Qwen ablation, Top-k=%d", top_k)
        rows = []
        for _, qrow in tqdm(
            questions_df.iterrows(),
            total=len(questions_df),
            desc=f"Ablation ColQwen top-{top_k}",
        ):
            fid = qrow["financebench_id"]
            question = qrow["question"]

            sub = retr_df[retr_df["financebench_id"] == fid].sort_values("rank")
            page_uids = sub["page_uid"].tolist()[:top_k]
            evidence_pages = select_evidence_pages(pages_df, page_uids, top_k=top_k)

            qa_result = run_qa_for_question(
                llm=llm,
                qa_cfg=qa_cfg,
                question=question,
                evidence_pages=evidence_pages,
            )

            rows.append(
                {
                    "financebench_id": fid,
                    "method": f"colqwen_top{top_k}",
                    "top_k": top_k,
                    "question": question,
                    "pred_answer": qa_result["answer_text"],
                    "model_raw_output": qa_result["raw_output"],
                    "evidence_pages_used": page_uids,
                    "evidence_pages_text": qa_result["evidence_pages_text"],
                }
            )

        out_path = qa_dir / f"colqwen_top{top_k}_answers.parquet"
        write_table(pd.DataFrame(rows), out_path)
        logger.info("Saved ablation QA outputs for Top-%d to %s", top_k, out_path)


if __name__ == "__main__":
    main()

