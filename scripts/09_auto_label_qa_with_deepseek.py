import argparse
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from fbbench.gold_mapping import build_gold_page_mapping
from fbbench.qa.qa_pipeline import load_llm_client
from fbbench.utils.config_loader import load_yaml_with_env
from fbbench.utils.io_utils import ensure_dir, read_table, write_table
from fbbench.utils.logging_utils import setup_logger


LABEL_SYSTEM_PROMPT = """你是一个严谨的财报问答评估助手。

请根据下面的信息，对模型回答进行打分：
- label: 在下列四个中选择一个标签：
  - correct: 回答正确且完整
  - incorrect: 回答明显错误
  - insufficient: 回答部分正确但不完整或缺少关键信息
  - hallucinated: 回答内容在给定证据中找不到依据，明显胡编乱造
- evidence_supported: true/false，表示回答中使用的关键信息是否都能在 gold pages 中找到依据。

只输出一个 JSON 对象，形如：
{"label": "correct", "evidence_supported": true}
不要输出额外解释。"""


def build_gold_lookup(questions_path: Path) -> Dict[str, List[str]]:
    """构建 {financebench_id: gold_page_uids} 映射，便于在 prompt 中展示 gold pages。"""
    gold_df = build_gold_page_mapping(questions_path)
    lookup: Dict[str, List[str]] = {}
    for _, row in gold_df.iterrows():
        fid = row["financebench_id"]
        lookup[str(fid)] = list(row.get("gold_page_uids") or [])
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 DeepSeek 自动标注 QA 结果，生成 qa_labels.csv。")
    parser.add_argument("--paths-config", type=str, default="config/paths.yaml")
    parser.add_argument("--api-config", type=str, default="config/api.yaml")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["qwen_only", "bm25", "bge", "colqwen", "oracle"],
        help="需要评估的 QA 方法，对应 results/qa/{method}_answers.parquet",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/annotations/qa_labels.csv",
        help="输出标注结果 CSV 路径",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=20,
        help="并发标注的线程数（默认 20）",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="仅用于调试：最多评估前 N 条问题",
    )
    args = parser.parse_args()

    logger = setup_logger("auto_label_qa")

    paths_cfg = load_yaml_with_env(Path(args.paths_config))
    qa_dir = Path(paths_cfg["results"]["qa_dir"])
    questions_path = Path(paths_cfg["tables"]["questions"])
    ensure_dir(Path(args.output).parent)

    # 构建 gold pages 映射
    gold_lookup = build_gold_lookup(questions_path)

    # 加载 DeepSeek LLM 客户端（经由 load_llm_client）
    llm = load_llm_client(Path(args.api_config))

    all_rows: List[Dict[str, object]] = []

    for method in args.methods:
        qa_path = qa_dir / f"{method}_answers.parquet"
        if not qa_path.exists():
            logger.warning("QA file not found for method %s: %s", method, qa_path)
            continue

        logger.info(
            "Loading QA results for method %s from %s (num_workers=%d)",
            method,
            qa_path,
            args.num_workers,
        )
        qa_df = read_table(qa_path)
        if args.max_questions is not None:
            qa_df = qa_df.head(args.max_questions)

        def _build_row(row):
            fid = str(row["financebench_id"])
            question = str(row["question"])
            pred_answer = str(row["pred_answer"])

            raw_used = row.get("evidence_pages_used")
            if raw_used is None:
                evidence_used = []
            elif isinstance(raw_used, list):
                evidence_used = raw_used
            else:
                try:
                    evidence_used = list(raw_used)
                except TypeError:
                    evidence_used = []

            gold_pages = gold_lookup.get(fid, [])

            user_prompt = f"""问题：
{question}

模型回答：
{pred_answer}

检索使用的证据页面 page_uid 列表：
{evidence_used}

gold pages（真正包含答案的 page_uid 列表）：
{gold_pages}

请根据上述信息，评估该回答的正确性（label）以及是否被 gold pages 支持（evidence_supported）。"""

            # 使用 DeepSeek 生成 JSON 标签
            raw = llm.generate_answer(
                prompt=f"{LABEL_SYSTEM_PROMPT}\n\n{user_prompt}",
                temperature=0.0,
                max_tokens=256,
            )

            # 简单从返回中解析 JSON（防御性处理），并去掉 ```json 包裹
            label = "insufficient"
            evidence_supported = False
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                # 去掉 markdown 代码块包裹
                cleaned = cleaned.strip("`")
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()

            try:
                import json

                data = json.loads(cleaned)
                if isinstance(data, dict):
                    label = str(data.get("label", label))
                    evidence_supported = bool(data.get("evidence_supported", evidence_supported))
            except Exception:  # noqa: BLE001
                logger.warning("Failed to parse JSON for fid=%s, method=%s, raw=%r", fid, method, raw)

            return {
                "financebench_id": fid,
                "method": method,
                "label": label,
                "evidence_supported": evidence_supported,
                "raw_judgement": raw,
            }

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(_build_row, row): idx
                for idx, (_, row) in enumerate(qa_df.iterrows())
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Labeling {method}"):
                all_rows.append(future.result())

    out_path = Path(args.output)
    if all_rows:
        write_table(pd.DataFrame(all_rows), out_path)
        logger.info("Saved QA labels to %s (rows=%d)", out_path, len(all_rows))
    else:
        logger.warning("No labels generated; please check QA files and methods list.")


if __name__ == "__main__":
    main()

