# scripts/11_label_ablation_and_compute_table_5_3.py
"""
对 Top-k 消融实验（scripts/06_run_ablation_topk.py 输出的
results/qa/colqwen_top{k}_answers.parquet）使用 DeepSeek 自动标注，
并计算每个 Top-k 的 Accuracy / SupportedAccuracy，汇总为表 5-3。

输出：
- 每个 k 一份标注文件：
  results/annotations/qa_labels_colqwen_top{k}.csv
- 总表：
  results/qa/table_5_3.csv
"""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pandas as pd
from tqdm import tqdm

from fbbench.eval.qa_metrics import compute_qa_metrics
from fbbench.utils.config_loader import load_yaml_with_env
from fbbench.utils.io_utils import ensure_dir, read_table, write_table
from fbbench.utils.logging_utils import setup_logger


def load_llm_config(api_config_path: Path) -> Dict[str, Any]:
    api_cfg = load_yaml_with_env(api_config_path)
    llm_cfg = api_cfg["llm"]
    return {
        "base_url": llm_cfg["base_url"].rstrip("/"),
        "model": llm_cfg["model"],
        "api_key_env": llm_cfg["api_key_env"],
        "timeout": llm_cfg.get("timeout", 60),
    }


def call_deepseek(
    client: httpx.Client,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    url = f"{base_url}/v1/chat/completions"
    resp = client.post(
        url,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        },
        timeout=None,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


_JSON_BLOCK_RE = re.compile(r"```json(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    从 DeepSeek 返回文本中提取 JSON：
    - 优先解析 ```json ... ``` 代码块；
    - 否则尝试整体作为 JSON 解析。
    """
    m = _JSON_BLOCK_RE.search(text)
    raw = m.group(1).strip() if m else text.strip()
    # 去掉可能多余的包裹反引号
    raw = raw.strip("`").strip()
    try:
        return json.loads(raw)
    except Exception:
        # 解析失败时返回兜底
        return {"label": "insufficient", "evidence_supported": False}


def build_judge_prompt(question: str, pred_answer: str, evidence_text: str) -> str:
    instruction = (
        "你是一个严谨的评估员。请根据给定的“问题、模型回答、证据文本”，判断：\n"
        "1）模型回答是否正确（完全正确为 'correct'，明显错误为 'incorrect'，信息不足为 'insufficient'）；\n"
        "2）如果回答正确或部分正确，证据文本是否足够支撑这个回答（true/false）。\n"
        "只输出一个 JSON 对象，格式如下：\n"
        '{\"label\": \"correct/incorrect/insufficient\", \"evidence_supported\": true/false}\n'
        "不要输出任何多余文字。"
    )
    return (
        instruction
        + "\n\n"
        + "【问题】\n"
        + question
        + "\n\n【模型回答】\n"
        + pred_answer
        + "\n\n【证据文本（模型使用的页面内容拼接）】\n"
        + evidence_text
    )


def label_single_row(
    row: pd.Series,
    client: httpx.Client,
    base_url: str,
    model: str,
) -> Dict[str, Any]:
    question = str(row["question"])
    pred_answer = str(row["pred_answer"])
    evidence_text = str(row.get("evidence_pages_text") or "")

    system_prompt = "你是一个用于评估问答质量和证据充分性的评估助手。"
    user_prompt = build_judge_prompt(question, pred_answer, evidence_text)

    raw = call_deepseek(client, base_url, model, system_prompt, user_prompt)
    parsed = extract_json_from_text(raw)

    label = str(parsed.get("label", "insufficient")).lower()
    if label not in {"correct", "incorrect", "insufficient"}:
        label = "insufficient"
    ev_supported = bool(parsed.get("evidence_supported", False))

    return {
        "financebench_id": row["financebench_id"],
        "question": question,
        "pred_answer": pred_answer,
        "label": label,
        "evidence_supported": ev_supported,
        "raw_judgement": raw,
    }


def process_topk_file(
    answers_path: Path,
    base_url: str,
    model: str,
    max_workers: int = 20,
) -> pd.DataFrame:
    df = read_table(answers_path)
    rows: List[Dict[str, Any]] = []

    with httpx.Client(timeout=None) as client:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(label_single_row, row, client, base_url, model): idx
                for idx, (_, row) in enumerate(df.iterrows())
            }
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Labeling {answers_path.name}",
            ):
                rows.append(fut.result())

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use DeepSeek to label ColQwen Top-k ablation QA results and compute table 5-3."
    )
    parser.add_argument("--paths-config", type=str, default="config/paths.yaml")
    parser.add_argument("--api-config", type=str, default="config/api.yaml")
    parser.add_argument("--num-workers", type=int, default=20)
    args = parser.parse_args()

    logger = setup_logger("label_ablation_and_table_5_3")

    paths_cfg = load_yaml_with_env(Path(args.paths_config))
    qa_dir = Path(paths_cfg["results"]["qa_dir"])
    annotations_dir = Path(paths_cfg["results"]["annotations_dir"])
    ensure_dir(qa_dir)
    ensure_dir(annotations_dir)

    api_conf = load_llm_config(Path(args.api_config))
    base_url = api_conf["base_url"]
    model = api_conf["model"]

    # 找到所有 colqwen_top{k}_answers.parquet
    answer_files = sorted(qa_dir.glob("colqwen_top*_answers.parquet"))
    if not answer_files:
        logger.error("No colqwen_top*_answers.parquet found in %s", qa_dir)
        return

    summary_rows: List[Dict[str, Any]] = []

    for ans_path in answer_files:
        # 从文件名中解析 top_k
        # 形如 colqwen_top1_answers.parquet 或 colqwen_top5_answers.parquet
        stem = ans_path.stem  # colqwen_top{k}_answers
        parts = stem.split("_")
        top_k = None
        for p in parts:
            if p.startswith("top"):
                try:
                    top_k = int(p.replace("top", ""))
                except ValueError:
                    pass
        if top_k is None:
            logger.warning("Skip %s: cannot parse top_k from name", ans_path)
            continue

        logger.info("Processing top-k = %d from %s", top_k, ans_path)

        labels_df = process_topk_file(
            answers_path=ans_path,
            base_url=base_url,
            model=model,
            max_workers=args.num_workers,
        )

        # 保存该 top-k 的标注结果
        labels_path = annotations_dir / f"qa_labels_colqwen_top{top_k}.csv"
        write_table(labels_df, labels_path)
        logger.info("Saved labels for top-%d to %s", top_k, labels_path)

        # 计算该 top-k 的 QA 指标
        metrics_df = labels_df.copy()
        metrics_df["method"] = f"colqwen_top{top_k}"
        qa_metrics = compute_qa_metrics(metrics_df)

        summary_rows.append(
            {
                "top_k": top_k,
                "method": f"colqwen_top{top_k}",
                "Accuracy": qa_metrics["accuracy"],
                "SupportedAccuracy": qa_metrics["supported_accuracy"],
            }
        )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values("top_k").reset_index(drop=True)
        out_path = qa_dir / "table_5_3.csv"
        write_table(summary_df, out_path)
        logger.info("Saved Top-k ablation QA metrics to %s", out_path)
    else:
        logger.warning("No summary rows produced; check input files.")


if __name__ == "__main__":
    main()