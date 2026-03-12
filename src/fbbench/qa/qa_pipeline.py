from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from fbbench.api_clients.base_client import ApiConfig
from fbbench.utils.config_loader import load_yaml_with_env
from fbbench.api_clients.llm_client import LlmClient
from fbbench.qa.prompting import build_prompt
from fbbench.utils.io_utils import read_table


@dataclass
class QaConfig:
    default_topk: int
    max_context_tokens: int
    max_answer_tokens: int
    temperature: float


def load_qa_config(experiment_cfg_path: Path) -> QaConfig:
    cfg = load_yaml_with_env(experiment_cfg_path)
    qa_cfg = cfg["qa"]
    return QaConfig(
        default_topk=qa_cfg.get("default_topk", 3),
        max_context_tokens=qa_cfg.get("max_context_tokens", 24000),
        max_answer_tokens=qa_cfg.get("max_answer_tokens", 1024),
        temperature=qa_cfg.get("temperature", 0.0),
    )


def load_llm_client(api_cfg_path: Path) -> LlmClient:
    api_cfg = load_yaml_with_env(api_cfg_path)
    llm_cfg = api_cfg["llm"]
    cfg = ApiConfig(
        base_url=llm_cfg["base_url"],
        model=llm_cfg["model"],
        api_key_env=llm_cfg["api_key_env"],
        timeout=llm_cfg.get("timeout", 60),
    )
    return LlmClient(cfg)


def select_evidence_pages(
    pages_df,
    page_uids: List[str],
    top_k: int,
) -> List[Dict]:
    """从 pages_df 中按 page_uids 顺序取前 top_k 个页面内容。"""
    selected = []
    uid_set = set(page_uids)
    sub = pages_df[pages_df["page_uid"].isin(uid_set)]
    mapping = {row["page_uid"]: row for _, row in sub.iterrows()}
    for uid in page_uids[:top_k]:
        row = mapping.get(uid)
        if row is None:
            continue
        selected.append(
            {
                "page_uid": row["page_uid"],
                "doc_name": row["doc_name"],
                "page_id": int(row["page_id"]),
                "page_text": row["page_text"],
            }
        )
    return selected


def parse_answer(raw_output: str) -> Dict[str, str]:
    """
    从模型输出中抽取 Answer 与 Evidence Pages 段落。
    若格式不规范，则整体作为 answer_text。
    """
    answer_prefix = "Answer:"
    evidence_prefix = "Evidence Pages:"
    answer_text = raw_output
    evidence_pages_text = ""
    if answer_prefix in raw_output:
        parts = raw_output.split(answer_prefix, 1)[1]
        if evidence_prefix in parts:
            ans_part, ev_part = parts.split(evidence_prefix, 1)
            answer_text = ans_part.strip()
            evidence_pages_text = ev_part.strip()
        else:
            answer_text = parts.strip()
    return {
        "answer_text": answer_text,
        "evidence_pages_text": evidence_pages_text,
    }


def run_qa_for_question(
    llm: LlmClient,
    qa_cfg: QaConfig,
    question: str,
    evidence_pages: List[Dict],
) -> Dict[str, str]:
    prompt = build_prompt(question, evidence_pages)
    raw_output = llm.generate_answer(
        prompt=prompt,
        temperature=qa_cfg.temperature,
        max_tokens=qa_cfg.max_answer_tokens,
    )
    parsed = parse_answer(raw_output)
    return {
        "raw_output": raw_output,
        "answer_text": parsed["answer_text"],
        "evidence_pages_text": parsed["evidence_pages_text"],
    }

