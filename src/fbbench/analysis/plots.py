from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from fbbench.utils.io_utils import ensure_dir


def save_flow_diagram_text(out_path: Path) -> None:
    """
    保存 Mermaid 文本形式的整体流程图描述，便于在论文中直接引用或用在线工具渲染。
    """
    mermaid = """flowchart LR
  question[Question] --> retriever[PageRetriever(BM25/BGE/ColPali)]
  retriever --> topk[Top-k Pages]
  topk --> qwen[Qwen2.5-7B-Instruct]
  qwen --> answer[Answer]
"""
    out_path.write_text(mermaid, encoding="utf-8")


def main() -> None:
    figures_dir = Path("results/figures")
    ensure_dir(figures_dir)
    flow_path = figures_dir / "figure_5_1_flow.mmd"
    save_flow_diagram_text(flow_path)
    print(f"Saved flow diagram mermaid to {flow_path}")


if __name__ == "__main__":
    main()

