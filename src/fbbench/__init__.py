"""
FinanceBench page-level retrieval + QA 实验包。

核心子模块包含：
- data_loading: JSONL 读取与问题表构建
- page_building: PDF 页图像与文本抽取
- gold_mapping: gold evidence 映射
- api_clients: 在线 BGE/ColPali/Qwen API 封装
- retrievers: BM25/BGE/ColPali 检索器
- qa: 提示模板与问答流水线
- eval: 各类指标与统计
- analysis: 典型案例与图表
"""

