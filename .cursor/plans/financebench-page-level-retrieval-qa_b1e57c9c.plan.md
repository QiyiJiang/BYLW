---
name: financebench-page-level-retrieval-qa
overview: 基于 FinanceBench open-source 150 构建页级证据检索 + Qwen 回答生成的完整离线实验工程，统一使用在线 API 调用 BM25/BGE/ColPali/Qwen 能力，并复现主检索与问答实验。
todos:
  - id: setup-env
    content: 初始化项目骨架与依赖（pyproject/requirements、config、src、scripts 等目录）
    status: completed
  - id: impl-data-loading
    content: 实现 FinanceBench JSONL 读取与问题表构建模块
    status: completed
  - id: impl-page-building
    content: 实现 PDF 渲染为页图像、文本抽取与页面表输出
    status: completed
  - id: impl-gold-mapping
    content: 实现 gold evidence 页级映射工具
    status: completed
  - id: impl-api-clients
    content: 实现 BGE/ColPali/Qwen 在线 API 客户端抽象
    status: completed
  - id: impl-retrievers
    content: 实现 BM25/BGE/ColPali 检索器与索引/embedding 预计算脚本
    status: completed
  - id: impl-retrieval-metrics
    content: 实现检索指标计算并生成表 5-1
    status: completed
  - id: impl-qa-pipeline
    content: 实现问答提示模板、Qwen 调用与主问答实验脚本
    status: completed
  - id: impl-qa-metrics
    content: 实现问答 Accuracy/SupportedAccuracy 计算与表 5-2/5-3
    status: completed
  - id: impl-typewise-analysis
    content: 实现按 question_type 的检索与问答分组统计（表 5-4）
    status: completed
  - id: impl-cases-and-plots
    content: 实现典型案例筛选与图 5-1/5-2 生成
    status: completed
isProject: false
---

## 1. 整体技术路线与依赖

- **语言与运行方式**
  - **语言**：Python 3.10+（便于生态与 PDF/NLP 工具使用）。
  - **运行**：默认使用 `uv` 管理环境，通过 `uv run python ...` 执行各阶段脚本。
- **模型调用方式（全部走在线 API）**
  - **BM25**：不依赖深度模型，直接使用本地库实现（如 `rank-bm25`），**不违反“模型走在线 API”约束**，因为其是纯规则算法；若你坚持也走服务，可按计划预留 HTTP 接口封装层。
  - **BGE-base-en-v1.5**：通过统一的 `EmbeddingClient` 抽象，底层使用你现有的在线推理服务或第三方兼容 OpenAI Embedding API 的服务（模型名配置为 `BAAI/bge-base-en-v1.5`）。
  - **ColPali**：通过 `ColPaliClient` 抽象，假定有 HTTP 接口：输入页图片（base64 或 URL）+ 查询文本，返回多向量表示或检索得分；计划层面定义清晰接口，具体 URL/Key 通过配置实现。
  - **Qwen2.5-7B-Instruct**：通过统一的 `LLMClient` 抽象，底层对接你选择的 Qwen 在线 API（如 Ali DashScope、通义千问 API、或兼容 OpenAI Chat Completions 的代理）。
- **配置管理**
  - 使用 `config/` 下的 YAML/JSON（如 `config/paths.yaml`, `config/api.yaml`, `config/experiment.yaml`）统一管理：
    - 数据路径（原始 JSONL、PDF、缓存数据目录）。
    - API endpoint、API key、模型名、并发/速率限制。
    - 实验超参数（Top-k、批大小、随机种子等）。

---

## 2. 工程目录结构设计

- **根目录结构**
  - `README.md`：说明项目目标、数据下载方式、各阶段脚本用法，以及如何用 `uv` 运行。
  - `pyproject.toml` / `requirements.txt`：列出依赖（`pdfplumber`/`pymupdf`、OCR 库、`rank-bm25`、`pandas`、`numpy`、`tqdm`、`httpx`/`requests` 等）。
  - `config/`
    - `paths.yaml`：本地数据与输出目录设定（原始/中间结果/指标/图表）。
    - `api.yaml`：各在线模型 API 的 base_url、模型名、密钥字段名、超时等。
    - `experiment.yaml`：统一实验设置（Top-k 列表、批大小、随机种子等）。
  - `data/`
    - `raw/`：放 `financebench_open_source.jsonl`、`financebench_document_information.jsonl`、`pdfs/`。
    - `processed/`：标准化后的问题表、页面表、gold pages 映射等（统一为 `parquet`/`jsonl`）。
    - `cache/`：embedding、索引中间结果（如 BGE/ColPali 向量），便于断点续跑。
  - `src/`
    - `fbbench/data_loading.py`：数据读取与联合。
    - `fbbench/page_building.py`：PDF → 页图像 + 页文本提取。
    - `fbbench/gold_mapping.py`：gold pages 映射构建与查询工具。
    - `fbbench/api_clients/`
      - `__init__.py`
      - `base_client.py`：通用 HTTP Client 基类（重试、节流、日志）。
      - `embedding_client.py`：BGE embedding API 封装。
      - `colpali_client.py`：ColPali 检索/embedding API 封装。
      - `llm_client.py`：Qwen2.5 聊天/生成 API 封装。
    - `fbbench/retrievers/`
      - `bm25_retriever.py`
      - `bge_retriever.py`
      - `colpali_retriever.py`
      - `interfaces.py`：`BaseRetriever` 抽象（`retrieve(question, top_k)`）。
    - `fbbench/qa/`
      - `prompting.py`：问答提示模板与上下文组织。
      - `qa_pipeline.py`：给定检索结果 + 问题，调用 Qwen 生成答案。
    - `fbbench/eval/`
      - `retrieval_metrics.py`：Recall@k、MRR 计算。
      - `qa_metrics.py`：Accuracy、Supported Accuracy 计算（读取人工标注结果）。
      - `typewise_analysis.py`：按 `question_type` 分组统计。
    - `fbbench/analysis/`
      - `cases_selection.py`：从检索/问答结果中挑选代表性案例的脚本。
      - `plots.py`：图 5-1/5-2 流程图与可视化生成（可用 `matplotlib` 或 `graphviz`/`mermaid` 文本导出）。
    - `fbbench/utils/`
      - `io_utils.py`：读写 JSONL/Parquet、路径工具。
      - `logging_utils.py`：统一日志格式。
      - `timer.py`：简单计时器与性能统计。
      - `sampling.py`：可选的子集/调试样本抽样工具。
  - `scripts/`
    - `01_prepare_data.py`
    - `02_build_pages.py`
    - `03_build_indices.py`
    - `04_run_retrieval_experiments.py`
    - `05_run_qa_experiments.py`
    - `06_run_ablation_topk.py`
    - `07_run_typewise_analysis.py`
    - `08_select_cases.py`
    - 每个脚本支持命令行参数（如 `--topk 3`/`--retriever colpali`）。
  - `results/`
    - `retrieval/`：保存表 5-1 原始结果与汇总表。
    - `qa/`：表 5-2/5-3/5-4 的中间与最终表格（CSV/Markdown）。
    - `figures/`：图 5-1/5-2。
    - `annotations/`：人工 QA 标注结果（Correct/Incorrect/Insufficient + 引用页是否 gold）。

---

## 3. 数据准备阶段实现计划

- **3.1 读取官方 JSONL 与合并**（`fbbench/data_loading.py` + `scripts/01_prepare_data.py`）
  - 功能：
    - 读取 `financebench_open_source.jsonl`、`financebench_document_information.jsonl`。
    - 按官方 README 的 join 规则合并，生成统一问题表：
      - 字段：`financebench_id`, `question`, `answer`, `evidence`, `question_type`, `question_reasoning`, `company`, `doc_name` 等。
    - 清洗：
      - 确认 `evidence` 中 `evidence_doc_name`、`evidence_page_num` 与 `doc_name`、PDF 文件实际存在对应关系；对缺失/异常样本记录并单独导出日志以便检查。
    - 输出：
      - `data/processed/questions.parquet` 或 `questions.jsonl`。
- **3.2 构建页面库**（`fbbench/page_building.py` + `scripts/02_build_pages.py`）
  - 页级信息：
    - `page_uid = {doc_name}_{page_id}`（`page_id` 使用 0-index，与 `evidence_page_num` 对齐）。
    - `doc_name`, `page_id`, `page_image_path`, `page_text`。
  - PDF → 图像：
    - 使用 `PyMuPDF` 或 `pdf2image`，DPI 设为 144/150，输出到 `data/processed/page_images/{doc_name}/{page_id}.png`。
  - 页文本抽取：
    - 先用 `pdfplumber`/`PyMuPDF` 提取文本。
    - 若长度过短或为空（例如 `< min_char_threshold`）：调用 OCR（如 `pytesseract` 或在线 OCR API）对当前页图像进行识别，将结果填充为 `page_text_ocr`，并合并为最终 `page_text`。
  - 输出：
    - `data/processed/pages.parquet`：包含全部页元数据与文本。
- **3.3 正样本映射**（`fbbench/gold_mapping.py`）
  - 根据问题表 `evidence` 字段，生成：
    - 对每个 `financebench_id`：`gold_pages = set((evidence_doc_name, evidence_page_num))`。
    - 映射到 `page_uid` 形式，方便与检索结果对齐。
  - 输出：
    - `data/processed/gold_pages.parquet` 或 `gold_pages.jsonl`，字段：`financebench_id`, `gold_page_uids`。

---

## 4. 检索索引与检索器实现计划

- **4.1 检索接口抽象**（`fbbench/retrievers/interfaces.py`）
  - 定义 `BaseRetriever`：
    - `retrieve(question: str, top_k: int) -> List[RetrievalResult]`。
    - `RetrievalResult` 包含：`page_uid`, `score`, `doc_name`, `page_id`。
  - 可选：增加 `batch_retrieve(questions: List[str], top_k: int)` 用于后续批处理提速。
- **4.2 BM25 检索器**（`fbbench/retrievers/bm25_retriever.py`）
  - 索引构建：
    - 在 `scripts/03_build_indices.py` 或第一次调用时，将所有 `page_text` 分词（简单英文 tokenizer）并建立 `BM25Okapi` 模型；序列化至 `data/cache/bm25_index.pkl`。
  - 检索：
    - `retrieve`：对 `question` 分词，调用 BM25 排序，返回 Top-k `page_uid` 列表及 BM25 分数。
- **4.3 BGE 稠密检索器**（`fbbench/retrievers/bge_retriever.py` + `fbbench/api_clients/embedding_client.py`）
  - Embedding API 抽象：
    - `EmbeddingClient.embed_texts(texts: List[str], input_type: Literal['query','document']) -> np.ndarray`。
    - 内部：
      - 读取 `api.yaml` 中 BGE 的 endpoint、模型名、api_key。
      - 走 HTTP POST（如 OpenAI `/embeddings` 风格），处理批量、重试与限速。
  - 页面向量预计算：
    - `scripts/03_build_indices.py`：
      - 将所有 `page_text` 按批调用 BGE `input_type='document'` 编码，得到 `page_embeddings`。
      - 保存为 `data/cache/bge_page_embeddings.npy` + 索引 mapping（`page_uid` 顺序）。
  - 检索：
    - 对 `question` 调用 BGE `input_type='query'` 获取向量，与页面向量计算相似度（点积或余弦），Top-k 排序返回。
    - 相似度运算可直接用 `numpy`，也可加简单 FAISS（仍离线）以提速；但为简化可先用 `numpy`。
- **4.4 ColPali 页图像检索器**（`fbbench/retrievers/colpali_retriever.py` + `fbbench/api_clients/colpali_client.py`）
  - API 设计（计划层面）：
    - 方案 A：**embedding 式**
      - `ColPaliClient.embed_pages(images: List[bytes]) -> np.ndarray`（页面多向量或 pooled 向量）。
      - `ColPaliClient.embed_queries(questions: List[str]) -> np.ndarray`。
      - 本地实现 late interaction 打分（如 max-sim/pooling）。
    - 方案 B：**直接检索式**（推荐以减轻本地逻辑）
      - `ColPaliClient.search(question: str, top_k: int, page_image_ids: List[str]) -> List[RetrievalResult]`，由服务端持有索引。
    - 为通用性，计划中：
      - 先按 **embedding 式** 规划接口，这样即使你将来把 ColPali 部署为纯 embedding 服务，也能兼容。
  - 页面向量预计算：
    - `scripts/03_build_indices.py`：读取 `page_image_path`，按批加载图片为字节或 base64，调用 `ColPaliClient.embed_pages`，得到页面多向量/聚合向量，序列化到 `data/cache/colpali_page_embeddings/`。
  - 检索：
    - 对 `question` 调用 `embed_queries`，与页面多向量按 ColPali 官方推荐的 late interaction/相似度计算方式实现打分，返回 Top-k。
    - 为保持计划聚焦，可以约定：
      - 若服务端已经做了检索，则本地 `colpali_retriever` 只负责调用 `search` 并转成统一 `RetrievalResult`。

---

## 5. 检索实验流水线（表 5-1）

- **5.1 统一检索脚本**（`scripts/04_run_retrieval_experiments.py`）
  - 输入：
    - `--retriever {bm25,bge,colpali}`
    - `--topk_list 1 3 5`
  - 步骤：
    1. 读取问题表 `questions` 与 `gold_pages` 映射。
    2. 初始化相应检索器（加载索引/embedding）。
    3. 对每个问题：
      - 调用 `retrieve(question, max(topk_list))`，得到排序列表。
      - 记录结果：`financebench_id`, `rank`, `page_uid`, `score`。
    4. 将所有结果保存为 `results/retrieval/{retriever}_raw_results.parquet`。
- **5.2 检索指标评估模块**（`fbbench/eval/retrieval_metrics.py`）
  - 功能：
    - 给定 raw 检索结果 + `gold_page_uids`：
      - 计算单问题的命中位置（如果有多个 gold page，则取最小 rank）。
      - 计算：Recall@1 / 3 / 5、MRR。
    - 支持按 `question_type` 子集过滤（供后续类型分析复用）。
  - 输出：
    - 汇总 CSV：`results/retrieval/table_5_1.csv`（Method, Recall@1, Recall@3, Recall@5, MRR）。

---

## 6. 问答阶段与主实验（表 5-2）

- **6.1 Qwen LLM 客户端**（`fbbench/api_clients/llm_client.py`）
  - 提供：`generate_answer(prompt: str, temperature: float=0.0, max_tokens: int=512) -> str`。
  - 负责：
    - 拼接成对应 API 所需的请求格式（如 chat messages）。
    - 处理返回与错误重试。
    - 支持批量/流式可选，但基础实现单条即可。
- **6.2 提示模板与上下文组织**（`fbbench/qa/prompting.py`）
  - 实现：

```text
You are a financial document QA assistant.
Answer the question only based on the provided evidence pages.
If the evidence is insufficient, answer "Insufficient evidence".
Give a concise answer first, then provide the supporting page numbers.

Question:
{question}

Evidence Pages:
{retrieved_pages_text}
```

- 上下文拼接：
  - 从检索结果中按照分数排序取 Top-k 页面；按格式：

```text
[Document: {doc_name} | Page: {page_id}]
{page_text}
```

```
- 控制总 token 数不超过 Qwen2.5 支持的上下文（如预设上限，若超出则截断尾部页面或裁剪页文本）。
```

- **6.3 QA Pipeline**（`fbbench/qa/qa_pipeline.py`）
  - 功能：
    - `run_qa_for_question(question_obj, evidence_pages, mode)`，其中 `mode` 区分：
      - `qwen_only`：不提供 evidence，仅用问题与指令。
      - `bm25`, `bge`, `colpali`：使用相应检索结果的 Top-k 页文本。
      - `oracle`：使用 gold page 的 `page_text`（全部或 Top-k 级别上界）。
    - 解析 Qwen 输出，标准化为：
      - `answer_text`（去掉前缀文本，如 `Answer:` 行）。
      - `reported_pages`（从输出中解析 `Evidence Pages:` 后的内容，保留为字符串供后续人工核查）。
- **6.4 主问答实验脚本**（`scripts/05_run_qa_experiments.py`）
  - 目标：生成表 5-2 所需数据（Qwen-only, BM25+BGE+ColPali+Oracle）。
  - 配置：
    - 默认 Top-k=3（与设计一致），可通过命令行覆盖。
  - 流程：
    1. 读取问题表、页面表、检索结果（或 gold pages）。
    2. 对每个方法：
      - 构造对应 evidence 集合（Qwen-only 即为空上下文，Oracle 为 gold page 文本）。
      - 调用 QA pipeline + Qwen API，记录：
        - `financebench_id`, `method`, `pred_answer`, `evidence_pages_used`（实际拼接的 `page_uid` 列表），`model_raw_output`。
      - 考虑 API 速率：实现简单的 sleep 或并发控制，支持断点续跑（结果按问题 id 增量写文件）。
    3. 输出：`results/qa/{method}_answers.parquet`。
- **6.5 人工标注与指标计算**（`fbbench/eval/qa_metrics.py`）
  - 标注格式：
    - 针对每条 `(financebench_id, method)` 的模型输出，在 `results/annotations/qa_labels.csv` 中人工补充：
      - `label` ∈ {`correct`, `incorrect`, `insufficient`, `hallucinated`}。
      - `evidence_supported`：布尔，表示模型输出中引用的页（人工理解）是否在 gold pages 中。
  - 指标：
    - `Accuracy`：`label == correct` 比例。
    - `SupportedAccuracy`：`label == correct and evidence_supported == True` 比例。
  - 输出：
    - `results/qa/table_5_2.csv`：Method, Accuracy, SupportedAccuracy。

---

## 7. Top-k 消融实验（表 5-3）

- **7.1 消融配置**
  - 在 `experiment.yaml` 中设定：`topk_variants: [1, 3, 5]`。
- **7.2 实现**（`scripts/06_run_ablation_topk.py`）
  - 流程：
    1. 固定检索器为 ColPali（读取其完整检索结果）。
    2. 对每个 Top-k ∈ {1,3,5}：
      - 重新构造 evidence 文本（仅使用前 k 页）。
      - 调用 Qwen 生成答案，输出至 `results/qa/colpali_top{k}_answers.parquet`。
    3. 复用 `qa_metrics.py` 上的人工标注（可在标注时一并考虑不同 Top-k），统计 Accuracy 与 SupportedAccuracy。
    4. 合并为 `results/qa/table_5_3.csv`。

---

## 8. 分类型分析（表 5-4）

- **8.1 类型分组统计**（`fbbench/eval/typewise_analysis.py` + `scripts/07_run_typewise_analysis.py`）
  - 输入：
    - 检索结果 + `question_type` 字段。
    - QA 人工标注表。
  - 步骤：
    - 对每类：`metrics-generated`, `domain-relevant`, `novel-generated`：
      - 计算 ColPali 与 BGE 的：
        - Recall@1/3/5, MRR（调用 `retrieval_metrics` 且加 type 过滤）。
      - 计算 Accuracy/SupportedAccuracy（按方法过滤，典型为 BGE+Qwen 与 ColPali+Qwen）。
  - 输出：
    - `results/qa/table_5_4.csv`：按 question_type 展开的对比表。

---

## 9. 典型案例分析与可视化（图 5-1, 图 5-2）

- **9.1 案例筛选脚本**（`scripts/08_select_cases.py` + `fbbench/analysis/cases_selection.py`）
  - 案例 A：`BM25/BGE 未命中 gold page，ColPali 命中`。
  - 案例 B：`BM25/BGE 命中但 rank 较高，ColPali rank 更靠前`。
  - 案例 C：`检索命中 gold page，但 Qwen 回答标注为 incorrect`。
  - 将选中案例的详细信息（问题、gold pages、各方法 Top-5 检索结果、生成回答与人工标签）导出为 `results/analysis/cases_{A,B,C}.md`，方便直接引用到论文。
- **9.2 实验流程图（图 5-1）生成**（`fbbench/analysis/plots.py`）
  - 输出一个 Mermaid/Graphviz 描述或 PDF/PNG：

```mermaid
flowchart LR
  question[Question] --> retriever[PageRetriever(BM25/BGE/ColPali)]
  retriever --> topk[Top-k Pages]
  topk --> qwen[Qwen2.5-7B-Instruct]
  qwen --> answer[Answer]
```



- 保存为 `results/figures/figure_5_1_flow.png` 或 `figure_5_1_flow.mmd`。
- **9.3 典型案例检索结果可视化（图 5-2）**
  - 对案例 A/B：
    - 生成条形图或表格图，展示不同检索器对同一问题的 Top-k 排名与 gold page 位置。
    - 保存为 `results/figures/figure_5_2_case_visualization.png`。

---

## 10. 运行顺序与命令示例

- **第 1 步：数据准备**
  - `uv run python scripts/01_prepare_data.py`
  - `uv run python scripts/02_build_pages.py`
- **第 2 步：建立索引（或预计算向量）**
  - `uv run python scripts/03_build_indices.py --retrievers bm25 bge colpali`
- **第 3 步：主检索实验（表 5-1）**
  - `uv run python scripts/04_run_retrieval_experiments.py --retrievers bm25 bge colpali --topk_list 1 3 5`
- **第 4 步：主问答实验（表 5-2，Top-3）**
  - `uv run python scripts/05_run_qa_experiments.py --topk 3`
- **第 5 步：Top-k 消融（表 5-3）**
  - `uv run python scripts/06_run_ablation_topk.py --topk_list 1 3 5`
- **第 6 步：分类型分析（表 5-4）**
  - `uv run python scripts/07_run_typewise_analysis.py`
- **第 7 步：典型案例与图表**
  - `uv run python scripts/08_select_cases.py`
  - `uv run python -m fbbench.analysis.plots`

---

## 11. Todo 列表（按实现顺序）

- `setup-env`：初始化 `pyproject.toml`/`requirements.txt` 与基础目录结构。
- `impl-data-loading`：实现 JSONL 读取与问题表构建。
- `impl-page-building`：实现 PDF 渲染、页文本抽取与 OCR 兜底，以及页面表输出。
- `impl-gold-mapping`：实现 gold page 映射工具。
- `impl-api-clients`：实现 BGE/ColPali/Qwen 的在线 API 客户端抽象（含配置读取、重试与节流）。
- `impl-retrievers`：实现 BM25/BGE/ColPali 三种检索器与索引/embedding 预计算脚本。
- `impl-retrieval-metrics`：实现检索指标计算与表 5-1 生成。
- `impl-qa-pipeline`：实现提示模板、Qwen 调用与问答结果存储。
- `impl-qa-metrics`：定义人工标注格式并实现 Accuracy/SupportedAccuracy 计算与表 5-2/5-3。
- `impl-typewise-analysis`：实现按 `question_type` 的分组统计与表 5-4。
- `impl-cases-and-plots`：实现典型案例筛选与图 5-1/5-2 的生成。

