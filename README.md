# FinanceBench 页级检索 + QA 实验流程

本项目实现基于 **FinanceBench open-source 150** 的**页级证据检索 + 基于证据的问答**完整离线实验流程，支持三种检索方式（BM25、BGE、ColQwen），并使用 **DeepSeek** 作为问答与自动标注的 LLM。ColQwen 检索采用**多模态 + BM25 + BGE 三路融合 + BGE-large-zh 精排**的混合策略。

整条主线对应论文中的实验设置：

> 问题输入 → 页级检索（BM25 / BGE / ColQwen）→ 取 Top-k 证据页 → 调用 DeepSeek 回答 → 评测检索质量（Recall@k、MRR）与回答质量（Accuracy、SupportedAccuracy）

---

## 环境准备与运行方式

- **Python**：推荐 3.10+。
- **环境管理**：默认使用 **uv** 管理依赖与运行。

**安装依赖（项目根目录）：**

```bash
uv sync
```

**运行脚本：** 统一使用 `uv run`，例如：

```bash
uv run python scripts/01_prepare_data.py
```

---

## 配置说明

### 1. 环境变量与 `.env`

YAML 中的 API 地址、模型名、超时等通过 **环境变量** 注入，支持在 `config/*.yaml` 中使用 `${VAR}` 或 `${VAR:-默认值}`。项目使用 `python-dotenv` 在加载配置前读取 **`.env`**，因此无需在 shell 里手动 export（也可用 export 覆盖）。

- 复制示例并按需修改：

```bash
cp .env.example .env
# 编辑 .env，填入 API Key 与 base_url、model 等
```

- **必填项（示例）：**
  - **LLM（DeepSeek）**：`DEEPSEEK_LLM_API_KEY`、`LLM_BASE_URL`（如 `https://api.deepseek.com`）、`LLM_MODEL`（如 `deepseek-chat`）。
  - **Embedding（BGE 文本检索 / ColQwen 混合中的 BGE 一路）**：若使用远程 BGE 服务，需配置 `EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL`、`EMBEDDING_MODEL`；若仅用本地 BGE（见下文“索引构建”），可只保留占位。
- `.env` 已加入 `.gitignore`，不会提交密钥。

### 2. 配置文件

- **`config/paths.yaml`**  
  数据与结果路径：原始 JSONL、PDF 目录、`processed` 表、`cache` 缓存、`results` 输出、Milvus Lite DB 路径（如 `data/milvus/db`，程序会自动补全为 `db.db`）等。

- **`config/api.yaml`**  
  在线服务配置（会经 `load_yaml_with_env` 做 `${VAR}` 替换）：
  - **embedding**：BGE 文本向量接口（base_url、model、api_key_env、timeout），用于 BGE 检索器及 ColQwen 混合中的 BGE 一路。
  - **llm**：DeepSeek 对话接口（base_url、model、api_key_env、timeout），用于 QA 生成与自动标注。

- **`config/experiment.yaml`**  
  实验参数：检索 Top-k 列表、QA 默认 topk、上下文/答案 token 上限、温度；消融实验的 Top-k 列表由 **`ablation.colqwen_topk_variants`** 控制（如 `[1, 3, 5]`）。

配置加载统一通过 **`fbbench.utils.config_loader.load_yaml_with_env`**，会先加载 `.env` 再解析 YAML 并替换占位符。

---

## 目录结构（核心）

```
config/
  paths.yaml      # 数据/结果/缓存/Milvus 路径
  api.yaml        # Embedding / LLM API（含 ${VAR} 占位）
  experiment.yaml # 实验与消融参数（如 colqwen_topk_variants）

data/
  raw/
    financebench_open_source.jsonl
    financebench_document_information.jsonl
    pdfs/                    # 官方 PDF
  processed/
    questions.parquet        # 问题表
    pages.parquet           # 页面表（仅证据页，见下文）
    pages_by_doc/           # 按文档分片的页面 parquet（增量构建用）
  cache/                     # 中间缓存（可软链到数据盘）
    colqwen_multi_embeddings_*.parquet  # ColQwen 多向量编码分片
  milvus/
    db.db                   # Milvus Lite 向量库

src/fbbench/
  utils/
    config_loader.py        # YAML + .env 占位符加载
  page_building.py          # PDF → 页图像 + 页文本（仅证据页可选）
  gold_mapping.py           # evidence → gold page_uid 映射
  api_clients/              # Embedding / LLM 客户端
  index/
    milvus_client.py       # Milvus Lite 与 MilvusIndex
  retrievers/
    bm25_retriever.py       # BM25 文本检索
    bge_retriever.py        # BGE 向量检索（chunk 级聚合为页级）
    colqwen_retriever.py   # ColQwen 多模态 + BM25 + BGE 融合 + BGE-large-zh 精排
  qa/                       # 问答流水线（DeepSeek）
  eval/                      # 检索与 QA 指标

scripts/
  01_prepare_data.py        # 构建 questions.parquet
  02_build_pages.py         # 构建 pages.parquet（仅证据页）
  03_build_indices.py       # BGE 向量索引（本地）；可选 ColQwen 一步到位
  03a_colqwen_encode_only.py   # ColQwen 多向量编码 → cache 分片
  03b_colqwen_upsert_only.py   # 分片 → Milvus pages_colqwen
  04_run_retrieval_experiments.py  # 检索实验 → table_5_1
  05_run_qa_experiments.py  # QA 实验 → *_answers.parquet
  06_run_ablation_topk.py   # ColQwen Top-k 消融 → colqwen_top{k}_answers.parquet
  07_run_typewise_analysis.py   # 按 question_type 分析 → table_5_4
  08_select_cases.py        # 典型案例筛选
  09_auto_label_qa_with_deepseek.py  # DeepSeek 自动标注 → qa_labels.csv
  10_compute_qa_metrics.py  # 从 qa_labels 汇总 → table_5_2
  11_label_ablation_and_compute_table_5_3.py  # 消融标注 + 表 5-3

results/
  retrieval/                # 检索 raw 结果与 table_5_1.csv
  qa/                       # 各方法 answers、table_5_2/5_3/5_4
  annotations/              # qa_labels.csv、消融标注等
  figures/                  # 流程图等
  analysis/                 # 典型案例 Markdown
```

---

## 完整运行步骤（从数据到所有表）

假设已：

- 将 FinanceBench 官方 JSONL 与 `pdfs/` 放入 `data/raw/`；
- 配置好 `.env` 与 `config/api.yaml`（含 DeepSeek 与可选 BGE API）；
- 在项目根目录执行过 `uv sync`。

### 第 1 步：构建问题表（questions）

- **脚本**：`scripts/01_prepare_data.py`
- **作用**：读取 `financebench_open_source.jsonl` 与 `financebench_document_information.jsonl`，按官方规则构建问题表，输出 `data/processed/questions.parquet`（含 `financebench_id`、`question`、`answer`、`evidence`、`question_type` 等）。
- **命令**：

```bash
uv run python scripts/01_prepare_data.py
```

---

### 第 2 步：构建页面库（pages，仅证据页）

- **脚本**：`scripts/02_build_pages.py`
- **作用**：
  - 从 `data/raw/financebench_open_source.jsonl` 解析出所有 **evidence** 中的 `(doc_name, evidence_page_num)`，得到“证据页”集合；
  - **只处理这些证据页**：仅对每个 PDF 中出现在 evidence 里的页码做渲染与文本抽取（不处理整本 PDF 的其余页）；
  - 使用 **PyMuPDF** 渲染页图像（默认 150 DPI），保存到 `data/processed/page_images/{doc_name}/{page_id}.png`；
  - 使用 **pdfplumber** 抽取页文本（**不做 OCR**）；
  - 输出 `data/processed/pages.parquet`，字段：`page_uid`（`{doc_name}_{page_id}`）、`doc_name`、`page_id`、`page_image_path`、`page_text`。
- **命令**：

```bash
uv run python scripts/02_build_pages.py
```

可选：`--num-workers N` 并行处理多份 PDF；`--dpi`、`--min-char-threshold` 等见脚本帮助。

---

### 第 3 步：Gold 证据页映射

- **逻辑**：在 `src/fbbench/gold_mapping.py` 中实现，由后续检索/QA 脚本按需调用 **`build_gold_page_mapping(questions_path)`**，无需单独运行。
- **作用**：根据 `questions.parquet` 的 `evidence` 字段，为每个问题生成 `gold_page_uids` 等，用于计算 Recall@k、MRR 及 SupportedAccuracy。

---

### 第 4 步：预计算向量与建立索引

#### 4.1 BGE 文本向量（Milvus `pages_bge`）

- **脚本**：`scripts/03_build_indices.py --retrievers bge`
- **作用**：
  - 使用 **本地** `sentence_transformers.SentenceTransformer("BAAI/bge-m3")` 对 `pages.parquet` 的 `page_text` 做 **按 1000 字符的 chunk 切分**，每个 chunk 一条向量，`page_uid` 存为 `{原 page_uid}::c{chunk_idx}`；
  - 将向量写入 Milvus collection **`pages_bge`**；检索时按 chunk 检索再聚合回页级（每页取最佳 chunk 分数）。

```bash
uv run python scripts/03_build_indices.py --retrievers bge
```

#### 4.2 ColQwen 多模态向量（Milvus `pages_colqwen`）

ColQwen 采用 **两步**：先编码落盘，再写入 Milvus，便于断点续跑与减少重复计算。

- **4.2.1 编码**：`scripts/03a_colqwen_encode_only.py`
  - 读取 `pages.parquet`，用本地 **ColQwen2.5**（`colpali_engine`）对每页图像编码为 **多向量**（形状 `[M, D]`）；
  - 将每页的 M 条子向量展开为多行（`page_uid` 存为 `{原 page_uid}::mv{k}`），按 **每 200 页** 写一个分片到 `data/cache/colqwen_multi_embeddings_XXXX.parquet`。
- **4.2.2 写入 Milvus**：`scripts/03b_colqwen_upsert_only.py`
  - 读取所有 `colqwen_multi_embeddings_*.parquet`，按批写入 Milvus collection **`pages_colqwen`**。

```bash
uv run python scripts/03a_colqwen_encode_only.py
uv run python scripts/03b_colqwen_upsert_only.py
```

若希望 ColQwen 在单脚本内一气呵成（不落盘分片），也可使用：

```bash
uv run python scripts/03_build_indices.py --retrievers colqwen
```

（与 03a+03b 二选一即可，最终都是向 `pages_colqwen` 写入多向量。）

---

### 第 5 步：主检索实验（表 5-1）

- **脚本**：`scripts/04_run_retrieval_experiments.py`
- **作用**：
  - 在统一页面库上运行三种检索器：**BM25**、**BGE**、**ColQwen**；
  - **ColQwen** 内部为：ColQwen 多向量检索 + BM25 + BGE 三路结果做 **RRF（Reciprocal Rank Fusion）** 融合，再对融合后的候选集用 **BGE-large-zh（sentence-transformers）** 做精排，最后取 Top-k；
  - 对每个问题取 Top-k（k 由 `--topk_list` 指定，默认 1/3/5），结合 gold 计算 Recall@1/3/5、MRR；
  - 输出：`results/retrieval/{bm25,bge,colqwen}_raw_results.parquet`、`results/retrieval/table_5_1.csv`。
- **命令**：

```bash
uv run python scripts/04_run_retrieval_experiments.py --retrievers bm25 bge colqwen --topk_list 1 3 5
```

---

### 第 6 步：主问答实验（表 5-2 的输入）

- **脚本**：`scripts/05_run_qa_experiments.py`
- **作用**：
  - 固定回答模型为 **DeepSeek**（`config/api.yaml` 中 llm 配置），比较五种系统：
    - **qwen_only**：不检索，仅问题；
    - **bm25** / **bge** / **colqwen**：对应检索 Top-k 页文本 + DeepSeek；
    - **oracle**：直接使用 gold 证据页文本 + DeepSeek。
  - 默认 Top-k=3、并发数 20（`--num-workers 20`）；
  - 输出：`results/qa/{method}_answers.parquet`（含 `financebench_id`、`method`、`question`、`pred_answer`、`evidence_pages_used`、`evidence_pages_text` 等）。
- **命令**：

```bash
uv run python scripts/05_run_qa_experiments.py --methods qwen_only bm25 bge colqwen oracle --topk 3
```

---

### 第 7 步：DeepSeek 自动标注 → 表 5-2

- **脚本**：`scripts/09_auto_label_qa_with_deepseek.py`
- **作用**：对主问答实验产生的 `results/qa/{method}_answers.parquet` 调用 DeepSeek，按“问题 + 模型回答 + 证据文本”生成 **label**（correct/incorrect/insufficient/hallucinated）与 **evidence_supported**，写入 `results/annotations/qa_labels.csv`。
- **命令**：

```bash
uv run python scripts/09_auto_label_qa_with_deepseek.py --methods qwen_only bm25 bge colqwen oracle
```

- **脚本**：`scripts/10_compute_qa_metrics.py`
- **作用**：读取 `qa_labels.csv`，按 method 汇总 **Accuracy**、**SupportedAccuracy**，输出 **表 5-2**：`results/qa/table_5_2.csv`。
- **命令**：

```bash
uv run python scripts/10_compute_qa_metrics.py
```

---

### 第 8 步：Top-k 消融实验（表 5-3）

- **脚本**：`scripts/06_run_ablation_topk.py`
- **作用**：固定检索为 **ColQwen**（当前即 ColQwen+BM25+BGE 融合+精排），分别用 Top-1 / Top-3 / Top-5 证据页调用 DeepSeek 生成答案，输出 `results/qa/colqwen_top{k}_answers.parquet`。Top-k 列表由 `config/experiment.yaml` 的 **`ablation.colqwen_topk_variants`** 控制。
- **命令**：

```bash
uv run python scripts/06_run_ablation_topk.py
```

- **脚本**：`scripts/11_label_ablation_and_compute_table_5_3.py`
- **作用**：对上述 `colqwen_top{k}_answers.parquet` 使用 **DeepSeek** 自动标注，再按 top_k 汇总 Accuracy / SupportedAccuracy，写出 **表 5-3**：`results/qa/table_5_3.csv`；同时为每个 k 保存 `results/annotations/qa_labels_colqwen_top{k}.csv`。
- **命令**：

```bash
uv run python scripts/11_label_ablation_and_compute_table_5_3.py
```

---

### 第 9 步：按问题类型分析（表 5-4）

- **脚本**：`scripts/07_run_typewise_analysis.py`
- **前提**：已有 `results/retrieval/table_5_1` 与 `results/annotations/qa_labels.csv`（脚本会从 `questions.parquet` 合并 `question_type` 到标注表）。
- **作用**：按 **question_type**（如 metrics-generated、domain-relevant、novel-generated）分别计算 BGE 与 ColQwen 的 Recall@1/3/5、MRR，以及对应 QA 的 Accuracy、SupportedAccuracy，输出 **表 5-4**：`results/qa/table_5_4.csv`。
- **命令**：

```bash
uv run python scripts/07_run_typewise_analysis.py
```

---

### 第 10 步：典型案例与图表

- **案例筛选**：`scripts/08_select_cases.py`  
  结合检索结果与 gold，筛选“仅 ColQwen 命中”“BGE/BM25 排序靠后、ColQwen 更靠前”等典型 case，输出 `results/analysis/` 下 Markdown。
- **流程图**：若存在 `fbbench/analysis/plots.py`，可生成 Mermaid 等流程图到 `results/figures/`。

```bash
uv run python scripts/08_select_cases.py
```

---

## 表与脚本对应关系小结

| 输出 | 脚本 / 依赖 |
|------|-------------|
| **表 5-1**（检索 Recall/MRR） | `04_run_retrieval_experiments.py` → `results/retrieval/table_5_1.csv` |
| **表 5-2**（QA Accuracy/SupportedAccuracy） | `05_run_qa_experiments.py` → `09_auto_label_qa_with_deepseek.py` → `10_compute_qa_metrics.py` → `results/qa/table_5_2.csv` |
| **表 5-3**（ColQwen Top-k 消融） | `06_run_ablation_topk.py` → `11_label_ablation_and_compute_table_5_3.py` → `results/qa/table_5_3.csv` |
| **表 5-4**（按题型） | `07_run_typewise_analysis.py` → `results/qa/table_5_4.csv` |

按本 README 顺序执行脚本并完成自动标注与指标汇总，即可复现论文中第五章的主要表格与实验设置。
