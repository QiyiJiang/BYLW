import argparse
from pathlib import Path

from tqdm import tqdm
import torch
from PIL import Image
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from sentence_transformers import SentenceTransformer

from fbbench.utils.config_loader import load_yaml_with_env
from fbbench.index.milvus_client import MilvusIndex, connect_milvus
from fbbench.utils.io_utils import read_table
from fbbench.utils.logging_utils import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute embeddings / indices for BGE and ColQwen.")
    parser.add_argument("--paths-config", type=str, default="config/paths.yaml")
    parser.add_argument("--api-config", type=str, default="config/api.yaml")
    parser.add_argument(
        "--retrievers",
        nargs="+",
        default=["bm25", "bge", "colqwen"],
        help="Retrievers to prepare (bm25/bge/colqwen). BM25 无需额外预处理，这里主要处理 bge/colqwen 向量。",
    )
    args = parser.parse_args()

    logger = setup_logger("build_indices")

    paths_cfg = load_yaml_with_env(Path(args.paths_config))
    api_cfg = load_yaml_with_env(Path(args.api_config))

    pages_path = Path(paths_cfg["tables"]["pages"])
    pages_df = read_table(pages_path)
    logger.info("Loaded pages: %d", len(pages_df))

    # 连接本地 milvus-lite
    milvus_db_path = Path(paths_cfg["milvus"]["db_path"])
    milvus_client = connect_milvus(milvus_db_path)
    milvus_index = MilvusIndex(milvus_client)

    if "bge" in args.retrievers:
        logger.info("Precomputing BGE-M3 chunk embeddings locally (SentenceTransformer) and writing to Milvus...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        bge_model = SentenceTransformer("BAAI/bge-m3", device=device)

        MAX_CHARS_PER_CHUNK = 1000

        chunk_texts = []
        chunk_uids = []
        chunk_docs = []
        chunk_pids = []

        for _, row in pages_df.iterrows():
            page_uid = str(row["page_uid"])
            doc_name = str(row["doc_name"])
            page_id = int(row["page_id"])
            text = str(row["page_text"] or "")

            # 按字符切成最多 1000 字符的 chunk
            if not text:
                chunks = [""]
            else:
                chunks = [text[i : i + MAX_CHARS_PER_CHUNK] for i in range(0, len(text), MAX_CHARS_PER_CHUNK)]

            for idx, ch in enumerate(chunks):
                chunk_id = f"{page_uid}::c{idx}"
                chunk_texts.append(ch)
                chunk_uids.append(chunk_id)
                chunk_docs.append(doc_name)
                chunk_pids.append(page_id)

        batch_size = 32
        for i in tqdm(
            range(0, len(chunk_texts), batch_size),
            desc="BGE embedding batches",
            total=(len(chunk_texts) + batch_size - 1) // batch_size,
        ):
            batch_texts = chunk_texts[i : i + batch_size]
            batch_uids = chunk_uids[i : i + batch_size]
            batch_docs = chunk_docs[i : i + batch_size]
            batch_pids = chunk_pids[i : i + batch_size]

            vectors = bge_model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                normalize_embeddings=False,
            ).tolist()

            milvus_index.upsert_vectors(
                collection_name="pages_bge",
                page_uids=batch_uids,
                doc_names=batch_docs,
                page_ids=batch_pids,
                vectors=vectors,
            )

        logger.info("BGE embeddings written to Milvus collection 'pages_bge' (chunk-level)")

    if "colqwen" in args.retrievers:
        logger.info("Precomputing ColQwen (ColQwen2.5) multi-vector embeddings and writing to Milvus...")

        # 准备 ColQwen 模型与处理器
        model_name = "vidore/colqwen2.5-v0.2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ColQwen2_5.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
        ).eval()
        processor = ColQwen2_5_Processor.from_pretrained(model_name)

        page_uids = pages_df["page_uid"].astype(str).tolist()
        doc_names = pages_df["doc_name"].astype(str).tolist()
        page_ids = pages_df["page_id"].astype(int).tolist()
        img_paths = pages_df["page_image_path"].astype(str).tolist()

        batch_size = 8
        for i in tqdm(
            range(0, len(img_paths), batch_size),
            desc="ColQwen image batches",
            total=(len(img_paths) + batch_size - 1) // batch_size,
        ):
            batch_paths = img_paths[i : i + batch_size]
            batch_uids = page_uids[i : i + batch_size]
            batch_docs = doc_names[i : i + batch_size]
            batch_pids = page_ids[i : i + batch_size]

            # 加载并预处理图片
            images = []
            for p in batch_paths:
                try:
                    with Image.open(p) as img:
                        images.append(img.convert("RGB"))
                except Exception:
                    # 若图片加载失败，用空白图占位，避免中断整个流程
                    images.append(Image.new("RGB", (224, 224), color="white"))

            batch_images = processor.process_images(images).to(model.device)
            with torch.no_grad():
                embs = model(**batch_images)  # [B, M, D] multi-vector

            # 将每个页面的 multi-vector 展开成多条向量，page_uid 编成 chunk 风格
            multi_page_uids = []
            multi_docs = []
            multi_pids = []
            multi_vecs = []
            for j in range(embs.shape[0]):  # 遍历 batch 内页面
                page_uid = batch_uids[j]
                doc_name = batch_docs[j]
                page_id = batch_pids[j]
                vecs = embs[j]  # [M, D]
                for k in range(vecs.shape[0]):
                    chunk_id = f"{page_uid}::mv{k}"
                    multi_page_uids.append(chunk_id)
                    multi_docs.append(doc_name)
                    multi_pids.append(page_id)
                    multi_vecs.append(vecs[k].detach().cpu().tolist())

            milvus_index.upsert_vectors(
                collection_name="pages_colqwen",
                page_uids=multi_page_uids,
                doc_names=multi_docs,
                page_ids=multi_pids,
                vectors=multi_vecs,
            )

        logger.info("ColQwen embeddings written to Milvus collection 'pages_colqwen'")


if __name__ == "__main__":
    main()

