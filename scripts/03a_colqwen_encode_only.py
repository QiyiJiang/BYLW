import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
import pandas as pd

from fbbench.utils.config_loader import load_yaml_with_env
from fbbench.utils.io_utils import read_table, write_table, ensure_dir
from fbbench.utils.logging_utils import setup_logger


def main() -> None:
    """
    第一步：仅对页面图片做 ColQwen2.5 embedding，并将 multi-vector 结果分片落盘，不写入 Milvus。

    输出：data/cache/colqwen_multi_embeddings_*.parquet（每片约 200 个页面的所有向量，支持断点续传）
    """
    parser = argparse.ArgumentParser(description="Encode pages with ColQwen2.5 and save embeddings to disk (no Milvus).")
    parser.add_argument("--paths-config", type=str, default="config/paths.yaml")
    args = parser.parse_args()

    logger = setup_logger("colqwen_encode_only")

    paths_cfg = load_yaml_with_env(Path(args.paths_config))
    pages_path = Path(paths_cfg["tables"]["pages"])
    cache_dir = Path(paths_cfg["data"]["cache_dir"])
    ensure_dir(cache_dir)
    # 分片文件前缀
    shard_prefix = cache_dir / "colqwen_multi_embeddings_"

    pages_df = read_table(pages_path)
    logger.info("Loaded pages: %d", len(pages_df))

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

    rows = []
    shard_idx = 0
    PAGES_PER_SHARD = 200  # 每 200 个页面写一片
    pages_in_current_shard = 0

    batch_size = 32
    for i in tqdm(
        range(0, len(img_paths), batch_size),
        desc="ColQwen encode batches",
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
                logger.warning("Failed to load image %s", p)
                images.append(Image.new("RGB", (224, 224), color="white"))

        batch_images = processor.process_images(images).to(model.device)
        with torch.no_grad():
            embs = model(**batch_images)  # [B, M, D] multi-vector

        for j in range(embs.shape[0]):  # 遍历 batch 内页面
            page_uid = batch_uids[j]
            doc_name = batch_docs[j]
            page_id = batch_pids[j]
            vecs = embs[j]  # [M, D]
            m = vecs.shape[0]
            for k in range(m):
                chunk_id = f"{page_uid}::mv{k}"
                rows.append(
                    {
                        "chunk_uid": chunk_id,
                        "page_uid": page_uid,
                        "doc_name": doc_name,
                        "page_id": int(page_id),
                        "vec_idx": int(k),
                        "embedding": vecs[k].detach().cpu().tolist(),
                    }
                )

            # 一个页面处理完，计数 +1
            pages_in_current_shard += 1
            # 每 PAGES_PER_SHARD 个页面，写出当前 shard
            if pages_in_current_shard >= PAGES_PER_SHARD:
                shard_path = Path(f"{shard_prefix}{shard_idx:04d}.parquet")
                shard_df = pd.DataFrame(rows)
                write_table(shard_df, shard_path)
                logger.info("Wrote shard %s (pages=%d, rows=%d)", shard_path, pages_in_current_shard, len(shard_df))
                rows = []
                pages_in_current_shard = 0
                shard_idx += 1

    # 写出最后一片
    if rows:
        shard_path = Path(f"{shard_prefix}{shard_idx:04d}.parquet")
        shard_df = pd.DataFrame(rows)
        write_table(shard_df, shard_path)
        logger.info("Wrote final shard %s (pages=%d, rows=%d)", shard_path, pages_in_current_shard, len(shard_df))


if __name__ == "__main__":
    main()

