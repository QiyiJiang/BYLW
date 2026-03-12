import argparse
from pathlib import Path

from tqdm import tqdm
from glob import glob

from fbbench.index.milvus_client import MilvusIndex, connect_milvus
from fbbench.utils.config_loader import load_yaml_with_env
from fbbench.utils.io_utils import read_table
from fbbench.utils.logging_utils import setup_logger


def main() -> None:
    """
    第二步：从磁盘读取 ColQwen embedding 结果分片，并批量写入 Milvus。

    输入：data/cache/colqwen_multi_embeddings_*.parquet
    输出：Milvus collection 'pages_colqwen'
    """
    parser = argparse.ArgumentParser(description="Upsert ColQwen embeddings from disk into Milvus only.")
    parser.add_argument("--paths-config", type=str, default="config/paths.yaml")
    args = parser.parse_args()

    logger = setup_logger("colqwen_upsert_only")

    paths_cfg = load_yaml_with_env(Path(args.paths_config))
    cache_dir = Path(paths_cfg["data"]["cache_dir"])
    shard_paths = sorted(cache_dir.glob("colqwen_multi_embeddings_*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(f"No embedding shards found in {cache_dir}")

    milvus_db_path = Path(paths_cfg["milvus"]["db_path"])
    milvus_client = connect_milvus(milvus_db_path)
    milvus_index = MilvusIndex(milvus_client)

    batch_size = 200
    total_rows = 0

    for shard_path in shard_paths:
        emb_df = read_table(shard_path)
        logger.info("Loaded shard %s (rows=%d)", shard_path, len(emb_df))

        chunk_uids = emb_df["chunk_uid"].astype(str).tolist()
        doc_names = emb_df["doc_name"].astype(str).tolist()
        page_ids = emb_df["page_id"].astype(int).tolist()
        vectors = emb_df["embedding"].tolist()
        total_rows += len(emb_df)

        for i in tqdm(
            range(0, len(chunk_uids), batch_size),
            desc=f"Upserting {shard_path.name}",
            total=(len(chunk_uids) + batch_size - 1) // batch_size,
        ):
            batch_uids = chunk_uids[i : i + batch_size]
            batch_docs = doc_names[i : i + batch_size]
            batch_pids = page_ids[i : i + batch_size]
            batch_vecs = vectors[i : i + batch_size]

            milvus_index.upsert_vectors(
                collection_name="pages_colqwen",
                page_uids=batch_uids,
                doc_names=batch_docs,
                page_ids=batch_pids,
                vectors=batch_vecs,
            )

    logger.info("ColQwen embeddings upserted to Milvus collection 'pages_colqwen' (total rows=%d)", total_rows)


if __name__ == "__main__":
    main()

