from huggingface_hub import snapshot_download

repo_id = "McAuley-Lab/Amazon-Reviews-2023"

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=[
        "benchmark/0core/last_out_w_his/Amazon_Fashion.train.csv",
        "benchmark/0core/last_out_w_his/Amazon_Fashion.valid.csv",
        "benchmark/0core/last_out_w_his/Amazon_Fashion.test.csv",
        "raw/meta_categories/meta_Amazon_Fashion.jsonl",
    ],
    local_dir="data/amazon/raw",
)

print("Downloaded Amazon Fashion benchmark splits + metadata.")
