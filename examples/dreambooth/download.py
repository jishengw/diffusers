from huggingface_hub import snapshot_download
local_dir = "./flickr30k"
snapshot_download(    "nlphuji/flickr30k",
                      local_dir=local_dir,
                      repo_type="dataset",
                      ignore_patterns=".gitattributes",)
