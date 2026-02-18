curl --request POST \
  --url https://api.neomind.ai/v1/datasets/abc123/document/create-by-text \
  --header 'Authorization: Bearer my_secret_token' \
  --header 'Content-Type: application/json' \
  --data '
{
  "name": "My First Doc",
  "text": "This is the content of my first document in Neomind AI.",
  "indexing_technique": "high_quality",
  "doc_form": "text_model",
  "doc_language": "English",
  "process_rule": {
    "mode": "automatic",
    "rules": {
      "pre_processing_rules": [
        { "id": "remove_extra_spaces", "enabled": true }
      ],
      "segmentation": { "separator": "\n", "max_tokens": 500 },
      "parent_mode": "full-doc",
      "subchunk_segmentation": { "separator": "\n", "max_tokens": 200, "chunk_overlap": 50 }
    }
  },
  "retrieval_model": {
    "search_method": "hybrid_search",
    "reranking_enable": true,
    "reranking_mode": "reranking_model",
    "reranking_model": {
      "reranking_provider_name": "openai",
      "reranking_model_name": "text-davinci-003"
    },
    "top_k": 10,
    "score_threshold_enabled": true,
    "score_threshold": 0.7,
    "weights": 1
  },
  "embedding_model": "text-embedding-ada-002",
  "embedding_model_provider": "openai"
}
'
