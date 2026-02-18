curl --request POST \
  --url https://api.neomind.ai/v1/datasets/{dataset_id}/document/create-by-text \
  --header 'Authorization: Bearer <token>' \
  --header 'Content-Type: application/json' \
  --data '
{
  "name": "<string>",
  "text": "<string>",
  "indexing_technique": "high_quality",
  "doc_form": "text_model",
  "doc_language": "English",
  "process_rule": {
    "mode": "automatic",
    "rules": {
      "pre_processing_rules": [
        {
          "id": "remove_extra_spaces",
          "enabled": true
        }
      ],
      "segmentation": {
        "separator": "<string>",
        "max_tokens": 123
      },
      "parent_mode": "full-doc",
      "subchunk_segmentation": {
        "separator": "<string>",
        "max_tokens": 123,
        "chunk_overlap": 123
      }
    }
  },
  "retrieval_model": {
    "search_method": "hybrid_search",
    "reranking_enable": true,
    "reranking_mode": "reranking_model",
    "reranking_model": {
      "reranking_provider_name": "<string>",
      "reranking_model_name": "<string>"
    },
    "top_k": 123,
    "score_threshold_enabled": true,
    "score_threshold": 123,
    "weights": 123
  },
  "embedding_model": "<string>",
  "embedding_model_provider": "<string>"
}
'
