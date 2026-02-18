// uploadDocument.js
import fs from 'fs';
import fetch from 'node-fetch';

// ---------- CONFIG ----------
const API_TOKEN = 'YOUR_NEOMIND_API_KEY'; // Replace with your API token
const DATASET_ID = 'YOUR_DATASET_ID';     // Replace with your dataset ID
const FILE_PATH = './myDocument.txt';     // Path to your local text file
// ----------------------------

async function uploadDocument() {
  try {
    // Read text from file
    const text = fs.readFileSync(FILE_PATH, 'utf-8');

    // Prepare request body
    const body = {
      name: 'Uploaded from Node.js',
      text: text,
      indexing_technique: 'high_quality',
      doc_form: 'text_model',
      doc_language: 'English',
      process_rule: {
        mode: 'automatic',
        rules: {
          pre_processing_rules: [{ id: 'remove_extra_spaces', enabled: true }],
          segmentation: { separator: '\n', max_tokens: 500 },
          parent_mode: 'full-doc',
          subchunk_segmentation: { separator: '\n', max_tokens: 200, chunk_overlap: 50 }
        }
      },
      retrieval_model: {
        search_method: 'hybrid_search',
        reranking_enable: true,
        reranking_mode: 'reranking_model',
        reranking_model: {
          reranking_provider_name: 'openai',
          reranking_model_name: 'text-davinci-003'
        },
        top_k: 10,
        score_threshold_enabled: true,
        score_threshold: 0.7,
        weights: 1
      },
      embedding_model: 'text-embedding-ada-002',
      embedding_model_provider: 'openai'
    };

    // Send request
    const response = await fetch(`https://api.neomind.ai/v1/datasets/${DATASET_ID}/document/create-by-text`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${API_TOKEN}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(body)
    });

    const result = await response.json();
    console.log('Upload Result:', result);
  } catch (error) {
    console.error('Error uploading document:', error);
  }
}

// Run the script
uploadDocument();
