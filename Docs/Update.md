# Update a Document with Text

> Updates an existing document's content or settings using text.



## OpenAPI

````yaml en/api-reference/openapi_knowledge.json post /datasets/{dataset_id}/documents/{document_id}/update-by-text
openapi: 3.0.1
info:
  title: Knowledge API
  description: >-
    API for managing knowledge bases (datasets), documents, and segments,
    including creation, retrieval, and configuration.
  version: 1.0.0
servers:
  - url: '{apiBaseUrl}'
    description: The base URL for the Knowledge API.
    variables:
      apiBaseUrl:
        default: https://api.dify.ai/v1
        description: Actual base URL of the API
security:
  - ApiKeyAuth: []
tags:
  - name: Datasets
    description: Operations related to managing knowledge bases (datasets).
  - name: Documents
    description: >-
      Operations for creating, updating, and managing documents within a
      dataset.
  - name: Chunks
    description: Operations for managing document chunks (segments).
  - name: Metadata & Tags
    description: Operations for managing dataset tags and metadata.
  - name: Models
    description: Operations for retrieving available models.
paths:
  /datasets/{dataset_id}/documents/{document_id}/update-by-text:
    post:
      tags:
        - Documents
      summary: Update a Document with Text
      description: Updates an existing document's content or settings using text.
      operationId: updateDocumentByText
      parameters:
        - name: dataset_id
          in: path
          required: true
          description: The ID of the knowledge base containing the document.
          schema:
            type: string
            format: uuid
        - name: document_id
          in: path
          required: true
          description: The ID of the document to update.
          schema:
            type: string
            format: uuid
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UpdateDocumentByTextRequest'
      responses:
        '200':
          description: Document updated successfully.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/DocumentCreationResponse'
components:
  schemas:
    UpdateDocumentByTextRequest:
      type: object
      properties:
        name:
          type: string
          description: New name for the document (optional).
        text:
          type: string
          description: New text content for the document (optional).
        process_rule:
          $ref: '#/components/schemas/ProcessRule'
    DocumentCreationResponse:
      type: object
      properties:
        document:
          $ref: '#/components/schemas/Document'
        batch:
          type: string
          description: A batch identifier for tracking indexing progress.
    ProcessRule:
      type: object
      description: >-
        A set of rules for processing a document, including cleaning and
        segmentation.
      properties:
        mode:
          type: string
          description: 'The processing mode: automatic, custom, or hierarchical.'
          enum:
            - automatic
            - custom
            - hierarchical
        rules:
          type: object
          description: >-
            The specific rules to apply, used when mode is 'custom' or
            'hierarchical'.
          properties:
            pre_processing_rules:
              type: array
              items:
                $ref: '#/components/schemas/PreprocessingRule'
            segmentation:
              $ref: '#/components/schemas/SegmentationRule'
            parent_mode:
              type: string
              description: Retrieval mode for parent chunks in hierarchical mode.
              enum:
                - full-doc
                - paragraph
            subchunk_segmentation:
              $ref: '#/components/schemas/SubChunkSegmentationRule'
          nullable: true
    Document:
      type: object
      properties:
        id:
          type: string
          format: uuid
        position:
          type: integer
        data_source_type:
          type: string
        data_source_info:
          type: object
          nullable: true
        dataset_process_rule_id:
          type: string
          format: uuid
          nullable: true
        name:
          type: string
        created_from:
          type: string
        created_by:
          type: string
          format: uuid
        created_at:
          type: integer
          format: int64
        tokens:
          type: integer
        indexing_status:
          type: string
        error:
          type: string
          nullable: true
        enabled:
          type: boolean
        disabled_at:
          type: integer
          format: int64
          nullable: true
        disabled_by:
          type: string
          format: uuid
          nullable: true
        archived:
          type: boolean
        display_status:
          type: string
        word_count:
          type: integer
        hit_count:
          type: integer
        doc_form:
          type: string
    PreprocessingRule:
      type: object
      description: A rule for preprocessing document content.
      properties:
        id:
          type: string
          description: The unique identifier for the preprocessing rule.
          enum:
            - remove_extra_spaces
            - remove_urls_emails
        enabled:
          type: boolean
          description: Whether this rule is enabled.
    SegmentationRule:
      type: object
      description: Rules for segmenting document content into chunks.
      properties:
        separator:
          type: string
          description: The custom delimiter used to separate segments.
        max_tokens:
          type: integer
          description: The maximum number of tokens allowed in a single segment.
    SubChunkSegmentationRule:
      type: object
      description: >-
        Rules for segmenting parent chunks into smaller child chunks (for
        hierarchical mode).
      properties:
        separator:
          type: string
          description: The delimiter for sub-chunking.
        max_tokens:
          type: integer
          description: The maximum token length for a sub-chunk.
        chunk_overlap:
          type: integer
          description: The number of overlapping tokens between adjacent sub-chunks.
  securitySchemes:
    ApiKeyAuth:
      type: http
      scheme: bearer
      bearerFormat: API_KEY
      description: >-
        API Key authentication. For all API requests, include your API Key in
        the `Authorization` HTTP Header, prefixed with 'Bearer '. Example:
        `Authorization: Bearer {API_KEY}`. **Strongly recommend storing your API
        Key on the server-side, not shared or stored on the client-side, to
        avoid possible API-Key leakage that can lead to serious consequences.**

````

---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.dify.ai/llms.txt
