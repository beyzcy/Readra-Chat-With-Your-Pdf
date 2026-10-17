# Evaluation results

Embeddings `paraphrase-multilingual-MiniLM-L12-v2` + BM25 · reranker: `BAAI/bge-reranker-v2-m3-int8` · top 5 pages. Every rate shows its count and a 95% Wilson confidence interval: with few questions the interval is wide, and that is the honest size of the claim.

## Development set

Written while fixing known failures, so this set is optimistic by construction.

### Retrieval

| Metric | Result |
|---|---|
| Correct page ranked first | 100% (32/32; 95% CI 89-100%) |
| Correct page in the top 5 | 100% (32/32; 95% CI 89-100%) |
| Mean reciprocal rank | 1.00 |
| Non-English questions, page first | 100% (14/14; 95% CI 78-100%) |
| Evidence passage is the top chunk | 73% (8/11; 95% CI 43-90%) |
| Evidence passage in the top 3 chunks | 100% (11/11; 95% CI 74-100%) |

### Answers (`openai/gpt-oss-120b`, temperature 0)

| Metric | Result |
|---|---|
| Expected fact present | 100% (32/32; 95% CI 89-100%) |
| **A forbidden look-alike value appears** | 0% (0/32; 95% CI 0-11%) |
| Evidence quotes verified verbatim | 100% (32/32; 95% CI 89-100%) |
| Judge flagged a possible mix-up | 0% (0/32; 95% CI 0-11%) |
| A value absent from the cited text | 0% (0/32; 95% CI 0-11%) |
| Unanswerable questions refused | 100% (5/5; 95% CI 57-100%) |

By kind of question (correct and no forbidden value; for unanswerable: refused):

| Kind | Result |
|---|---|
| other | 100% (37/37; 95% CI 91-100%) |

## Held-out test set

Written once, not used to tune the pipeline. Two extraction bugs (a label and its value in different fonts were ordered wrongly; two text columns were read interleaved) were found by inspecting the extracted text of these documents and fixed *before any question was run*.

### Retrieval

| Metric | Result |
|---|---|
| Correct page ranked first | 96% (45/47; 95% CI 86-99%) |
| Correct page in the top 5 | 100% (47/47; 95% CI 92-100%) |
| Mean reciprocal rank | 0.98 |
| Non-English questions, page first | 100% (2/2; 95% CI 34-100%) |
| Evidence passage is the top chunk | 96% (45/47; 95% CI 86-99%) |
| Evidence passage in the top 3 chunks | 100% (47/47; 95% CI 92-100%) |

Not ranked first:

- “When is the payment due?” (invoice_en.pdf): rank 2
- “What is the late payment fee?” (contract_en.pdf): rank 2

### Answers (`openai/gpt-oss-120b`, temperature 0)

| Metric | Result |
|---|---|
| Expected fact present | 100% (41/41; 95% CI 91-100%) |
| **A forbidden look-alike value appears** | 0% (0/41; 95% CI 0-9%) |
| Evidence quotes verified verbatim | 100% (41/41; 95% CI 91-100%) |
| Judge flagged a possible mix-up | 0% (0/41; 95% CI 0-9%) |
| A value absent from the cited text | 2% (1/41; 95% CI 0-13%) |
| Unanswerable questions refused | 100% (5/5; 95% CI 57-100%) |

By kind of question (correct and no forbidden value; for unanswerable: refused):

| Kind | Result |
|---|---|
| column | 100% (10/10; 95% CI 72-100%) |
| computed | 100% (1/1; 95% CI 21-100%) |
| lookalike | 100% (10/10; 95% CI 72-100%) |
| ocr | 100% (4/4; 95% CI 51-100%) |
| table | 100% (4/4; 95% CI 51-100%) |
| text | 100% (12/12; 95% CI 76-100%) |
| unanswerable | 100% (5/5; 95% CI 57-100%) |

Failures:

- “Which GPU was used for training?”: the model call failed. Got: _RateLimitError…_
- “Who funded this research?”: the model call failed. Got: _RateLimitError…_
- “When does ICE 705 depart?”: the model call failed. Got: _RateLimitError…_
- “Which platform does ICE 583 leave from?”: the model call failed. Got: _RateLimitError…_
- “What is the destination of the 09:52 train?”: the model call failed. Got: _RateLimitError…_
- “Which train goes to Berlin?”: the model call failed. Got: _RateLimitError…_
- “When does the train to Hanover depart?”: the model call failed. Got: _RateLimitError…_
- “How long does the train to Bremen take?”: the model call failed. Got: _RateLimitError…_
- “Is the ICE 705 delayed today?”: the model call failed. Got: _RateLimitError…_
- “How much does a ticket to Munich cost?”: the model call failed. Got: _RateLimitError…_

_Correctness, forbidden-value and refusal checks are keyword heuristics, not human grading._
