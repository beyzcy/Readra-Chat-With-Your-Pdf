# Evaluation results

32 answerable + 5 unanswerable questions over 2 documents (a 12-page handbook and a one-page Turkish flight booking whose labels and values are stored in separate column streams) · English, Turkish, German · embeddings `paraphrase-multilingual-MiniLM-L12-v2` + BM25 · reranker: off · top 5 pages

## Retrieval

| Metric | Score |
|---|---|
| Page Hit@1 | 97% |
| Page Hit@5 | 100% |
| Page MRR | 0.98 |
| Page Hit@1 (non-English questions) | 93% |
| Evidence-chunk Hit@1 | 64% |
| Evidence-chunk Hit@3 | 100% |

_Evidence-chunk Hit@k: the passage containing the exact answer is among the top k retrieved chunks._

Not ranked first:

- “Uçuş ne zaman?” → evidence found at rank 2
- “Uçuş saati kaç?” → evidence found at rank 2
- “Berlin'e hangi saatte varıyorum?” → evidence found at rank 2
- “Gidiş uçuş numaram nedir?” → evidence found at rank 2

_Answer-level metrics need `GROQ_API_KEY`; run `python eval/run_eval.py` with a key to add them._
