# Golden sets

One JSON file per app, named `<app_name>.json` (e.g. `portfolio.json`,
`doclens.json`). Files ending in `.example.json` are templates and are
skipped by `scripts/eval.py`'s default run — copy one, drop `.example`, and
fill in real `doc_id`s from that app's actual ingested content.

```json
{
  "app_name": "portfolio",
  "client_prompt": "The exact client_prompt this app sends to /v1/chat — needed only for cases with check_faithfulness: true.",
  "cases": [
    {
      "id": "short-slug-for-this-case",
      "query": "What do you do for work?",
      "namespace": null,
      "doc_ids": null,
      "expected_doc_ids": ["profile-notes"],
      "expected_keywords": ["backend", "Microchip"],
      "check_faithfulness": true
    }
  ]
}
```

Field notes:

- `query` — the only required field per case. Everything else is optional;
  omit a field rather than leaving it `null` if you're not scoring on it.
- `expected_doc_ids` — the `doc_id`(s) that *should* show up in the
  retrieved chunks for this query. Drives `recall` and `mrr`. Find real
  `doc_id`s by calling `/v1/retrieve` against the live collection, or from
  whatever ingested the document (Portfolio/DocLens/Admin all choose their
  own `doc_id`s at ingest time).
- `expected_keywords` — a cheap fallback for cases where you don't want to
  pin an exact `doc_id` yet (e.g. the content is still evolving). Drives
  `keyword_coverage`: the fraction of these terms found anywhere in the
  retrieved chunk text.
- `check_faithfulness` — if `true`, the harness generates a full answer the
  same way `/v1/chat` would (same system-prompt assembly) and has a
  separate LLM call judge whether every claim in it is grounded in the
  retrieved chunks. Requires the golden file's top-level `client_prompt` to
  be set. Costs one extra LLM call per case — skip with
  `--skip-faithfulness` for a fast, free retrieval-only run.
- `doc_ids` / `namespace` — pass through to `/v1/retrieve`'s own scoping if
  this app uses them.

A case with neither `expected_doc_ids` nor `expected_keywords` still runs
(useful to eyeball retrieved chunks / catch a hard error) but scores 1.0 on
both by convention rather than penalizing something it was never asked to
check.
