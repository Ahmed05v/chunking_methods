# Arabic chunking pilot corpus

This directory contains 11 **Arabic Wikipedia article collections** prepared for
an Arabic-language pilot of the chunking pipeline. They are assembled from real
Arabic text and are **not** the original CLAIR documents, natural single-source
reports, or a direct reproduction of the paper's Arabic results.

The mix follows the 33-document English experiment approximately: 3 technology,
5 legal, and 3 social collections. Length targets use the English domain means
and the same `o200k_base` tokenizer: about 5,257, 30,895, and 79,862 tokens per
collection. Actual sizes and source-article URLs/revision IDs are in
[`manifest.json`](manifest.json). The 11 parsed documents are in
[`adi_parsed/`](adi_parsed/). All `pages` are *synthetic*, formed by grouping
paragraphs into about 2,000-token segments; `split_points` mark paragraph
boundaries. A page-based score therefore measures these artificial pages, and
block-integrity scores use paragraph boundaries as a proxy rather than the
original PDF parser's blocks.

The article text is from Arabic Wikipedia and is licensed separately under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The manifest
provides attribution via article titles, URLs, and revision IDs. Excerpts and
collections have been adapted for this pilot. This dataset is **not** covered by
the repository's MIT code license. The source cache is local and ignored by Git.

To rebuild from current Wikipedia content:

```powershell
$env:PYTHONUTF8="1"
.\.venv\Scripts\python.exe scripts\build_arabic_pilot.py
```

To run the supported, inexpensive chunkers on the Arabic documents:

```powershell
$env:PYTHONUTF8="1"
$env:STANZA_RESOURCES_DIR=(Resolve-Path .).Path + '\.stanza_resources'
.\.venv\Scripts\python.exe -c "import stanza; stanza.download('ar', processors='tokenize')"
.\.venv\Scripts\python.exe -m adaptive_chunking.paper.replicate `
  --data-dir data/arabic_pilot --output-dir results/arabic_pilot `
  --steps chunking --language ar --device cpu --skip-semantic `
  --approximate-llm-regex
```

This command uses a local **heading regex approximation**, not the paper's LLM
regex method. ICC and DCC pilot scores were subsequently computed with local
Qwen embeddings; see `results/arabic_pilot/intrinsic_vs_paper.csv`. They are
not directly comparable to the paper's Jina-based scores.
The provided English coreference model is unsuitable for Arabic, so reference
completeness cannot be claimed without an Arabic coreference system and checked
annotations. The corpus also has no Arabic evaluation questions or gold answers;
RAG accuracy needs those generated and verified first.

Eleven collections are enough to confirm that the methods run and to locate
Arabic-specific failures. For credible, more stable comparisons, use at least
33 **independent, natural Arabic documents** with matched topic and length
distributions, and report confidence intervals. Matching token lengths alone
does not make these assembled collections equivalent to the paper corpus.
