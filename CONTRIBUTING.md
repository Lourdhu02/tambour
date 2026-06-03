# Contributing

## Dev setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -r requirements.txt -e ".[dev]"
python tests/test_engine.py        # all tests run on CPU with synthetic data
```

## Conventions

- Tests must stay **CPU-only and dataset-free** (synthetic fixtures) so CI is fast.
- Keep the model ONNX-exportable: no dynamic-output-size ops (e.g. adaptive pools),
  collapse height with `mean`, keep CTC decoding outside the graph.
- New features land behind config flags, defaulting off, so the baseline stays stable.
- One logical change per commit; conventional-commit prefixes (`feat:`, `fix:`, …).

## Layout

`config` · `text` (codec) · `data/` (manifest, transforms, dataset, sampler, shards) ·
`models` · `losses` · `engine` · `confidence` · `infer` · `export` · `agents` · `__main__` (CLI).
