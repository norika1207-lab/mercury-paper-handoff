# Mercury Paper Handoff

Working notes, analysis scripts, and observation data for the Mercury LLM observability project.

This repository contains the per-paper handoff drafts, analysis tools, and Tier-A / Tier-B / Tier-C observation outputs used to write the Mercury paper series.

## Papers on Zenodo

- Mercury method (paper 1): [10.5281/zenodo.20313154](https://doi.org/10.5281/zenodo.20313154)
- Mercury-Viewer (paper 4): [10.5281/zenodo.20313150](https://doi.org/10.5281/zenodo.20313150)
- Discovery (paper 2): [10.5281/zenodo.20313748](https://doi.org/10.5281/zenodo.20313748)
- Scaling (paper 3): [10.5281/zenodo.20325676](https://doi.org/10.5281/zenodo.20325676)
- Truth contract (paper 5): [10.5281/zenodo.20313796](https://doi.org/10.5281/zenodo.20313796)
- Cross-model dim (paper A, qwen family): [10.5281/zenodo.20346626](https://doi.org/10.5281/zenodo.20346626)
- Functional control (paper B): [10.5281/zenodo.20347062](https://doi.org/10.5281/zenodo.20347062)
- TIES merge (paper D): [10.5281/zenodo.20346628](https://doi.org/10.5281/zenodo.20346628)
- Viz suite (paper E): [10.5281/zenodo.20346638](https://doi.org/10.5281/zenodo.20346638)
- Mercury MCP v0.1: [10.5281/zenodo.20352085](https://doi.org/10.5281/zenodo.20352085)

## Folder layout

- `paper-A-cross-arch/` — cross-architecture Tier-B / Tier-C analysis (in progress, reviewer-driven A1 / A2 split)
- `paper-B-functional-control/` — single-dim subspace ablation paper draft
- `paper-C-merge-failures/` — 8-method cross-fine-tune merge negative results
- `paper-D-tier-b-layer-map/` — per-layer anchor evolution across model scale
- `paper-E-100m-observability/` — 100M-cell grid demo on 70B-scale models
- `paper-G-composable-llm/` — composable LLM concept paper + engineering tools
- `tier-b-data/` — raw Tier-B residual-stream observation binaries
- `tier-c-data/` — Tier-C per-head attention fingerprint observations
- `layer-fingerprints/` — Tier-C cross-model layer fingerprint analyses
- `*.py` — analysis scripts

## License

- Code: MIT
- Data and observation outputs: CC-BY-4.0
- ORCID: 0009-0006-6816-9891
- Maintainer: norika (Chen Ho Yiing), Independent Researcher, Taiwan
