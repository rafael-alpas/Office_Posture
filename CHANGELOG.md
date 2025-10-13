# Changelog

## 2025-10-13 08:36:00 - Inference Instrumentation
- change: Wrapped `infer_video.py` with diagnostics and landmark-refine flags.
- hypothesis: Capturing coverage and landmark-aware boxes will help tune stability without code rewrites.
- metrics: detection_rate_box=1.000, avg_box_area_fraction=0.620 (baseline run).
- decision: keep (await user evaluation).
- tweak: landmark refinement now clamps to min/max box fractions to avoid tiny crops.\n
