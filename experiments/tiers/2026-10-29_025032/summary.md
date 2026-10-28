# Piece-6 tiers, run 2026-10-29_025032

simulated on one laptop; wearable person and runner are different people (systems demo, not injury evidence)

| Stream | Mode | Windows | Bytes into cloud | Median / p95 decision latency (ms) |
|---|---|---|---|---|
| pamap2 | tiered | 329 | 2,932 | 4.25 / 5.86 |
| pamap2 | cloud_only | 329 | 1,739,186 | 2.77 / 4.37 |
| pamap2 | reduction | | 593x fewer | |
| sportdb2_hr | tiered | 200 | 1,600 | 0.78 / 1.20 |
| sportdb2_hr | cloud_only | 200 | 135,619 | 1.00 / 1.55 |
| sportdb2_hr | reduction | | 85x fewer | |
