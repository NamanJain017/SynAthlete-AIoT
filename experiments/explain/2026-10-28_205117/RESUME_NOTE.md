# Resumed run
- Started 20:51 at commit eb3e4de (clean); rounds 0-3 finished and were checkpointed (checkpoints/explain_r0..r3.json).
- Paused 21:17 (process stopped to free memory for the generator follow-up stage), before round 4 finished.
- Resumed 22:17 at commit f484cb6 (clean) with `python -m src.xai.run --fused 2026-10-28_060435 --resume 2026-10-28_205117 --workers 1`: rounds 0-3 read back from their checkpoints unchanged, round 4 computed, metrics.json written. run_info.json records the resume commit.
- `git diff --stat eb3e4de f484cb6`: only docs/AI_LOG.md, docs/DECISIONS.md, docs/EVAL_PROTOCOL.md and src/models/final.py changed; no file under src/xai, src/models/fused.py or configs changed, so all 5 rounds ran the same explanation code.
