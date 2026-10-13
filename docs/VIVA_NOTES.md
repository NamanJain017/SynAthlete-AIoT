# VIVA NOTES

Plain-language explanation of every piece, added as each piece is built. Each section: what it does, why it is needed, what the numbers mean, likely questions with answers.

## The project in 30 seconds
Coaches can't see injury risk building up over days. We built a software-only AIoT system: wearable-style sensor data flows through three simulated tiers (watch, phone, server); a model trained on real runners' daily training logs with real injuries predicts risk; generative models (GANs) create extra realistic data for the rare injury cases; every prediction comes with a reason a coach can read.

## Likely questions (general)
**Q: Why is it "AIoT" if there is no hardware?**
A: The tiers (edge, fog, cloud) run as separate programs that pass messages, and we measure latency and data sent per tier. The May 2026 Applied Sciences AIoT framework we build on was also software-only.

**Q: Why not just split the data randomly?**
A: Windows from the same person look alike. A random split puts the same person in training and test, so the model "recognises" them and the score is inflated. We split by person, so the test is always on unseen athletes.

**Q: What changed since Review 1?**
A: We found a real dataset (Lövdal et al., 74 runners, 7 years, real injuries). So injury results are now measured on real injuries, and the GAN creates extra rare-injury examples instead of being the only source of injury data.

<!-- Add one section per piece below. -->
