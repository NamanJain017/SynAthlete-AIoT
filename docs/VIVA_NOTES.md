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

## Piece 0: setup
**What it does.** Prepares the laptop so every later step runs the same way every time.
- A **virtual environment** (`.venv`) is a private folder of Python packages just for this project, so other projects on the laptop cannot change our versions.
- **PyTorch with CUDA** lets the RTX 4050 GPU train the neural networks (generators, sequence models) much faster than the CPU.
- **`requirements.lock.txt`** lists the exact version of every installed package (70 of them), written by `pip freeze`. Anyone can rebuild the same environment later, so results can be reproduced.
- **`python tasks.py check`** runs ruff (finds code mistakes and style problems) and pytest (automatic tests). Green means the code base is in a known-good state.

**What the numbers mean.** `Python 3.12.2`; `2.14.1+cu132 True NVIDIA GeForce RTX 4050 Laptop GPU` means PyTorch version 2.14.1 built for CUDA 13.2, and `True` means it can see the GPU; `3 passed` means all three tooling tests pass (path guard and snapshot filter).

**Q: Why a lock file if you already have requirements.txt?**
A: requirements.txt says *which* packages we need; the lock file says *which exact versions* we had when results were produced. Libraries change behaviour between versions, so the lock file is what makes a result reproducible months later.

**Q: Your teammates have no GPU. Can they still run the project?**
A: Yes. They install the CPU build of PyTorch and requirements.txt. Only generator and model training need the GPU; data audit, splits, baselines, explanations and the dashboard run on CPU.

**Q: Why check `torch.cuda.is_available()`?**
A: On Windows a plain `pip install torch` silently installs the CPU-only build. Training would still run, just many times slower, so we check explicitly that PyTorch sees the GPU.
