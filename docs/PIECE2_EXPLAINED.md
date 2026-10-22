# Piece 2 explained simply (beginner-friendly)

This file explains piece 2 in easy words, with everyday examples. Read it top to bottom once; then use it to revise before reviews and the viva.

Every number here comes from a file our code wrote. The file is named under each section, so you can always point to where a number came from.

---

## First, a few words you need

| Word | What it means here | Everyday picture |
|---|---|---|
| **Dataset** | A collection of recorded data | A big box of notebooks |
| **Runner** (or athlete) | One person in the Lövdal data. There are 74 | One student in a class |
| **Event** | One row of data: one runner on one day, with the days before it | One page in a student's diary |
| **Label** | The answer we want to predict: injury (1) or no injury (0) | The answer key |
| **Model** | A computer program that learns patterns from examples | A student who studies |
| **Training data** | What the model learns from | The textbook |
| **Validation data** | What we use to tune the model | The practice test |
| **Test data** | What we use for the honest score | The real exam |

**What is piece 2?** Before you cook, you wash and cut the vegetables. Before you set an exam, you decide which questions are practice and which are on the real exam. Piece 2 did both for our data:
1. **Cleaning and sorting the data** (loaders), so models can read it.
2. **Deciding which people are textbook, practice test or real exam** (splits), so nobody can cheat.

---

## 1. Loaders: turning messy files into neat tables

**In one sentence:** a loader reads a raw data file and turns it into a neat block of numbers that a model can read.

**Explain it like I'm new.** Imagine a shoebox full of receipts in random order. Before you can add them up, you sort them by person and by date and put them in neat rows. A loader is that sorting machine for data files.

**What our Lövdal loader makes**
- The **day file**. Each event looks at the **7 days before** that day. For each of those days there are **10 numbers**: how many sessions, total km, km at hard speeds, sprint km, strength training, hours of other sports, and three feelings (how hard it felt, how well it went, how recovered the runner felt). So one event is a small table of **7 rows × 10 numbers**.
- The **week file**. Each event looks at the **3 weeks before**. Each week has **22 summary numbers**, plus 3 numbers that compare the weeks' distances.
- **How much data:**
  - the whole day file has **42,766** events, and only **583** are injuries (**1.36 %**);
  - the whole week file has **42,798** events with **575** injuries.
  - So injuries are **rare**: fewer than 2 in every 100 events.

**What the loader does NOT do (on purpose)**
- It doesn't "fix" or rescale the numbers, and it doesn't fill in gaps. Those steps must be learned from the **training runners only**, later. If the loader did them using everybody, information from the exam runners would leak into the textbook.
- A strange value, **-0.01**, means "rest day". We keep it and add a separate yes/no flag saying "this was a rest day", so nothing is hidden.

**The same idea for the other datasets.** Sensor data becomes short clips (section 6). SoccerMon and PMData become one row per person per day (section 7). The demo tables are saved as clean tables.

**Where the outputs live:** `data/processed/` on the laptop only. That folder isn't put on GitHub, because the raw data stays private.

*Numbers from:* `data/processed/lovdal/facts.json`, `docs/data_cards/lovdal.md`.

**Viva question:** *"Why doesn't your loader normalise the data?"*
**Answer:** "Normalising uses averages. If we computed averages using the exam runners too, the model would get a peek at the exam. So normalising happens later, using training runners only."

---

## 2. Time order: the trap in the two files

**In one sentence:** the two Lövdal files count time in **opposite directions**, so our loader turns them both the same way round: oldest day first, newest day last.

**Explain it like I'm new.** Imagine two comic strips. In one, the pictures go left to right. In the other, right to left. If you read both the same way, one story makes no sense: people would "un-break" a glass. A sequence model reads data like a comic strip, so the direction matters.

**The trap, exactly (from the authors' own README)**
- **Day file:** the column with no number is **7 days before** the event, and the column marked **.6** is **the day before**. Increasing numbers already go oldest → newest, so we keep them.
- **Week file:** the column with no number is **the week before** the event, and **.2** is **three weeks before**. That's newest → oldest, so we **flip** it.
- Even the paper itself described the day file the wrong way round once; the authors corrected it in their README.

**How we proved our order is right, using the data itself**
- If day *d* and day *d+1* belong to the same runner, then day *d+1*'s 7-day window should be day *d*'s window **slid by one day**.
- We checked this on more than **40,000** pairs of consecutive days, and it matched **exactly**.
- For the week file, one column is "this week's km ÷ last week's km". It matches our newest two weeks.

These checks are automatic tests, so if anyone breaks the order later, the tests fail.

*Checked by:* `tests/test_loaders.py` (real-data tests).

**Viva question:** *"How do you know which column is the most recent day?"*
**Answer:** "The authors' README says so, and we proved it on the data: each day's window equals the previous day's window shifted by one day, for over 40,000 pairs."

---

## 3. Splits: who is textbook, who is practice test, who is real exam

**In one sentence:** we split by **person**, so a runner is entirely in training, practice or exam, never in two of them.

**Explain it like I'm new.** Suppose a teacher grades handwriting tests. If Aisha's Monday page is in the textbook and her Tuesday page is on the exam, a student could "recognise Aisha's handwriting" instead of learning to read. Days from the same runner look alike in the same way. So if one runner's days were split between training and exam, the model would recognise the runner, and the score would look better than it really is. That's why we split by person (rule D-002).

**How the 74 Lövdal runners are split**
```
74 runners
├── 10 runners (IDs 64-73)  →  LOCKED FINAL EXAM (section 4)
└── 64 runners (IDs 0-63)   →  DEVELOPMENT, used in 5 rounds
       each round:  12-14 runners = exam for that round
                    9-10 runners  = practice test
                    41-43 runners = textbook (training)
```
- Every development runner is on the exam in **exactly one** round. Over the 5 rounds, all 64 are tested once.
- **Fair rounds:** each round's exam has a similar number of injuries: **105, 108, 108, 105 and 107**. Our rule says each round must be within 20 % of the average, otherwise we stop and ask.
- **The same runner groups** are used for the day file, the week file and the combined model, so all three are judged on the same people.
- **The final model** learns from 51 runners, practises on 13, and takes the locked exam once.

**One thing we say openly.** Practice runners can repeat across rounds: 46 practice places are filled by 30 different runners. For example, runner 20 is on the practice test in rounds 0, 2 and 4 but on the exam in round 1. That's fine *as long as* each round tunes only with its own practice runners. So that is our rule (D-021). A test pins these numbers.

**The other datasets, in short**

| Dataset | How it is split | Why |
|---|---|---|
| PAMAP2 (9 people) | 4 exam groups of 2 people; person 109 always in the textbook | 109 has almost no data from the main activity protocol |
| WESAD (15 people) | 5 exam groups of 3 people | simple and balanced |
| Sport DB 2.0 | 5 groups by person; the 4 skating recordings stay together | we can't tell whether they are one person |
| SoccerMon (50 players) | leave-one-out: train on 49, test on 1, repeat for everyone | pain reports come from only a few players |
| PMData (15 people) | leave-one-out; person p08 left out (their injury file is missing) | very few people |
| SIRP-600 | 80 % / 20 % of rows, "demo only" | it has no person IDs at all |

*Numbers from:* `data/splits/lovdal_folds.json` (and the other split files), checked by `tests/test_splits.py`.

**Viva question:** *"Why not just split the rows randomly?"*
**Answer:** "Rows from the same runner look alike. A random split puts the same runner in training and test, so the model recognises them and the score is inflated. We split by person."

---

## 4. The locked 10 runners: the sealed final exam

**In one sentence:** 10 runners are locked away and used **once, at the very end**, so we get one score nobody could have tuned.

**Explain it like I'm new.** A school prints the final exam and locks it in a safe. Teachers can't look at it while they teach, or they'd "teach to the test". The safe opens only on exam day, in front of witnesses, and only once.

**Why these 10 runners?**
- They are **exactly the paper's test runners** (IDs 64-73, the 10 who joined the team last).
- In the day file they hold **50 injuries and 2,994 healthy events**, the same counts as the paper. So at the end we can compare with the paper fairly.
- They joined last, so the exam also asks "does it work on newer athletes?" (Not "future data": just different runners who joined later.)

**How we keep the safe shut**
1. Their data is saved in a **separate folder** (`data/processed/lovdal_lockbox/`).
2. Only **one piece of code** can open it (`src/data/lockbox.py`). It refuses unless:
   - the project has an official, signed **"eval-freeze"** marker that is also on GitHub;
   - no unfinished code is lying around;
   - it has never been opened at a different version of the code.
3. Every opening is written in a **log book** (`experiments/lockbox_access.log`).
4. **Alarms (tests)** go off if any other code tries to read these runners, or if they ever turn up in the development data.

**Honest note:** the raw file still contains these runners, so this is a strong guard rail plus a log book, not an unbreakable safe. We say so.

**Main score vs final check (your decision, D-017)**
- **Main score:** the average over the 5 development rounds (533 injuries, so it's more stable).
- **Final check:** the locked 10 runners (50 injuries), opened once at the end, **even if Review 2 comes first**. At Review 2 we show the development results and say "the final exam is still sealed". Examiners usually see that as a strength.

**Viva question:** *"How do you know nobody peeked at the final test?"*
**Answer:** "The data is in a separate folder, only one function can open it, it needs an official signed and pushed marker, it opens only once, every opening is logged, and tests fail if any other code reads it."

---

## 5. The one-time re-make: fixing the practice test, once

**In one sentence:** we re-made two split files (PAMAP2 and WESAD) **once**, before any model existed, because their practice tests kept reusing the same people.

**Explain it like I'm new.** Imagine you practise for 5 exams, but every practice test is set by the same two friends. You'd get very good at *their* style, not at the subject. In WESAD, **10 practice places were filled by only 5 people**: S4 and S14 were used three times each. An independent checker spotted it.

**What we did**
- **Exam groups:** stayed **exactly the same**. The code refuses to re-make if any exam group would change.
- **Practice tests:** now rotate. Round *k* practises on people from round *k+1*'s exam group, so nobody practises twice. PAMAP2 now uses 4 different practice people, and WESAD 10.
- **Old files:** kept for anyone to see (`data/splits/superseded/`).
- **Decision:** written down first (D-020). The code refuses to run unless the decision is logged.
- **Sealed:** afterwards the folder was locked with fingerprints (`data/splits/SHA256SUMS`). A second re-make was tried and refused: "data/splits is sealed".

**Why this is not cheating:** the rule "never re-make splits" exists so nobody reshuffles groups until the *results* look good. No model had been trained, so no results existed, and the exam groups didn't change.

**Viva question:** *"Your rules say splits are made once. Why were two re-made?"*
**Answer:** "A reviewer found the practice people repeated. Before any model existed, we re-made only those two files, keeping the exam groups identical, kept the old files, logged the decision, then sealed the folder."

---

## 6. Sensor windows: cutting long recordings into short clips

**In one sentence:** long sensor recordings are cut into **short clips of about 5 seconds**, each with one activity, so a model can learn "what does walking look like?".

**Explain it like I'm new.** A sensor records non-stop, like a camera filming a whole day. To teach someone what "running" looks like, you'd cut out short clips where the person is clearly running. That's what we do with numbers.

**The recipe**
- **Same speed for everything:** every recording is changed to **25 readings per second**.
- **One clip** = **128 readings** ≈ **5 seconds**. A new clip starts every 64 readings, so clips overlap by half. That gives more clips without inventing data.
- **Clean clips only:**
  - the whole clip, plus a small margin on each side, must show **one activity**;
  - no broken stretches: short gaps (up to half a second for motion sensors, 2 seconds for heart rate) are filled smoothly, and longer gaps make the clip unusable.
- **Heart rate in PAMAP2** is only measured about 9 times per second, so the values in between are **filled in**. We say this openly.

**What each clip contains**
- **PAMAP2:** heart rate plus movement sensors on the hand, chest and ankle (3 directions each) = **10 signals**. Result: **10,361 clips**, 18 activities.
- **WESAD (wrist device):** blood-volume pulse, skin sweat (EDA), skin temperature and movement (3 directions) = **6 signals**. Result: **17,401 clips**: baseline 6,846, stress 3,861, amusement 2,144, meditation 4,550.
- **Missing values:** 0 in both.

**What they are for:** these are *different people* from the runners and have **no injury records**. They feed the IoT part (watch → phone → server) and the data generators, never injury claims.

*Numbers from:* `data/processed/pamap2/windows_meta.json`, `data/processed/wesad/windows_meta.json`.

**Viva question:** *"Can your sensor data predict injuries?"*
**Answer:** "No, and we don't claim it. Those people aren't the runners and have no injury records. The sensor data shows the IoT pipeline and trains the generators; injury results come only from the Lövdal runners."

---

## 7. SoccerMon and PMData: the 21-day rule for pain

**In one sentence:** when someone reports the same pain on many days, we count it as **one episode**, not many injuries.

**Explain it like I'm new.** If you hurt your knee on Monday and tell your coach "my knee hurts" every day that week, you didn't get 7 injuries. You got one, and you talked about it 7 times. So we group repeated reports together.

**The rule (copied from the Lövdal authors, D-018)**
- **Same episode:** a report **within 21 days** of the person's previous report.
- **New episode:** a report **22 or more days** later.
- **Injury event:** the **first day** of an episode.
- **Healthy day:** a day with **no report within 21 days before or after**. Days too close to the start or end of someone's data can't be judged, so they are left out.
- **No report on a day counts as "no pain".** That's an assumption, and we state it.

**What happened when we applied it**
- **SoccerMon:** 162 pain reports, on 156 different days, become **27 episodes** in **14 of 50 players**.
- **PMData:** 77 reports, on 72 days, become **8 episodes** in **6 of 15 people**.
  - Person **p12** reported pain on 44 of those 72 days: one long problem, so 0 new episodes.
  - Person **p08** is left out of injury work, because their injury file is missing (missing is not the same as "no injuries").

**Their role:** side experiments only (exploratory), never the main evidence. The main evidence is the Lövdal runners.

*Numbers from:* `data/processed/soccermon/daily_meta.json`, `data/processed/pmdata/daily_meta.json`.

**Viva question:** *"Why 21 days?"*
**Answer:** "People report the same pain on many days. Counting every report would turn one problem into many. We used the Lövdal authors' rule: within 3 weeks it is the same problem."

---

## 8. Data checks: fingerprints, MANIFEST and the fresh-copy test

**In one sentence:** every data file has a **fingerprint**, and we proved that anyone who rebuilds the project gets **exactly the same results**.

**Explain it like I'm new**
- **A fingerprint (sha256)** is a long code calculated from a file's contents. Change even one tiny letter in the file, and the code changes completely. It works like a seal on a jar: if the seal is broken, you know someone opened it.
- **The MANIFEST** (`data/MANIFEST.csv`) is a list of the fingerprints of all **729** raw files we use.
- **Before using a file**, every loader recomputes its fingerprint and compares it with the MANIFEST. If it's different, or the file was never checked, the program **stops**.
- **The split files** are sealed the same way (`data/splits/SHA256SUMS`), and the tests compare them every time.

**The fresh-copy test (22 Oct).** We pretended to be a new teammate:
- downloaded a clean copy of the code;
- copied only the 729 listed raw files;
- rebuilt *everything* from scratch: the audit, every data file, the data cards and all 7 split files;
- ran all the tests.

Result: **everything was identical**, and the tests passed in the copy too. The evidence is in `experiments/repro/2026-10-22_0731/`.

**Why it matters:** anyone with the same raw files gets the same numbers. In science, that's called **reproducible**, and it's one of the main things examiners check.

**Viva question:** *"How do you know your results can be reproduced?"*
**Answer:** "We tested it: a fresh copy of the code plus the listed raw files rebuilt everything, and every output and split file was identical."

---

## 9. Today's leak finding: why a runner's history gives the answer away

**In one sentence:** in this dataset, just **counting** a runner's recent events almost perfectly finds the injuries. That's a trick of how the data was built, so we ban it.

**Explain it like I'm new.** Imagine a quiz where every true answer happens to be printed in **red ink**. You could score almost 100 % by looking at the ink colour without knowing anything. A model would happily "learn the ink colour" and look like a genius. We must make sure it can't.

**Where the "red ink" comes from.** When the authors built the data, they **removed healthy days close to an injury**. So:
- an **injury** event always stands alone: there are **at least 22 days** with no other event before it;
- a **healthy** event usually has another event **the day before** (99.0 % of them).

So "how many events did this runner have in the last 3 weeks?" almost gives the answer away. We measured it on the development runners:
- **ROC-AUC 0.9946**, where 0.5 means guessing and 1.0 means perfect.
- Nothing about the body is involved; it's just how the rows were picked.

**What we decided (D-023, your choice)**
- **One scaler only:** numbers are rescaled with **one scaler learned from training runners only**. No per-runner rescaling based on a runner's own past.
- **Never inputs:** dates, event counts, gaps between events, how long a runner's history is, and the runner's ID number.
- **A "leak canary" test** in piece 3 proves none of these sneak into a model.
- **The paper's method:** it rescaled each runner using their own healthy days, which reads the answers. We use their exact method **only once**, as a labelled check on stand-in runners 54-63, never in a real comparison.

**Credit and honesty:** I had first suggested per-runner scaling. The independent checker caught the problem, we confirmed it with code, and nothing had been trained, so no harm was done. Catching a leak *before* training is exactly what our checks are for.

*Numbers from:* `data/processed/lovdal/event_design_check.json` (development runners only).

**Viva question:** *"Why don't you scale each runner against their own history, like the paper?"*
**Answer:** "In this dataset a runner's history gives the answer away: counting recent events finds injuries with ROC-AUC 0.9946, because healthy days near injuries were removed. So we use one scaler from training runners, and a test checks that no history signal is used."

---

## Where the whole project stands (22 Oct 2026)

| # | Piece | Status |
|---|---|---|
| 0 | Setup (laptop, Python, GPU, tests) | ✅ Done |
| 1 | Download + audit all datasets | ✅ Done |
| 2 | Loaders + person-level splits | ✅ Built and checked twice by an independent reviewer; **waiting for you to say "clear"** |
| 3 | First real models (baselines) | ⏳ Next. Rules ready; needs your OK and its own plan |
| 4 | Evaluation protocol | 🟡 Mostly done early: scoring and tuning rules fixed (D-021 to D-023). Targets for the generators and explanations are still blank, to be set before pieces 5 and 8 |
| 5 | Data generators (GANs, TimeVAE) | Not started |
| 6 | Edge-fog-cloud tiers (watch, phone, server) | Not started |
| 7 | Combined (fused) model | Not started (its rules are already fixed) |
| 8 | Explanations (SHAP and more) | Not started |
| 9 | Dashboard + demo | Not started |
| 10 | Paper + Review 2 slides | Not started (lots of usable text already in the docs) |

**How much is done?** About **3 of 11 pieces**, with a 4th mostly done: roughly **30 % by piece count**. By *amount of work left*, it's closer to **a quarter**, because the heaviest building (training models, generators, tiers, dashboard, paper) is still ahead. This is an estimate, not a measurement.

The part that's done is the foundation: clean data, fair splits, a sealed final exam, and written rules against cheating. That makes every later result trustworthy, and it will save time later, because the rules are already decided.

**What happens next**
1. You read this file and ask about anything unclear.
2. You say **"piece 2 is clear"**.
3. I bring the **piece 3 plan**: the first real models, plus the six small details still open. You approve it, then models are trained.
