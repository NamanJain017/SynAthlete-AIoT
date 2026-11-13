# CHAPTER 1 | INTRODUCTION

This chapter explains why athlete injury risk is an important problem, what earlier systems leave unsolved, what this project set out to achieve, and how the rest of the report is organised.

## 1.1 Background

Athletes improve by stressing their bodies and then recovering. When the stress grows faster than the body can adapt, or when recovery is cut short for several days in a row, the risk of an overuse injury rises. Such injuries cost training time, competitions and sometimes careers. Coaches try to prevent them by watching training diaries, distances, intensities and how tired their athletes feel, but a coach responsible for many athletes cannot read every number every day.

Two technologies promise help. Wearable devices such as chest straps, wrist sensors and inertial measurement units record heart rate and movement without effort from the athlete [@caviedes2020,@abdalalim2024]. Machine learning can turn these signals, and the daily training records that clubs already keep, into estimates of activity and risk [@wang2019,@an2025]. When the computation is split between the device, a nearby gateway and a remote server, the system is called an Artificial Intelligence of Things (AIoT) system. Edge and fog processing keep most raw data close to the athlete, which saves network bandwidth and reacts faster than sending everything to the cloud [@cao2021,@aazam2018].

## 1.2 Problem Statement

Published AIoT frameworks for athlete health usually describe the hardware and the data flow, but they rarely build and test the part that matters most to a coach: a prediction of injury risk that is accurate enough to act on and that explains itself. Three problems stand in the way.

- **No suitable public data.** No public dataset combines high-frequency body signals with real injury records from the same athletes. Injury records exist only as daily training logs [@lovdal2021,@lovdal2021data], while sensor datasets contain no injuries [@reiss2012,@schmidt2018].
- **Optimistic evaluation.** Injuries are rare, groups of athletes are small, and many studies let information from the test set leak into training. Leakage of this kind has produced inflated results in many fields [@kapoor2023], and injury prediction is particularly exposed because a previous injury strongly predicts the next one [@stern2021].
- **Untested explanations.** Coaches need to know why a risk score is high. Explanation methods exist, but their output is often shown without checking whether it reflects what the model really uses [@samek2017,@arrieta2020].

The problem addressed in this project is therefore: **to build a complete, explainable AIoT pipeline for athlete injury risk, and to measure honestly how well each part works, using only real injury labels as injury evidence.**

## 1.3 Objectives

The five objectives were fixed at the first review and every one of them was implemented.

- **Objective 1, bridging the data gap:** build generators that create synthetic wearable data (TimeGAN, WGAN-GP and TimeVAE, compared with classic augmentation) and test whether the data are realistic, useful and private.
- **Objective 2, edge-fog-cloud tiers:** implement the three processing tiers as separate programs and measure the data they send and the time they need to decide.
- **Objective 3, fused risk model:** combine a sequence model of day-by-day training with a tree model of weekly totals into one calibrated 0-100 risk score.
- **Objective 4, validated explanations:** explain every risk score and test the explanations for faithfulness.
- **Objective 5, coach dashboard:** present risk scores, reasons, live sensor views and results in a dashboard that a coach can use.

A sixth, cross-cutting objective was honest testing: baselines, a leak alarm, a rebuild of a published method and a sealed final test.

## 1.4 Scope and Contributions

The project treats the two kinds of data as two separate tracks. Track A uses real training logs and real injuries and is the only source of injury claims. Track B uses wearable streams to build and measure the IoT tiers and the generators. The main contributions are:

- an open, end-to-end pipeline in which every dataset and every component has one clearly stated role;
- an evaluation protocol written before any experiment, with person-level splits, a leak alarm, runner-level uncertainty, paired comparisons and a sealed final test opened only once;
- a rebuild of a published injury-prediction method together with a variant that does not read the labels;
- a comparison of three deep generators and classic augmentation against criteria fixed in advance;
- explanations that are tested by deletion, randomisation and stability checks and compared with expectations written before running.

## 1.5 Software Requirements Specification

The system was developed and run on a single Windows laptop and is designed so that every step except deep-network training also runs on a computer without a graphics card. Table 1.1 lists the requirements.

[[TABLE:srs|Software and hardware requirements]]

## 1.6 Organisation of the Report

Chapter 2 surveys related work and identifies the research gap. Chapter 3 presents the system architecture and the design of each module. Chapter 4 describes the methodology: data preparation, the evaluation protocol and every model. Chapter 5 explains how the code is organised and tested. Chapter 6 reports and discusses the results, and Chapter 7 concludes the report and outlines future enhancements.

# CHAPTER 2 | LITERATURE SURVEY

This chapter reviews five areas of earlier work that the project builds on: injury prediction and training load, wearable sensing, edge and fog computing, synthetic data, and the evaluation and explanation of machine-learning models. It ends with the research gap that the project addresses.

## 2.1 Injury Prediction and Training Load

Several studies have tried to predict sports injuries with machine learning. In elite youth football, a tree-based approach improved the understanding of which screening measures relate to injury, although its ability to predict individual injuries remained limited [@oliver2020]. A prospective study of elite youth ski racers linked characteristics of training load to the risk of injury and illness [@hildebrandt2021]. Load is often summarised by the ratio of recent (acute) to longer-term (chronic) load, but alternative measures have been proposed because the ratio has statistical weaknesses [@tysoe2020].

Two further insights shaped this project. Injury risk behaves like a non-linear system in which small changes in load can have large and delayed effects [@stern2020], and a previous injury is one of the strongest predictors of a future one [@stern2021]. The second insight is a warning for evaluation: a model that is allowed to see an athlete's own injury history can appear accurate without learning anything useful about training load. Lövdal and colleagues released daily training logs of competitive runners with real injuries and predicted injuries with bagged gradient-boosted trees [@lovdal2021,@lovdal2021data]; their data are the injury evidence used in this project.

## 2.2 Wearable Sensing and Activity Recognition

Wearable sensor systems can monitor posture and movement during exercise and give biofeedback to the user [@caviedes2020]. Machine learning applied to wearable signals can recognise physiological states such as stress outside the laboratory [@abdalalim2024]. Deep learning is now the standard tool for recognising activities from body-worn sensors [@wang2019], and convolutional networks can learn useful features directly from raw accelerometer and gyroscope data [@ronao2016]. The PAMAP2 dataset provides heart rate and three inertial sensors for a range of daily and sports activities [@reiss2012], and the WESAD dataset provides chest and wrist signals from a different group of people [@schmidt2018].

## 2.3 Edge, Fog and Cloud Computing

Edge computing places computation on or near the device that produces the data, and edge-cloud architectures divide the work between the device and a remote server to reduce latency and network traffic [@cao2021]. Fog computing adds an intermediate layer, such as a gateway, between the devices and the cloud and has been studied widely for the industrial Internet of Things [@aazam2018]. These ideas map naturally onto athlete monitoring: a watch can check each few seconds of data, a phone can summarise a training session, and a server can run heavier models once a day.

## 2.4 Synthetic Data Generation

Generative adversarial networks learn to produce new samples that look like a training set by training a generator against a discriminator [@gui2023]. For multivariate time series, adversarial models have been used to fill in missing values [@guo2019], and recent augmentation methods mix or deform real series to create extra training examples [@liu2024]. A synthetic dataset is useful only if it is realistic, helps a downstream task and does not copy real records; earlier wearable-health systems rarely test all three.

## 2.5 Evaluation, Calibration and Explanation

Data leakage, model selection on the test set and similar shortcuts are a major source of irreproducible machine-learning results [@kapoor2023]. The way data are divided into training and test parts changes performance estimates [@wong2015]. The ROC curve summarises ranking quality [@fawcett2006], but with rare positive events it should be complemented by measures that focus on the positives [@he2009]. Risk estimates shown to people should also be calibrated, so that a predicted 10 % risk means about 10 % [@vancalster2016]. Tree ensembles remain strong on tabular data [@shwartzziv2022], while gated recurrent networks model sequences [@greff2017]. Explainable artificial intelligence is needed for trust and accountability [@arrieta2020], and explanation methods must themselves be evaluated, for example by checking that removing the inputs they mark as important changes the prediction more than removing random inputs [@samek2017,@montavon2018].

## 2.6 Summary of the Survey and Research Gap

Table 2.1 summarises the most relevant works and what each leaves open.

[[TABLE:survey|Summary of related work]]

The survey shows a clear gap. Injury studies rarely combine prediction with IoT processing and tested explanations; IoT studies rarely use real injury outcomes; and few studies of either kind protect their evaluation against leakage with a sealed test. This project fills the gap by building all the parts and evaluating each of them honestly.

# CHAPTER 3 | SYSTEM ARCHITECTURE AND DESIGN

This chapter describes the overall architecture, the role of each dataset, the design of each software module and the measures that protect the data used for the final test.

## 3.1 Overall Architecture

Figure 3.1 shows the architecture. It has two tracks that share a common evaluation protocol.

[[FIG:0_architecture.png|System architecture of SynAthlete-AIoT]]

**Track B** carries a wearable stream through three tiers. The edge tier receives each 5-second window of heart rate and motion, applies a reflex rule (very high heart rate or a violent acceleration triggers an immediate alert) and recognises the activity with a small convolutional network; it then sends a summary of about a hundred bytes instead of the raw window. The fog tier collects these summaries into sessions. The cloud tier stores the sessions and runs the daily injury model. A cloud-only design, in which every raw window travels to the cloud, is implemented for comparison.

**Track A** turns real training logs into an injury-risk score. The logs are split by person into development rounds and a sealed final test. A sequence model reads the last seven days and a tree model reads weekly totals; a stacking model combines them, Platt scaling calibrates the result, and the calibrated risk is expressed as a 0-100 score with an alert level chosen on validation data.

Around the two tracks sit the generators, which may add synthetic data to training rows only, the explanation module, which explains each score and tests the explanations, and the coach dashboard.

## 3.2 Datasets and Their Roles

Every dataset was audited before use, and each received exactly one role so that no dataset is used as evidence for something it cannot show. Table 3.1 lists them.

[[TABLE:datasets|Datasets and their roles]]

The runner logs come in two files. The day file describes each event by the seven days before it, with ten measures per day such as total distance, distance in different intensity zones, perceived exertion and recovery. The week file describes the three weeks before the event with twenty-two measures per week plus three relative changes. Only days present in both files are used by the fused model. The wearable participants are different people from the runners, so results from Track B are evidence about the system, not about injuries.

## 3.3 Design of Modules

The code is organised into seven modules, each in its own folder under src/. Table 3.2 lists them.

[[TABLE:modules|Software modules]]

### 3.3.1 Data module

The data module reads every raw file, records a fingerprint of each file so that later steps can prove they used the same data, builds the processed arrays and creates the person-level splits once. A guard prevents any experiment from reading the sealed test runners through the normal loaders.

### 3.3.2 Model module

The model module contains the baseline models, the neural networks, the seeded tuning procedure, the fused model and the final-test program. Every fitting function receives one round's training rows, and every choice of setting uses that round's validation rows only.

### 3.3.3 Generator module

The generator module implements the three deep generators and classic augmentation, trains them on one round's training people, and evaluates realism, usefulness and privacy against the real test people of the same round.

### 3.3.4 Pipeline module

The pipeline module implements the edge, fog and cloud tiers as separate operating-system processes connected by queues, so that the measured message sizes and delays come from real inter-process communication.

### 3.3.5 Explanation module

The explanation module computes tree-based SHAP values, integrated gradients, attention weights and Grad-CAM maps, runs the faithfulness tests and writes plain-language reasons.

### 3.3.6 Dashboard module

The dashboard is a Streamlit application with eight pages: home, athletes, athlete detail, live sensor, synthetic data, results, a training lab and a data-and-honesty page. Its data are prepared in advance so that a live demonstration starts instantly.

### 3.3.7 Paper module

The paper module writes every number, table and graph of the paper, the slides and this report from the saved result files, so that no result is ever typed by hand.

## 3.4 Protection of the Final Test

Ten runners were set aside at the start of the project and stored separately. They can be loaded only by one function, which refuses unless the code has been frozen with an annotated version-control tag that has been published, the working copy is unchanged, and no earlier opening happened at a different version. Every opening is written to an access log. The final test was opened once, after all development decisions were recorded. Table 3.3 gives the procedure step by step.

[[ALGO:Opening the sealed final test|Write the final-test program and record its protocol in the decision log, before any sealed data is touched.;; Run a rehearsal of the whole program on development runners only and confirm that its numbers can be recomputed from the saved scores.;; Commit all code and create an annotated eval-freeze tag on that commit; publish the tag.;; Start the final run: it first checks every condition of the guard (tag, unchanged code, single frozen version, data fingerprint) and stops at once if one fails.;; Train the declared final model on development runners only.;; Only now load the sealed runners through the guard, which writes the opening to the access log.;; Score the sealed runners once, save every raw score, then compute the summary numbers from the saved files.;; Commit the results together with the access log; never reopen at another version.]]

## 3.5 Data Flow Through the System

A single day of use illustrates how the modules cooperate. During a training session the wearable produces a 5-second window; the edge tier checks it immediately and raises an alert if the heart rate or acceleration is dangerous, otherwise it forwards a short summary. When the session ends, the fog tier turns the summaries into one session record. Once a day, the cloud tier receives the athlete's training log for that day, builds the seven-day sequence and the weekly totals, and passes them to the fused model, which returns a calibrated probability, a 0-100 risk score, an alert decision and a reason sentence. The dashboard reads these outputs and shows them to the coach. During development, the same functions are run offline on the stored data, and the evaluation protocol decides which data each function may see.

# CHAPTER 4 | METHODOLOGY

This chapter describes how the data were prepared, how every result was evaluated, and how each model, generator and tier works.

## 4.1 Data Preparation

### 4.1.1 Runner logs

Each injury event in the runner logs is described by the training before it. Healthy events are the days on which no injury followed. The day and week files are aligned so that the same events can be compared across models. Inputs that identify a runner, give a date or describe the runner's own past events are excluded, because in this dataset a runner's own event history reveals whether an injury is coming.

### 4.1.2 Wearable windows

The PAMAP2 and WESAD recordings are resampled to 25 Hz and cut into windows of 128 samples (5.12 seconds). Windows near a change of activity are dropped so that each window carries a single label. Heart rate from Sport DB 2.0 is checked for plausible values and used as a realism reference.

## 4.2 Evaluation Protocol

The protocol was written down before the first experiment, and every later change was recorded with its reason.

### 4.2.1 Person-level splits

All splits are by person, never by row, so a model is always tested on people it has never seen. The 64 development runners form five rounds; each round has its own training, validation and test runners. The headline result is the average over the five rounds. The ten newest runners form the sealed final test.

### 4.2.2 Leakage controls

One standardisation scaler is fitted on the training runners of each round and applied unchanged to validation and test rows. Settings, early stopping, calibration and the alert level are chosen on validation runners. A leak alarm trains a model on shuffled labels; if its test score were clearly above chance, information would be leaking. Its ROC-AUC was {canaryAuc}, which is at chance.

### 4.2.3 Metrics

ROC-AUC measures how often an injured event receives a higher score than a healthy one:

[[EQ:AUC = (1 / (|P| |N|)) Σ_{i∈P} Σ_{j∈N} 1[s_i > s_j]]]

where P is the set of injured events, N the set of healthy events and s the model score. PR-AUC (average precision) summarises precision across all recall levels; its chance level equals the injury rate, which makes it the headline metric for rare injuries [@he2009]. Calibrated probabilities are obtained by Platt scaling fitted on validation runners:

[[EQ:p = 1 / (1 + exp(-(a · logit(s) + b)))]]

The risk score shown to a coach is the percentile of the calibrated probability among the validation events of the same round. The alert level is the threshold that makes sensitivity and specificity as equal as possible on validation runners; Table 4.1 describes how it is chosen.

[[ALGO:Choosing the alert level on validation runners|Collect the model scores of all validation events of the round and their true labels.;; Use every distinct validation score, plus infinity ("alert nobody"), as a candidate threshold.;; For each candidate, compute sensitivity and specificity when an alert is raised for scores greater than or equal to the candidate.;; Keep the candidate with the smallest absolute difference between sensitivity and specificity.;; Break ties by the larger sum of sensitivity and specificity, then by the higher threshold.;; Apply the chosen threshold unchanged to the test runners; it is never adjusted on test data.]]

### 4.2.4 Uncertainty and comparisons

Uncertainty is estimated with a runner-level bootstrap: whole runners are resampled 2,000 times within each test group and the metric is recomputed each time. Two models are compared on the same rows by resampling the difference of their scores; a difference is called clear only when its 95 % interval excludes zero. These intervals are not adjusted for the number of comparisons. Table 4.2 shows the procedure.

[[ALGO:Runner-level bootstrap interval|Group the test rows of each round by runner.;; Draw, within every round, as many runners as the round has, with replacement.;; Collect all rows of the drawn runners and recompute the metric (or the difference between two models on the same rows), averaging over the three seeds.;; Repeat 2,000 times with a fixed random seed; discard a draw only if it contains no injury.;; Report the 2.5th and 97.5th percentiles as the 95 % interval.]]

## 4.3 Baseline Models

Seven injury models form the baseline set: a constant risk equal to the training injury rate, an acute-to-chronic style rule, logistic regression, random forest, CatBoost, a gated recurrent unit (GRU) network and the published Lövdal method. The Lövdal method was rebuilt in two forms. The exact copy includes its normalisation step, which uses each athlete's healthy events and therefore reads the labels; it was run once as a sanity check on stand-in runners. The label-free variant replaces that step with ordinary training-row scaling so that it could be used in practice, and it is the version compared on all five rounds. Each tunable model tries eight settings per round from grids written before the first run, and the chosen setting is refitted with three random seeds.

## 4.4 Fused Model

The fused model has two arms. The sequence arm reads the seven-day sequences with a GRU, a long short-term memory network or a temporal convolutional network; the tabular arm reads the weekly totals with a random forest or CatBoost. The best candidate of each arm is chosen on validation runners. A logistic-regression stacker combines the two arms. It is trained on out-of-fold predictions, which are produced by refitting each arm five times on the training runners with one group of runners held out each time, so the stacker never sees a prediction made by a model that trained on the same runner. Platt scaling and the alert level are then fitted on validation runners. In total the fused model tries 44 configurations per round, the same budget per component as the baselines. It was declared in advance as the model for the sealed test. Table 4.3 summarises one round.

[[ALGO:Training the fused model in one round|Scale the day and week inputs with statistics of the training runners only.;; Try eight settings of each sequence candidate (GRU, LSTM, TCN) and each tabular candidate (random forest, CatBoost); keep the best of each arm by validation PR-AUC.;; For each refit seed, produce out-of-fold arm predictions on the training runners: split them into five groups by runner, train each arm on four groups and predict the fifth.;; Train the logistic-regression combiner on the out-of-fold predictions; choose its regularisation strength C on validation PR-AUC.;; Refit both arms on all training runners and score the validation and test rows.;; Fit Platt scaling and the alert level on the validation rows; compute the 0-100 risk score as the percentile among validation events.;; Score the test rows once and save every raw score.]]

## 4.5 Synthetic Data Generators

TimeGAN combines an autoencoder with adversarial and supervised losses so that generated sequences follow realistic step-to-step dynamics. WGAN-GP uses a Wasserstein critic with a gradient penalty and one-dimensional convolutions; because of time limits it uses one declared setting and is reported as untuned. TimeVAE is a variational autoencoder that represents each window by a small latent code. Classic augmentation creates new windows by adding noise, scaling the signal and warping its time axis. All generators are conditioned on the activity label and trained only on each round's training people.

Four criteria were fixed in advance. A classifier trained to tell real from synthetic windows should be close to 50 % accurate (within 0.15). A model trained only on synthetic data and tested on real people (train-on-synthetic, test-on-real, TSTR) should come within 0.15 macro-F1 of a model trained on real data. The autocorrelation and the correlation between channels should differ from real data by less than 0.10. At most 5 % of synthetic windows may be closer to a real training window than real test windows typically are, which guards privacy. For injury data, a conditional TimeGAN adds synthetic injury sequences to the training rows only; validation and test data always stay real.

## 4.6 Edge, Fog and Cloud Tiers

The three tiers run as three processes that exchange messages through queues; the cloud-only design uses two processes. Every process finishes its start-up before the first window is released, and windows are then released at a steady pace of one every 0.1 seconds. The benchmark records the bytes that reach the cloud and the median time from the moment a window is made to the moment a decision is taken, for one PAMAP2 test person and one Sport DB 2.0 heart-rate recording, with three repetitions per design.

## 4.7 Explanations

TreeSHAP attributes each tabular prediction to the weekly measures, and integrated gradients attribute each sequence prediction to the daily measures. An attention model trained only for explanation is always displayed next to integrated gradients, never on its own, and Grad-CAM highlights the time steps that drive the activity network. Three tests check faithfulness with thresholds fixed in advance: removing the most important inputs must lower the risk at least 1.5 times more than removing random inputs; the explanations of a model with random weights must differ (rank correlation below 0.5); and small changes to the input must leave the explanation almost unchanged (rank correlation above 0.8). Before running, a list of expected risk factors was written down: recent load spikes, monotony and low recovery. Each risk score is turned into a plain-language sentence; a real example from the first round reads: "{realReason}"

## 4.8 Hyperparameters

Table 4.4 summarises the fixed values and search grids. The values chosen for the final model are recorded in the run files.

[[TABLE:hyper|Main hyperparameters and search grids]]

# CHAPTER 5 | CODING AND TESTING

This chapter explains how the code is organised, how it is run, and how it was tested so that its results can be trusted.

## 5.1 Code Organisation and Tools

The project is written in Python with type hints. PyTorch is used for the neural networks, scikit-learn, CatBoost and XGBoost for the tree and linear models, SHAP and Captum for explanations, Streamlit for the dashboard, and Matplotlib for graphs. One command runner, tasks.py, starts every stage. Table 5.1 lists its main commands.

[[TABLE:commands|Main project commands]]

Each experiment writes its own folder that contains the configuration, a record of the code version, the random seeds and the data fingerprints, and a metrics file. The paper, the slides, the dashboard and this report read their numbers from these metrics files.

## 5.2 Testing Strategy

Automated tests cover every module and run with a single command. They use small, made-up data so that they run in seconds and never touch private data. Table 5.2 lists the test groups; at the end of the project all 289 tests passed.

[[TABLE:tests|Automated test groups]]

## 5.3 Key Code Listings

The following listings are copied automatically from the source files when this report is built, so they always match the code that produced the results.

The first listing is the guard of the sealed final test. It checks every condition before anything is loaded and stops with a clear message if the code is not frozen properly.

[[CODE:src.data.lockbox.check_can_open|guard of the sealed final test]]

The second listing produces the out-of-fold predictions used to train the combiner of the fused model. Each training row is scored by a model that never saw that row's runner.

[[CODE:src.models.fused.oof_predictions|out-of-fold predictions grouped by runner]]

The third listing turns a calibrated probability into the 0-100 risk score shown to the coach.

[[CODE:src.models.fused.risk_score|0-100 risk score]]

The fourth listing chooses the alert level from validation scores only, following Table 4.1.

[[CODE:src.eval.metrics.choose_alert_threshold|choice of the alert level]]

The fifth listing is the deletion test for explanations: removing the inputs an explanation marks as most important must lower the risk more than removing random inputs.

[[CODE:src.xai.checks.deletion|deletion test for explanations]]

The sixth listing releases wearable windows at a steady pace in the tiers benchmark, so that latency measures decisions rather than start-up waiting.

[[CODE:src.pipeline.tiers._Stream|steady-pace window stream]]

## 5.4 Leakage and Integrity Checks

Several checks protect the results beyond ordinary unit tests:

- **Leak alarm.** A shuffled-label model must score at chance before any real model is trained.
- **Declared inputs.** A guard refuses any model input that is not on the declared list, so identity or date columns cannot slip in.
- **Sealed test guard.** The final-test runners can only be loaded at a frozen, published code version, and every opening is logged.
- **Protected files.** Repository hooks block hand edits to raw data, splits and metrics files.
- **Reproducibility.** A fresh copy of the repository rebuilt every processed file and every split identically, and a rerun of the GRU baseline reproduced its earlier results exactly.

## 5.5 Reviews and Continuous Integration

Every push to the online repository runs the linter and the test suite automatically. Independent reviews of the evaluation code were carried out before the code freeze and before results were reported; each review produced a list of findings that were fixed and recorded in the decision log.

## 5.6 Issues Found and Fixed

Honest testing found real problems, which were fixed before they could affect a reported result:

- the first generator runs were too slow and later ran out of memory; they were restarted with memory release, checkpoints and automatic resumption;
- a diverged generator setting produced invalid values; such settings now receive the worst validation score instead of stopping the run;
- the first draft of the final-test program computed numbers before saving the raw scores; it was changed to save every raw score first and to load the sealed runners only after training;
- the first timing trial of the tiers measured start-up waiting instead of decision time; the benchmark was changed so that all tiers start first and windows arrive at a steady pace;
- the online test service rewrote the freeze tag on its own machine, which made one automated check fail; the cause was reproduced and the workflow fixed.

# CHAPTER 6 | RESULTS AND DISCUSSIONS

This chapter reports every result, explains what it means and discusses the limits of the project.

## 6.1 Injury Prediction

Table 6.1 and Figure 6.1 report every injury model on runners never seen in training, averaged over the five development rounds. Learned models reach ROC-AUC {injuryAucRange}: above chance, but far from reliable individual prediction. The fused model reaches ROC-AUC {fusedAuc} and PR-AUC {fusedPrAuc}. The exact copy of the published method passed its sanity check with day-file ROC-AUC {reproDayAuc}, and the label-free variant reaches ROC-AUC {labelFreeAuc}; because the two were scored on different runners, the effect of the label-reading step was not measured directly.

[[TABLE:injury|Injury prediction, mean ± standard deviation over five development rounds]]

[[FIG:1_injury_models.png|ROC-AUC and PR-AUC of every injury model]]

Figure 6.2 shows the full ROC and precision-recall curves. All curves stay close to the diagonal and close to the chance line of precision, which confirms that the signal in daily training logs is weak. Figure 6.3 shows that model scores also vary noticeably from round to round, which is why the protocol reports means with their spread rather than a single split.

[[FIG:8_roc_pr_curves.png|ROC and precision-recall curves of the injury models]]

[[FIG:11_per_round_scores.png|Test ROC-AUC of every model in every development round]]

## 6.2 Sealed Final Test

{lockboxSummary} Figure 6.4 places this result next to the development estimate. The two agree, which shows that the development results were not inflated by tuning. The secondary results were scored on different rows and with more training runners, so they are not used to rank models, and no model was changed after the test.

[[TABLE:lockbox|Sealed final test (ten runners, opened once)]]

[[FIG:2_locked_final_test.png|Sealed final test compared with the development estimate]]

Figure 6.5 shows what the alert means in practice. At the alert level chosen on validation runners, the model catches a majority of injuries but also raises many false alarms, because injured and healthy days receive overlapping scores. Figure 6.6 shows that the calibrated probabilities roughly match the observed injury rates, and that the risk scores of injured days are higher on average but overlap heavily with those of healthy days. The alert is therefore a prompt for a conversation with the athlete rather than a diagnosis.

[[FIG:9_confusion_injury.png|Confusion matrices at the alert level chosen on validation runners]]

[[FIG:10_calibration_and_risk_score.png|Calibration curve and distribution of the 0-100 risk score]]

## 6.3 Model Comparisons

Table 6.3 and Figure 6.7 compare models on exactly the same rows. Against the label-free Lövdal variant, logistic regression and the GRU on the day file are clearly better on both metrics, and CatBoost on the day file on PR-AUC only, while constant risk and the acute-to-chronic style rule are clearly worse. Figure 6.8 shows that the fused model performs about as well as its own two arms: its gain over the sequence arm is borderline on ROC-AUC and absent on PR-AUC.

[[TABLE:comparisons|Paired differences with 95 % runner-bootstrap intervals]]

[[FIG:3_paired_comparisons.png|Paired differences between models]]

[[FIG:12_fused_vs_arms.png|Fused model compared with its two arms]]

## 6.4 Synthetic Data

Table 6.4 and Figure 6.9 summarise the generators on PAMAP2 against real-only training (macro-F1 {pamapRealFone}). A classifier separated real from synthetic windows easily for every generator, so none met the realism criterion. Classic augmentation came closest to real data when training on synthetic data alone, and adding synthetic windows to real ones changed little. {wesadSummary} {augmentSummary} Figure 6.10 shows the injury-augmentation result.

[[TABLE:generators|Generators on PAMAP2 (mean of four rounds)]]

[[FIG:4_generators.png|Realism and usefulness of the generators]]

[[FIG:7_injury_augmentation.png|Effect of synthetic injury examples on the GRU]]

## 6.5 Activity Recognition

The activity network recognises activities of people it has never seen with macro-F1 {pamapRealFone}. Figure 6.11 shows the confusion matrix for the first round. Clear movements such as walking, running, cycling and Nordic walking are recognised almost perfectly, while quiet activities such as sitting, standing and computer work are often confused, because they produce similar body signals.

[[FIG:13_confusion_activity.png|Confusion matrix of the activity network]]

## 6.6 Edge, Fog and Cloud Tiers

Table 6.5 and Figure 6.12 show the tier benchmark. Processing at the edge reduced the data sent to the cloud by {tiersReduction} for the wearable stream and by {tiersReductionHr} for the heart-rate stream. Decisions took a few milliseconds in both designs. Because everything ran on one computer there was no network delay to save, so the main result is the reduction in data, which would save bandwidth and battery in a real deployment.

[[TABLE:tiers|Edge-fog-cloud compared with cloud-only]]

[[FIG:5_edge_fog_cloud.png|Data sent to the cloud and decision time]]

## 6.7 Explanations

{explainSummary} Against the expectations written before running, {explainPriors} (Figure 6.13). The attention weights often disagreed with integrated gradients, which supports the rule of showing attention only next to a tested method.

[[FIG:6_explanations.png|Faithfulness tests and agreement with prior expectations]]

## 6.8 Coach Dashboard

The dashboard brings the system together for a coach. The athletes page lists runners with their current risk score; the athlete detail page shows the score over time with the plain-language reason; the live sensor page replays a wearable stream through the edge tier; the synthetic data page compares real and generated windows; the results page shows every evaluation result including the sealed test and the comparisons; the training lab lets a user retrain a fast model with different settings on development data only; and the honesty page lists the role and limits of every dataset. The coach pages replay development runners and never show the sealed runners.

## 6.9 Discussion and Limitations

The results support three conclusions. First, daily training logs contain only a weak warning signal before an injury, so every honest model sits modestly above chance and a balanced alert raises many false alarms. Second, the evaluation design matters: the sealed test reproduced the development estimate, and paired intervals showed that most apparent differences between models are not reliable. Third, the IoT design gives a clear benefit in data volume even when the injury signal is weak.

The project also has limits. The sealed test contains only ten runners, so its interval is wide and still understates the uncertainty. The dataset's authors removed healthy days close to injuries, so alert rates are not daily rates. The tiers were simulated on one computer, and the wearable participants are different people from the runners, so the IoT results demonstrate the system rather than injury prediction from sensors. Finally, synthetic data did not reach realism, so it was used for training only.

# CHAPTER 7 | CONCLUSION AND FUTURE ENHANCEMENT

This chapter reviews the work, states its conclusions and suggests how it can be extended.

## 7.1 Review of the Work

Chapter 1 described the need for explainable injury-risk prediction and set five objectives. Chapter 2 showed that earlier work leaves prediction, IoT processing, tested explanations and leakage-safe evaluation unconnected. Chapter 3 presented a two-track architecture with clear dataset roles and a protected final test. Chapter 4 described the evaluation protocol and every model, generator, tier and explanation method. Chapter 5 showed how the code is organised, tested and checked for leakage. Chapter 6 reported every result, including negative ones.

## 7.2 Conclusions

All five objectives were implemented and evaluated. The injury-risk model predicts injuries only modestly above chance, and this level held on runners that nobody looked at until the end, which shows that the result is honest. The edge tier sharply reduces the data sent to the cloud. The explanations passed every faithfulness test, while one expected risk factor, low recovery, was mostly not confirmed. The generators did not produce realistic data and did not improve injury prediction, which is reported as a negative result. The main lesson of the project is that a correct, carefully tested modest result is more useful than an impressive result that cannot be trusted.

## 7.3 Future Enhancements

- **Paired data:** collect wearable signals and real injury records from the same athletes, so that physiology can become injury evidence.
- **Real deployment:** run the tiers on real watches, phones and servers to measure energy use and real network delay.
- **Prospective validation:** test the model on new athletes and new seasons, with larger groups.
- **Better generators:** improve the realism of synthetic data with larger datasets and privacy-aware training.
- **Coach studies:** study with coaches how risk scores and reason sentences change training decisions.
- **Hardware prototype:** build a wearable prototype that runs the edge tier on the device itself.
