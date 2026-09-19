# Phase 6 Implementation Plan — Stage 16 (Experiment Registry & Submission Strategy) and Stage 17 (Champion Model Gate)

**Author:** Henry Otsyula. **Competition:** Climate & Health Risk Prediction Challenge (closes 19 Oct 2026, $500 first place). **Governing documents:** `docs/PROJECT_BLUEPRINT.md` (Phase 6), `docs/experiment_registry.md` (current state).

## 0. Objective, scope, and why this phase decides the competition

**Objective:** turn the evidence Phases 4–5 already produced (a tuned, ensembling-tested, calibrated champion — `catboost_tuned` trial #26 + Platt calibration, one real Zindi submission at 0.830979519) into a disciplined, auditable submission process that maximizes the probability of holding or improving 1st place through 19 Oct 2026, and survives a private-leaderboard reveal that could reorder every visible rank.

**Why this phase is the one that actually wins or loses the competition, not Phases 4–5:** the modeling work is functionally done and closed with real evidence (Stage 14.3: ensembling doesn't help; Stage 8.5: three feature experiments rejected). What is *not* done is the discipline that decides whether the good model you already have actually converts into 1st place: submission timing and count, protection against the single most common way strong competitors lose (overfitting the public leaderboard, which the blueprint explicitly flags as a live risk given rivals already at 40–80 submissions each), and a hard gate that stops a plausible-looking but leakage-compromised or unstable candidate from ever becoming the one submitted at close. A brilliant model with an undisciplined submission strategy loses to a good model with a disciplined one — that is the entire premise of Stage 16 existing as its own phase in the blueprint rather than being folded into Stage 15.

**In scope:** completing Stage 16 (registry formalization, submission-numbering retrofit, go/no-go checkpoint — partially done in the prior session, this plan closes the rest) and Stage 17 (the champion gate checklist, run for real against the actual current champion) in full, plus the ongoing submission-strategy discipline that governs every remaining Zindi upload between now and 19 Oct.

**Out of scope (explicitly, so this plan doesn't quietly expand):** any new feature engineering (Stage 8.5 already closed this line — three rejects), any new model/ensemble experiments (Stage 14.3 already closed this line — ensembling rejected with real evidence), Phase 7/8 production and portfolio engineering (deployment, interpretability, README, model card — these matter for the portfolio goal but have no hard deadline and do not affect the competition outcome). If Stage 17's gate finds a real problem, this plan explicitly re-opens Phase 5 rather than trying to patch it inside Stage 17 — see Task 8.

## 1. Current state (verified against the live repo, not assumed)

- Champion candidate: `catboost_tuned` (Optuna trial #26, `configs/model_best.yaml`) + Platt calibration (`src/climate_health/models/calibrate.py`, Stage 15.2).
- One real submission made: **C-001**, 2026-09-14, public LB **0.830979519**, generation commit `19bdb9a`. Logged in `docs/experiment_registry.md`'s Submission log and `submissions/leaderboard_log.csv`.
- Go/no-go checkpoint already run: public LB exceeds local Tier-2 mean (0.8156) by +0.0154 — the safe direction, not a red flag.
- Stage 8.5 (post-champion feature stretch): closed, F-005/F-006/F-007 all rejected on Tier-2 mean.
- Stage 14 (ensembling): closed, Stage 14.3 rejected the blend on full repeated CV — standalone `catboost_tuned` is the champion.
- **Not yet done:** `competition-submission-final` git tag; retrofitted S/M/E-series IDs on Stage 10–14's existing entries; a written submission-strategy policy for the remaining ~4 weeks; Stage 17's full gate checklist run against the actual current champion; a stale `.git/index.lock` was blocking commits as of the last session (assume resolved — Task 1 verifies this).

## 2. Architecture / process decisions (ADRs for this phase)

**Decision 1 — the champion gate runs on the exact artifact that would be submitted, not on a description of it.** Every Stage 17 check (leakage, adversarial-validation residuals, stability, reproducibility, schema) executes against the real serialized model + real `predict.py` output, never "this should be fine because Stage 13/14 already checked something similar." A gate that isn't run adversarially against the actual candidate is not a gate.

**Decision 2 — retrofitting IDs onto Stage 10–14's already-completed work is a documentation exercise, not new experimentation.** No CV is re-run to produce these IDs; existing numbers are relabeled for portfolio/audit consistency with the S/F/M/E/C scheme the blueprint specifies. This must not be allowed to balloon into re-litigating already-closed decisions.

**Decision 3 — the submission-count budget is decided explicitly and in writing, now, not implicitly by "submit whenever something seems better."** Zindi competitions typically cap daily/total submissions; the blueprint's own text flags rivals already at 40–80 submissions as a chasing-the-public-LB risk. This project makes at most one new `C`-series submission per genuinely validated candidate change (a real Tier-2 CV improvement, not a hunch), explicitly to avoid becoming the kind of leaderboard-chaser the blueprint warns against — that discipline is itself a competitive advantage against rivals who are not exercising it.

**Decision 4 — a failed Stage 17 gate check does not get patched inside Stage 17.** If the gate finds real leakage, real instability, or a reproducibility break, the fix happens back in the relevant Phase 4/5 stage (with its own tests and registry entry), and Stage 17 is re-run afterward against the corrected artifact. Stage 17 is a gate, not a repair shop — patching inside it would undermine the entire reason it exists as an independent check.

## 3. Task list

Each task lists: Description, Why, Deliverable, Acceptance criteria, Verification, Dependencies, Skill to invoke.

---

### Task 1 — Unblock git and confirm working-tree hygiene

**Description:** Remove the stale `.git\index.lock`, run `git status`, and resolve every uncommitted/untracked item left from the last session (the two Stage 16 backfill files, the Stage 8.5 commits from the earlier commit plan, the `notebooks/04_modeling_experiments_2.ipynb` change, `submission.csv`/`submissions/submission_745a64f.csv`) before any further work in this phase touches the repo.

**Why:** every later task in this plan produces commits and a final git tag. A repo with a stuck lock file or an ambiguous pile of uncommitted changes makes "what commit is the champion actually built from" — the single most important fact Stage 17 needs — impossible to state with confidence. This is infrastructure hygiene that everything downstream depends on.

**What's built:** nothing new — a clean, fully-committed working tree with an accurate `git log`.

**Deliverable:** `git status` returns clean (or explicitly documents any deliberately-untracked file, e.g. `submission.csv` if it's meant to stay gitignored per the blueprint's Stage 1 `.gitignore` convention).

**Acceptance criteria:**
- [ ] `.git\index.lock` removed, `git status` runs without error.
- [ ] Every file from the Stage 8.5 backfill session (climate.py, pipeline.py, test files, scripts, `docs/experiment_registry.md`, `submissions/leaderboard_log.csv`, `tasks/`) is committed with a clear message.
- [ ] `notebooks/04_modeling_experiments_2.ipynb`'s pending change is either committed (with a stated reason) or reverted — not left ambiguous.
- [ ] A decision is recorded on whether `submission.csv` / `submissions/*.csv` should be tracked in git going forward (the blueprint's Stage 1 scaffold lists `submissions/` as "every submission.csv + its metadata row, kept for audit" — meaning they should likely be tracked, unlike `models/`/`mlruns/`; confirm and fix `.gitignore` if it's currently excluding them by accident).

**Verification:** `git log --oneline -15` shows a clean, readable history; `git status` clean.

**Dependencies:** none — this is the entry point for the whole phase.

**Files touched:** `.gitignore` (possibly), whatever is currently uncommitted.

**Skill to invoke:** `/agent-skills:git-workflow-and-versioning` — this is exactly its domain (atomic commits, resolving a messy working tree into clean history) and nothing here is complex enough to need anything heavier.

---

### Task 2 — Retrofit S/F/M/E/C submission IDs onto Stage 10–15's existing results

**Description:** Add explicit experiment IDs to the registry entries that predate the ID scheme: Stage 10.3 (baselines) doesn't need IDs (pre-dates the model zoo), but Stage 11.3 (model zoo comparison) should get `M-001` through `M-00N` per candidate, Stage 13.3 (CatBoost tuning) becomes `M-0xx`, Stage 14.1–14.3 (diversity analysis, ensembling strategies, repeated-CV validation) becomes `E-001`–`E-00N`, and Stage 15.1/15.2 (calibration) gets its own `M`-series or a new letter if the scheme doesn't cleanly fit calibration — decide and document that choice explicitly rather than forcing a bad fit. F-005/F-006/F-007 already have correct IDs from the last session. C-001 already has its ID.

**Why:** the blueprint states IDs are "matched 1:1 to an entry in `docs/experiment_registry.md`" and the machine-readable `leaderboard_log.csv` uses `experiment_id` as a real column — right now that column is empty for C-001 because the upstream M/E-series IDs it should reference don't exist yet. This is also a genuine portfolio-quality issue: a hiring manager or judge reading the registry should see one consistent numbering scheme end to end, not "some stages have IDs, some don't."

**What's built:** ID labels added to existing markdown headers/tables (e.g. `## Stage 11.3 — Model zoo comparison (M-001–M-006)`); `leaderboard_log.csv`'s C-001 row's `experiment_id` field filled in with the specific tuning/calibration experiment it descends from.

**Deliverable:** every experiment in `docs/experiment_registry.md` has an ID; `leaderboard_log.csv`'s `experiment_id` column is non-empty for every row that has an upstream experiment.

**Acceptance criteria:**
- [ ] Every `##` experiment section in the registry has an ID in its own header.
- [ ] IDs are sequential and non-colliding within their series.
- [ ] `leaderboard_log.csv`'s C-001 row references the correct calibration/tuning experiment ID(s).
- [ ] No CV numbers are changed, re-run, or reinterpreted — this task only labels what already exists.

**Verification:** a single read-through of the registry top to bottom should show an unbroken, correctly-ordered ID sequence with no gaps that aren't explained.

**Dependencies:** Task 1 (clean tree to commit against).

**Files touched:** `docs/experiment_registry.md`, `submissions/leaderboard_log.csv`.

**Skill to invoke:** `/agent-skills:documentation-and-adrs` — this is a documentation-fidelity task (recording decisions/structure accurately), not a coding task.

---

### Task 3 — Write the explicit submission-strategy policy for the remaining competition weeks

**Description:** A short, written policy document answering, in advance and explicitly (per ADR-001's own precedent of deciding things "in writing, before" rather than improvising mid-flight): (a) the maximum number of remaining `C`-series submissions this project will make and under what trigger each one is allowed (a genuine validated Tier-2 improvement, not "let's see"); (b) what public-LB movement, if any, is allowed to influence a submission decision, given the blueprint's explicit "scientific evidence and robust CV outrank the public leaderboard" rule; (c) a concrete date/week before 19 Oct by which the *final* submission must be locked, leaving buffer for the unexpected (the blueprint's own timeline already reserves week 7 as buffer); (d) what happens if a competitor's public rank changes dramatically — explicitly, that this project does not react to leaderboard chasing per the blueprint's stated risk.

**Why:** the single most avoidable way to lose a competition this close to a good model is panic-submitting late changes against public-LB noise in the final days, which is exactly the failure mode the blueprint calls out competitors are already exhibiting (40–80 submissions each). Writing the policy now, while calm and with weeks of runway left, is what makes it actually followed under end-of-competition pressure — an unwritten "I'll be disciplined" intention reliably fails exactly when it matters most.

**What's built:** `docs/submission_strategy_policy.md` (new file).

**Deliverable:** a short (under one page), concrete, dated policy document.

**Acceptance criteria:**
- [ ] States the maximum remaining submission count/cadence and the trigger for each.
- [ ] States explicitly that public-LB rank changes among competitors do not, by themselves, trigger a new submission.
- [ ] States the final-lock date/week with the blueprint's week-7 buffer explicitly reserved.
- [ ] Cross-references the go/no-go checkpoint already recorded for C-001.

**Verification:** the document exists, is under one page, and a future you (or a judge reading the portfolio) can state the submission rule in one sentence after reading it.

**Dependencies:** Task 2 (accurate registry to reference).

**Files touched:** new `docs/submission_strategy_policy.md`.

**Skill to invoke:** `/agent-skills:documentation-and-adrs` — this is precisely an ADR: a decision, its rationale, and its consequence, recorded before it's needed.

---

### Task 4 — Re-verify leakage checks against the actual champion artifact (Stage 17, check 1 of 6)

**Description:** Re-run Stage 3's forensics checks and Stage 9's OOF-encoding leakage tests specifically against `catboost_tuned` + Platt calibration's actual feature set and fitted state — not a general "we checked this back in Stage 3/9" assumption. Confirm: no twin-record leakage across the folds this specific config was tuned/calibrated on; the OOF target-encoding used by this exact feature matrix never saw its own fold's labels; Platt calibration (Stage 15.2) was fit on OOF predictions only, never in-fold.

**Why:** Stage 3/9's checks were run once, generically, early in the project. A champion gate exists specifically because a later config (different hyperparameters, added calibration step) can silently reintroduce a leakage path the general checks didn't anticipate — e.g. confirming calibration itself doesn't leak is a Stage 15-specific concern the original Stage 3/9 work couldn't have covered, since calibration didn't exist yet when those checks were written.

**What's built:** a short verification script or notebook cell that asserts, programmatically, on the actual champion's data: (a) OOF encoding fold-assignment never overlaps with the fold being scored (already should have a unit test — confirm it exists and passes for this config specifically); (b) Platt calibration's fit-fold and score-fold are disjoint in `cv_calibrate` (read the actual code path, don't assume from the docstring).

**Deliverable:** a written pass/fail verdict with evidence (not just "looks fine") added to Stage 17's gate record.

**Acceptance criteria:**
- [ ] Existing OOF-encoding leakage tests (`tests/` — confirm which file) are located, read, and confirmed to cover this exact champion config's code path (not a stale test for an earlier config).
- [ ] `cv_calibrate`'s fold-disjointness is read directly from `calibrate.py` and confirmed correct, not assumed.
- [ ] Verdict recorded: "checked against the actual champion artifact, no leakage found" or a named, specific problem.

**Verification:** the relevant test(s) pass; the code read-through is documented (file/line references), not just asserted.

**Dependencies:** Task 1.

**Files touched:** none required (a verification task); possibly a new small script under `scripts/` if no existing test covers this exact path.

**Skill to invoke:** `/agent-skills:doubt-driven-development` — explicitly appropriate per its own trigger list ("stakes are high," "a confident output would be cheaper to verify now than to debug later"). This check exists precisely to cross-examine an assumption ("Stage 3/9 already covered this") rather than accept it, which is doubt-driven-development's exact purpose. Follow with `/agent-skills:test-driven-development` only if a genuine gap is found and a new test needs writing.

---

### Task 5 — Re-run adversarial validation on the champion's residuals (Stage 17, check 2 of 6)

**Description:** Per the blueprint's Stage 17 checklist item "no unexplained train/test dependency (re-run adversarial validation on this candidate's residuals)": train a classifier to distinguish train rows from test rows using the champion's *prediction residuals/errors* as an additional signal, not just the raw features (which Stage 4 already checked). This asks a sharper question than Stage 4 did: does the champion's specific error pattern itself carry a train/test fingerprint, which would suggest the model learned something location-specific rather than genuinely generalizable.

**Why:** Stage 4's adversarial validation checked whether the *raw data* distinguishes train from test (it does, driven mostly by geography — already understood and expected). This is a different, sharper question specific to the champion: does the *model's own mistakes* correlate with train/test membership in a way that would suggest it's exploiting something that won't hold on the real private test set. This is exactly the kind of check that's cheap to do now and expensive to discover was missing after 19 Oct.

**What's built:** a script producing OOF residuals for `catboost_tuned` on train, and the model's raw predictions on Test.csv (no labels available, so use predicted-probability distribution comparison as a proxy alongside a train-only adversarial-validation-on-residuals check where labels exist).

**Deliverable:** a short written finding: either "no unusual residual pattern found, consistent with genuine generalization" or a named concern requiring follow-up.

**Acceptance criteria:**
- [ ] OOF residuals for the actual champion config are computed (reuse Stage 14.1's OOF infrastructure — do not rebuild it).
- [ ] Residual-based adversarial check run and its AUC reported (a value near 0.5 supports "no unusual fingerprint"; a high value is a named concern, not something to explain away).
- [ ] Prediction-probability distribution on Test.csv compared against OOF-predicted-probability distribution on Train.csv (a large mismatch here is itself informative even without test labels).

**Verification:** the finding is written down with actual numbers, either clearing the gate or naming a specific concern for Task 8.

**Dependencies:** Task 1.

**Files touched:** new script under `scripts/` (e.g. `scripts/run_stage17_residual_adversarial_check.py`).

**Skill to invoke:** `/agent-skills:doubt-driven-development` for the same reason as Task 4 — this is precisely a "cross-examine before it stands" check on the champion. `/agent-skills:test-driven-development` is not the right tool here since this is a one-time diagnostic, not ongoing behavior to regression-test.

---

### Task 6 — Confirm stability under geographic validation and reproducibility from a clean environment (Stage 17, checks 3–4 of 6)

**Description:** Two checks, bundled since both are about "does this actually work the way it's claimed to, not just once, on this machine": (a) re-confirm the champion's Tier-2 stability numbers (mean/std) are exactly what's recorded in Stage 13.3/13.4/14.3 by re-running the evaluation once more (not re-tuning — just re-scoring the frozen config) and diffing against the recorded numbers; (b) reproduce the full pipeline from a genuinely clean environment (fresh clone or fresh conda env per `environment.yml`/`requirements-lock.txt`) through to a `submission.csv` that matches the archived `submissions/submission_745a64f.csv` (or documents exactly why it differs, e.g. a non-determinism source that needs a fixed seed).

**Why:** this is the blueprint's own stated purpose for Stage 15.3's "single scripted entry point" — but that claim has never actually been tested from a clean environment in this project. A pipeline that only reproduces on the machine that built it is not actually reproducible, and "reproducible from a clean environment" is explicitly one of Stage 17's named gate criteria, not an assumption to wave through because it was scripted.

**What's built:** nothing new in `src/` — this is a verification run. If it fails, the fix (e.g. a missing seed, an unpinned dependency version) is a real Task 8 item.

**Deliverable:** a written confirmation that a clean-environment run reproduces the archived submission bit-for-bit (or numerically within float tolerance), with the exact commands used.

**Acceptance criteria:**
- [ ] Tier-2 mean/std re-scored once against the frozen `catboost_tuned` config; matches Stage 13.3/14.3's recorded numbers (or the discrepancy is explained, e.g. random-seed sensitivity not previously pinned).
- [ ] A fresh environment (new conda env or venv, from the pinned lockfile) successfully runs `predict.py` end to end.
- [ ] The resulting `submission.csv` matches the archived one (row-for-row, or numerically within a stated tolerance), and any difference is explained, not ignored.

**Verification:** the diff (or match) between the clean-environment output and the archived submission file is shown explicitly, not asserted.

**Dependencies:** Task 1.

**Files touched:** none expected; a note added to `docs/experiment_registry.md`'s Stage 17 section either way.

**Skill to invoke:** `/agent-skills:debugging-and-error-recovery` if the clean-environment run fails and needs root-causing; otherwise no skill is needed beyond careful execution — this is closer to `/agent-skills:test-driven-development`'s "prove it" spirit for an environment claim rather than a code claim, so treat any needed fix as a genuine bug per that skill's Prove-It Pattern (reproduce the environment failure, then fix, then re-verify).

---

### Task 7 — Validate `submission.csv` against `SampleSubmission.csv`'s schema exactly (Stage 17, check 5 of 6) and finalize the calibrated-probability sanity check (check 6 of 6)

**Description:** Two remaining named Stage 17 checks: (a) a strict schema-equality check — column names, order, row count, ID coverage (every Test.csv ID present exactly once), value domain (predictions in the competition's required range/format) — between the archived submission and `SampleSubmission.csv`; (b) confirm the "reasonable, calibrated probability behavior" check already partially done in Stage 15.1/15.2 is complete and its verdict is written down plainly in the registry (the blueprint explicitly wants "checked, didn't matter" recorded as a real artifact either way, per Stage 1's cross-cutting principle).

**Why:** a schema mismatch is the single most embarrassing and avoidable way to score zero or get disqualified on Zindi — it must be checked mechanically, not eyeballed. And an unrecorded calibration verdict leaves Stage 17's gate incomplete on paper even if it was actually fine in practice, which matters both for competition audit trail and portfolio credibility.

**What's built:** a small `pytest` test (`tests/test_submission_schema.py` or similar) that runs this check automatically, so every future submission gets this validation for free rather than requiring a manual re-check each time — directly serves both the "must win" goal (never risk a disqualified submission) and Phase 7's later CI work (the blueprint already wants this test to exist per Stage 18).

**Deliverable:** a passing automated schema-validation test; the calibration verdict from Stage 15.1/15.2 restated explicitly in Stage 17's gate record.

**Acceptance criteria:**
- [ ] Test asserts column names/order match `SampleSubmission.csv` exactly.
- [ ] Test asserts row count and ID set match `Test.csv` exactly (no missing/extra/duplicate IDs).
- [ ] Test asserts prediction value domain is valid (matches whatever the competition's submission format actually requires — check the info page/data dictionary, don't assume `{0,1}` vs. probability without confirming which Zindi wants).
- [ ] Test passes against the archived `submissions/submission_745a64f.csv`.
- [ ] Calibration verdict (Stage 15.1: needed or not; Stage 15.2: which method chosen and why) is quoted plainly in Stage 17's gate record, not just referenced by stage number.

**Verification:** `pytest tests/test_submission_schema.py -v` passes.

**Dependencies:** Task 1.

**Files touched:** new `tests/test_submission_schema.py`.

**Skill to invoke:** `/agent-skills:test-driven-development` — this is squarely a "write the test first, prove the behavior" task, and it's also genuinely reusable regression protection for every future submission, which is exactly what TDD is for here (not a one-off script like Tasks 4–6's diagnostics).

---

### Task 8 — Synthesize Stage 17's gate verdict and decide: promote, fix-and-recheck, or hold

**Description:** Bring Tasks 4–7's individual findings together into one explicit go/no-go decision on the champion, following the blueprint's Stage 17 checklist verbatim: improves the competition score (already established); improvement holds across Tier 2 folds not just on average (Stage 13.3/14.3 std already recorded); no leakage found (Task 4); no unexplained train/test dependency (Task 5); reasonable calibrated probability behavior (Task 7); stable under geographic validation (Task 6); reproducible from clean environment (Task 6); schema-valid (Task 7). If every check passes: promote formally and proceed to Task 9. If any check fails: name the specific failure, route it back to the relevant Phase 4/5 stage as a new, properly-scoped fix task (not patched inside Stage 17 per ADR/Decision 4 above), and re-run only the affected Stage 17 checks once fixed — don't re-run everything blindly.

**Why:** this is the actual point of the whole phase — an explicit, written, checklist-driven decision beats an implicit "it's probably fine, ship it" every time, especially with $500 and a portfolio centerpiece on the line. Writing the verdict down, with each check's evidence linked, is also exactly the kind of artifact a technical judge or hiring manager reading the registry would find credible in a way a bare "champion locked" sentence would not.

**What's built:** a `## Stage 17 — Champion Model Gate` section in `docs/experiment_registry.md` with all 8 checks listed, each with a pass/fail verdict and a one-line evidence pointer (linking to Tasks 4–7's specific findings), and a final promote/hold decision.

**Deliverable:** the gate section in the registry; a clear final sentence stating whether `catboost_tuned` (trial #26) + Platt calibration is the promoted Phase 5/6 champion.

**Acceptance criteria:**
- [ ] All 8 blueprint-named checks appear with individual verdicts.
- [ ] Every verdict links to real evidence produced in Tasks 4–7 (not asserted from memory).
- [ ] The final decision sentence is unambiguous.
- [ ] If any check failed and was fixed, the fix's own commit/PR is referenced and the specific re-checked items are named.

**Verification:** re-reading the gate section alone (without needing to reopen Tasks 4–7) is enough to understand the verdict and why.

**Dependencies:** Tasks 4, 5, 6, 7.

**Files touched:** `docs/experiment_registry.md`.

**Skill to invoke:** `/agent-skills:documentation-and-adrs` for writing the gate record itself; if any check actually failed, invoke `/agent-skills:debugging-and-error-recovery` (root-cause it) followed by `/agent-skills:test-driven-development` (prove the fix) for that specific fix, before returning to close this task.

---

### Task 9 — Tag `competition-submission-final` and lock the champion artifact

**Description:** Once Task 8's gate passes, create the git tag `git tag competition-submission-final` at the exact commit that produced the currently-standing submission (or a corrected commit if Task 8 required a fix), per the blueprint's Stage 15.3 deliverable. Record the tag's commit hash explicitly in the registry so every later interpretability/model-card artifact (Phase 8) can be pinned to it without ambiguity.

**Why:** the blueprint calls this out as "the single most important reproducibility guarantee in the whole project" — without it, "which commit is the champion" stays a matter of git-log archaeology (as Task 1/this session already had to do once) rather than a fact anyone can check in one command.

**What's built:** the git tag itself; a one-line registry entry recording it.

**Deliverable:** `git tag -l` shows `competition-submission-final`; `git show competition-submission-final --stat` matches the expected commit.

**Acceptance criteria:**
- [ ] Tag exists and points to the correct commit (the one that generated the currently-standing best submission, post any Task 8 fixes).
- [ ] Tag is pushed to the remote (`git push origin competition-submission-final`) so it's visible on GitHub for the portfolio.
- [ ] Registry records the tag's commit hash and date.

**Verification:** `git show competition-submission-final` on a clean clone reproduces the same commit content as locally.

**Dependencies:** Task 8 (must pass the gate first — never tag a candidate that hasn't been gated).

**Files touched:** none in-repo besides the registry note; a git tag object.

**Skill to invoke:** `/agent-skills:git-workflow-and-versioning` — tagging conventions are explicitly its domain.

---

### Task 10 — Set up the ongoing Phase 6 cadence for the remaining competition weeks

**Description:** A lightweight recurring check-in (not a new stage, a habit): each time a genuinely new validated result would change the champion (which, given Stages 8.5/14 are closed, should be rare unless something specific changes — e.g. new competition data, a rule clarification, or a deliberately time-boxed new idea after Stage 17 is locked), re-run Task 3's policy check before submitting, log it with the next sequential `C`-series ID, and re-run Task 7's schema test before upload every single time without exception.

**Why:** this is what makes Stage 16's discipline durable through the emotionally pressured final weeks of a competition, rather than a one-time exercise that gets abandoned under deadline pressure — which is exactly when discipline matters most for actually winning.

**What's built:** nothing new — this formalizes reusing Tasks 3/7's artifacts every time, going forward.

**Deliverable:** a one-paragraph addition to `docs/submission_strategy_policy.md` (from Task 3) stating this cadence explicitly.

**Acceptance criteria:**
- [ ] Policy states: every future submission gets a `C`-series ID, a registry entry, a `leaderboard_log.csv` row, and a passing schema test before upload — no exceptions, regardless of time pressure.

**Verification:** re-read the policy; it should read as a checklist you'd actually be able to follow on 18 Oct at 11pm under real pressure, not an idealized process that assumes calm conditions.

**Dependencies:** Tasks 3, 7, 9.

**Files touched:** `docs/submission_strategy_policy.md`.

**Skill to invoke:** none required — this is a short documentation addendum, below the threshold where a skill adds value (mirroring how Stage 8.5's Task 8 correctly used "no skill" for its own cheap, low-stakes step).

---

## 4. Checkpoints

**Checkpoint A (after Task 3):** Stage 16 is functionally complete — registry fully ID'd, submission policy written. Pause here and confirm with the user before starting Stage 17's technical checks, since Stage 17 involves real code/test work rather than documentation.

**Checkpoint B (after Task 8):** the single most important checkpoint in this entire plan — this is the actual win/loss decision point. Do not proceed to Task 9 (tagging) on anything but an unambiguous pass.

**Checkpoint C (after Task 10):** Phase 6 is fully closed. Next decision point is explicitly out of scope for this plan: whether remaining competition time (per the blueprint's own week-7 buffer framing) goes toward Phase 7/8 portfolio engineering, a fresh idea-refine pass (only now legitimate, since Phase 6 has closed the loop on the current champion rather than leaving it ambiguous), or simply holding position and watching the leaderboard per Task 3's policy.

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Task 5/6's deeper checks surface a real problem this late | ADR Decision 4 — route it back to the correct Phase 5 stage properly rather than rushing a patch; there is still buffer time per the blueprint's week-7 reserve. |
| Temptation to submit reactively if a rival's public rank jumps | Task 3's written policy exists specifically to be pointed at in that moment. |
| Retrofitting IDs (Task 2) creates busywork that delays the actual gate (Task 4-8) | Time-box Task 2 explicitly — it's pure labeling of already-known numbers, should take a fraction of the time of any single Stage 17 check. |
| Clean-environment reproduction (Task 6) fails for an environment reason unrelated to the model itself | Still must be fixed — an unreproducible pipeline is a real gate failure per the blueprint's own criteria, not a technicality to wave through given "we must win." |

## 6. Open questions requiring your decision

1. Task 1: should `submission.csv`/`submissions/*.csv` be tracked in git going forward, given the blueprint's Stage 1 scaffold lists them under "kept for audit"? Confirm before Task 1 closes.
2. Task 3: what's your actual maximum-submission-count comfort level for the remaining ~4 weeks? The policy needs a real number from you, not one I invent.
3. Task 7: confirm the exact submission value-domain requirement (probability vs. hard 0/1) directly from the Zindi competition page rather than assumed — this affects the schema test's correctness.
