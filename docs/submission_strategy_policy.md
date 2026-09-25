# Submission Strategy Policy — Phase 6, Stage 16 (Task 3)

**Author:** Henry Otsyula. **Written:** 2026-09-19, with C-001 (0.830979519, 2026-09-14) the only submission made so far and ~4 weeks left before the competition closes on 2026-10-19. **Decided now, deliberately, while calm** — per ADR-001's own precedent of locking a decision in writing before pressure makes it hard to follow, and because the blueprint names this exact failure mode (panic-submitting against public-LB noise in the final days) as the single most avoidable way a good model loses.

## 1. Submission format (confirmed from data, not assumed)

`data/raw/SampleSubmission.csv` has two target columns: `TargetF1` (hard 0/1) and `TargetRAUC` (raw probability, 0-1). `predict.py`'s current output matches this shape exactly. No open question here — already correct.

## 2. Maximum remaining submissions and the trigger for each

**At most 3 more `C`-series submissions before the lock date (Section 4)** — C-002, C-003, C-004 at most, not a running total across S/F/M/E rows (those don't touch Zindi).

**The only valid trigger for a new `C`-series submission:** a candidate that (a) improves the Tier-2 mean over the current champion (M-007+M-008, i.e. C-001's config) on the same 5×5 repeated geographic CV used throughout this project, AND (b) does not blow up Tier-2 std — the identical standing decision rule already used to reject F-005, F-006, F-007, and the Stage 14 ensembles. "It looks like it might help" or "it's been a while since the last submission" are not triggers. A rejected experiment (the outcome of every Stage 8.5/14 test run so far) does not submit, no matter how much work went into testing it.

## 3. What public-LB movement is allowed to influence a submission decision

**None, by itself.** Per the blueprint's own explicit rule ("scientific evidence and robust CV outrank the public leaderboard, not the reverse"): the public leaderboard is one partial-test-set score, not the private one that decides prize money, and rivals already at 40-80 submissions each are the blueprint's own named example of what chasing it looks like. Public LB score is *read and recorded* (as the go/no-go checkpoint already did for C-001), but it is diagnostic information about whether local CV is trustworthy — never itself a reason to submit. A future submission is triggered by Section 2's rule alone.

## 4. Final-lock date, with the blueprint's week-7 buffer honored

**Final `C`-series submission must be locked by 2026-10-12 — one full week before the 2026-10-19 close.** This maps the blueprint's own timeline (week 7, 13-19 Oct, reserved as buffer "for last CV-guided submissions before close," not for new modeling work) onto real calendar dates. No new model/feature work starts inside that final week; it exists only to resubmit an already-validated candidate if something breaks, or to do nothing if C-001 (or its successor) is already the final answer.

## 5. Competitor rank changes — explicitly not a trigger

If another competitor's public rank jumps, drops, or otherwise moves dramatically, **this project does not react.** No new experiment is started, no submission is made, in response to a rank change alone. The blueprint identifies leaderboard-chasing as a real, named risk among rivals already visible in this competition; not reacting to it is itself the competitive discipline this policy exists to enforce. The only thing that changes this project's plans is new evidence from this project's own CV (Section 2).

## 6. Cross-reference: the go/no-go checkpoint already run

C-001's go/no-go comparison (`docs/experiment_registry.md`, Submission log section) found public LB (0.830979519) sits *above* local Tier-2 mean (0.8156) by +0.0154 — the safe direction. This policy does not change because of that result; it's recorded here only so a future re-read of this policy has the full picture in one place without needing to reopen the registry.

## 7. Ongoing cadence (every submission, not just the next one)

Every future submission — C-002 onward — gets: a new sequential `C`-series ID, a registry entry with full Tier 1/2/3 numbers and the trigger that justified it, a `submissions/leaderboard_log.csv` row, and Stage 17's schema-validation test passing before upload. No exceptions, regardless of how close 2026-10-19 is when it happens.

## 8. Phase 6 closure note (added 2026-09-19, Task 10)

Stage 16 (registry backfill, this policy) and Stage 17 (champion gate) are both
complete as of 2026-09-19 — all 6 named Stage 17 checks passed cleanly for C-001
(`docs/experiment_registry.md`, Stage 17 gate summary), and commit `19bdb9a` is
tagged `competition-submission-final`. Section 7's cadence rule is what governs
everything from here: it already required Stage 17's schema-validation test to
pass before any future upload, which is now a real, checked-in test
(`tests/test_submission_schema.py`), not an aspiration. No new policy is needed for
C-002 onward — Section 7 already covers it, and this note exists only to mark that
Phase 6 itself is closed, not still in progress.

---

**One-sentence summary of this policy:** submit again only when local Tier-2 CV genuinely improves without blowing up variance, never in reaction to the public leaderboard or a rival's rank, and lock everything by 2026-10-12 so the final week is buffer, not a scramble.
