# Lesson 11 — Teaching notes

The capstone has no tests, so the teacher's review takes their place.

## Learning objectives

By the end, the learner can:

1. Turn a vague idea into an inputs → outputs task with a metric.
2. Split data without leakage, and justify the grouping.
3. Build and beat a baseline, or recognise honestly that the baseline wins.
4. Diagnose and iterate using training curves and error analysis.

## Format

| Session | Activity |
|---|---|
| Kick-off (30 min) | Learner presents the task sentence, metric, split plan, baseline and target. Teacher challenges each. |
| Check-in (30 min) | Baseline score, first training curves, worst-20 mistakes. |
| Review (45 min) | Final results against the target; what they would try next. |

## Review rubric

| Area | Good | Needs work |
|---|---|---|
| Task | One clear sentence; output is something a person acts on | "Use AI on my photos" |
| Metric | In real units; chosen before training | Only the training loss reported |
| Split | Grouped (by shoot, user, time), justified | Random rows with obvious near-duplicates across sets |
| Baseline | Implemented, scored on the same split | Missing |
| Training | Curves shown and diagnosed; sensible fixes tried | Hyperparameters changed at random |
| Error analysis | Worst examples inspected; data issues found or ruled out | None |
| Honesty | Reports where the model does not beat the baseline | Only the best run shown |

## Questions to ask at the kick-off

1. What would you do with a perfect model's output? (If the answer is vague, the task is.)
2. Which examples are similar enough to leak? How will you split them?
3. What is the dumbest model that could work, and what score do you expect from it?
4. How much labelled data do you have? Is a pretrained backbone available for this kind of input?
5. What result would make this worth using? Write it down now.

## Common pitfalls

- Spending the first week on model architecture before having a data pipeline.
- Evaluating on data the model (or the pretrained backbone) has already seen.
- Reporting MSE on standardised targets, which nobody can interpret.
- Declaring success from training accuracy.
