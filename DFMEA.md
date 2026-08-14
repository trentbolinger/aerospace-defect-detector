# DFMEA: AI Surface Defect Detector

**Design Failure Mode and Effects Analysis** of the ResNet-18 steel surface
defect classifier, scoped to a hypothetical deployment in a high-consequence
manufacturing context (e.g., aerospace component inspection), consistent
with the target use case for this analysis.

## Methodology

Each failure mode is scored on three 1–10 scales:

- **Severity** — how bad the consequence is if this failure occurs
- **Occurrence** — how likely this failure is to happen
- **Detection** — how likely the failure is to go *unnoticed* (high = poor detection)

**Risk Priority Number (RPN) = Severity × Occurrence × Detection**

Higher RPN indicates a higher-priority risk to address.

## Analysis

| Failure Mode | Severity | Occurrence | Detection | RPN |
|---|---|---|---|---|
| Model drift (undetected performance degradation over time) | 9 | 7 | 9 | **567** |
| False-negative defect classification (real defect missed) | 9 | 4 | 9 | 324 |
| Confidence-threshold miscalibration | 4 | 8 | 4 | 128 |
| False-positive rejection (good part wrongly flagged) | 3 | 4 | 5 | 60 |

## Key Findings

**The highest-priority risk is not a wrong prediction — it's the absence of
post-deployment monitoring.** Model drift scores highest (RPN 567) not
because any single failure is worse than a missed defect, but because it is
systemic (affects the full prediction stream, not one part) and currently
has zero detection mechanism. The project's calibrated-rul work
independently demonstrated this same phenomenon — model error can increase
sharply under distribution shift — reinforcing that this is a real,
recurring risk for deployed ML systems, not a theoretical concern.

Confidence-threshold miscalibration is scored as a currently latent risk:
the project does not yet use confidence scores for any downstream decision,
so miscalibration causes no direct harm today. However, it directly
undermines the most natural mitigation for the top two risks (routing
low-confidence predictions to human review), so it must be resolved before
that mitigation is trustworthy.

## Recommended Mitigations (Prioritized)

1. **Add post-deployment monitoring** — track prediction confidence
   distributions and accuracy over time against a labeled reference set to
   detect drift before it silently degrades performance.
2. **Add confidence-threshold flagging** — route low-confidence predictions
   to human review rather than fully automated accept/reject, directly
   reducing Detection risk on the two highest-RPN failure modes.
3. **Run a calibration check** — evaluate whether stated confidence scores
   match true accuracy (e.g., a reliability diagram) before relying on any
   confidence threshold for (2). Deep neural networks are commonly
   overconfident by default without explicit calibration.
