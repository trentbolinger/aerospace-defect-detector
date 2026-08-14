# Design of Experiments: Confidence Threshold vs. Image Resolution

A 2x2 full factorial experiment testing how confidence threshold and
image resolution affect classification accuracy and rejection rate in
the Industrial Surface Defect Detector.

## Methodology

Two factors, each tested at two levels, for 4 total runs:

- **Resolution**: high (224x224, the model's native training resolution)
  vs. low (downsampled to 128x128, then upsampled back to 224x224 to
  simulate a lower-quality source image at the correct input size)
- **Confidence threshold**: 60% (loose) vs. 85% (the current production
  value used in the live demo)

All 4 combinations were run against the same held-out validation split
(384 real labeled images, 20% of the dataset, same split method used
during training) using the actual trained model
(`outputs/best_model.pth`). Two metrics were measured per run:

- **Accuracy**: percentage of accepted predictions that matched the true
  label
- **Rejection rate**: percentage of images whose top confidence fell
  below the threshold and were therefore rejected rather than classified

## Results

| Run | Resolution | Threshold | Accuracy | Rejection Rate |
|---|---|---|---|---|
| 1 | High | 60% | 100.00% | 0.00% |
| 2 | High | 85% | 99.74% | 0.26% |
| 3 | Low | 60% | 64.06% | 3.91% |
| 4 | Low | 85% | 60.42% | 12.50% |

## Key Finding

Resolution is the dominant factor. Accuracy is essentially perfect at
high resolution (99.74-100%) and collapses to 60-64% at low resolution,
regardless of threshold.

More importantly, the two factors interact. At high resolution, raising
the confidence threshold from 60% to 85% costs almost nothing (a 0.26
point accuracy drop, 0.26% rejection rate) because the model is already
confident and correct on nearly every image. At low resolution, the same
threshold increase causes rejection rate to jump from 3.91% to 12.5%,
while accuracy actually decreases slightly (64.06% to 60.42%) rather
than improving. The stricter threshold is not successfully filtering
out wrong predictions at low resolution. It is simply rejecting more
images overall without making the accepted predictions any more
reliable.

**Practical implication**: raising the confidence threshold is not an
effective mitigation for degraded image quality. If low-resolution or
poor-quality input images are a realistic risk in deployment (camera
degradation, poor lighting, compression), the correct intervention is
addressing image quality directly, not tightening the confidence
threshold, since tightening the threshold under those conditions
increases rejected images without recovering accuracy.

## How to Reproduce

    cd aerospace-defect-detector
    python3 doe/doe_experiment.py
