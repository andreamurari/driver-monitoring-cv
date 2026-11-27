# Model card

Work in progress (day 7 of 14). It describes the models trained so far; it will be
completed with the final results on the test drivers.

## Intended use

- Personal study and portfolio project on driver distraction recognition from
  in-cabin images.
- Not a safety system. Not intended for use in a vehicle, for monitoring real
  people, or for any commercial purpose.

## Models

| Model | Task | Input | Params | Training | Weights published |
| --- | --- | --- | --- | --- | --- |
| ResNet18 (torchvision, ImageNet) | 10-class distraction classifier | 320 px letterbox | 11.2 M | head 3 epochs, then full fine-tuning 5 epochs, AdamW, cosine | no |
| ViT-S/16 (timm, ImageNet-21k) | same | 224 px | 21.7 M | same schedule, lr 5e-5, weight decay 0.05 | no |
| YOLO11n (Ultralytics, COCO) | detector: phone, cup, hand, face | 640 px | 2.6 M | 60 epochs on 172 annotated images, CPU | no |

Configs: `configs/resnet18_320.yaml`, `configs/vit_s16.yaml`, `configs/yolo11n_batch2.yaml`.

## Data

- State Farm Distracted Driver Detection (Kaggle): 22,424 labelled images,
  26 drivers, 10 classes. See [DATA.md](DATA.md) for terms of use.
- Splits by driver (`splits.json`): 18 train, 4 validation, 4 test drivers.
- Detector: 254 images annotated by hand in two batches (172 / 41 / 41, same driver split),
  see [annotations/GUIDELINES.md](annotations/GUIDELINES.md). Labels are not published.

## Results so far

| Model | Metric | Value | Set |
| --- | --- | --- | --- |
| ResNet18, 320 px | accuracy / macro-F1 / log-loss | 0.912 / 0.908 / 0.276 | validation, 4 unseen drivers |
| ViT-S/16, 224 px | log-loss (2 seeds) | 0.360 / 0.347 | validation |
| YOLO11n | mAP@0.5 / mAP@0.5:0.95 (2 seeds) | 0.789 / 0.376, 0.773 / 0.375 | detector test, 41 images |
| ResNet18, 320 px | CPU latency, median (p95) | 50 ms (53 ms) | batch 1, 4 threads |

The classifier numbers are on the validation drivers, which were also used to choose the
checkpoint and the configuration, so they are slightly optimistic. The test drivers have
not been used yet. Full log: [reports/experiments.md](reports/experiments.md).

## Known failure modes

From [reports/error_analysis_v1.md](reports/error_analysis_v1.md):

- **Talking to the passenger (c9)** has recall 0.65 and is mostly predicted as safe
  driving: a missed distraction. Grad-CAM suggests the model relies on the background
  rather than on the head for this class.
- **A phone held at chest height** is read as a cup (drinking), and **a hand in the hair**
  as reaching behind: the arm's pose weighs more than the small object.
- **Accuracy varies by driver**: 0.83 to 0.97 across the 4 validation drivers.
- **Detector**: phones are the weakest class (AP@0.5 0.58-0.64) and also the most
  ambiguous to annotate. Errors are mostly missed objects. Face recall is 0.40-0.53 for one
  test driver (face in profile against an overexposed window, partly covered by hair) and
  0.94-1.00 for the others.
- **Label noise**: the same annotator agrees with themselves at mean IoU 0.73, so
  mAP@0.5:0.95 mostly measures label noise; mAP@0.5 is the main metric.

## Known limits

- Staged dataset: the car was towed by a truck and the "drivers" were not actually
  driving. One fixed 2D visible-light camera, daytime, few drivers. No infrared,
  no measured depth, no in-vehicle validation.
- With 4 validation and 4 test drivers, every metric is noisy; two seeds of the same
  configuration differ by about 0.1 log-loss.
- No trained weights, annotation labels or dataset images are published (see DATA.md).

## Risks and privacy

The dataset shows real people. No images of them are published in this repository.

During annotation, the pre-trained models used for pre-labels (MediaPipe for hands and
faces) worked noticeably worse for drivers with darker skin and for heavier-set drivers
(qualitative observation). A driver monitoring system must work equally well for everyone:
in the per-driver detector metrics the worst driver for faces is also a heavier-set
driver, but with 4 test drivers pose, lighting and body type cannot be separated.
Performance across demographic groups cannot be measured reliably with 26 drivers.
