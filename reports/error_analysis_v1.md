# Error analysis v1

Model: `resnet18_320_s0` (ResNet18, 320 letterbox, the best so far). Data: **validation**,
4 drivers never seen in training, 4,347 images. The test set was not used.
Reproduce: `python -m dms.evaluate --run runs/kaggle/resnet18_320_s0` and
`python -m dms.explain.gradcam --run runs/kaggle/resnet18_320_s0 --top 30`.

No dataset image appears in this report (State Farm competition rules): the examples and
Grad-CAM maps were inspected locally and are described here in words.

**Summary.** The model gets 8.8% of the images wrong (382). Almost half of the errors
involve one class, c9 "talking to the passenger", often taken for safe driving: a false
negative, the most serious kind of error for a safety system. The model's confident errors
follow two patterns (a phone read as a cup, a raised arm read as "reaching behind"), and
Grad-CAM suggests that for c9 the model uses the background instead of the gaze direction.

## Metrics

| Metric | Value |
|---|---|
| Accuracy | 0.912 |
| Macro-F1 | 0.908 |
| Log-loss | 0.2763 (identical to the Kaggle measurement: local pipeline verified) |

![Normalised confusion matrix](figures/confusion_resnet18_320_s0.png)

**Per class** (recall = how many images of the class are recognised; precision = how many
predictions of the class are correct):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| c0 safe driving | 0.804 | 0.874 | 0.837 |
| c1 texting, right | 0.995 | 0.872 | 0.929 |
| c2 phone call, right | 0.996 | 0.991 | 0.993 |
| c3 texting, left | 0.942 | 1.000 | 0.970 |
| c4 phone call, left | 0.982 | 0.982 | 0.982 |
| c5 radio | 0.861 | 0.978 | 0.916 |
| c6 drinking | 0.918 | 1.000 | 0.957 |
| c7 reaching behind | 0.821 | 0.964 | 0.887 |
| c8 hair and makeup | 0.907 | 0.801 | 0.851 |
| **c9 talking to the passenger** | 0.917 | **0.647** | **0.759** |

**Main confusions** (share of the true class):

| True → predicted | Share |
|---|---|
| c9 passenger → c0 safe driving | 21% |
| c8 hair → c7 reaching behind | 10% |
| c0 safe driving → c5 radio | 8% |
| c1 texting right → c6 drinking | 6% |
| c9 passenger → c5 radio | 6% |

**Per driver:**

| Driver | Images | Accuracy | Errors | Classes with most errors |
|---|---|---|---|---|
| p022 | 1,233 | 0.965 | 43 | c9 (29) |
| p024 | 1,226 | 0.940 | 73 | c9 (36), c1 (30) |
| p021 | 1,237 | 0.875 | 155 | c8 (60), c9 (37), c1 (28) |
| p039 | 651 | 0.829 | 111 | c9 (56), c0 (39) |

c9 is the class with most errors for all four drivers (158 of the 382 errors).

## Hypotheses written before looking at the results

| Hypothesis | Outcome |
|---|---|
| The most confused pairs are left/right: c2↔c4 and c1↔c3 | **Refuted.** c2, c3 and c4 have recall between 0.98 and 1.00. The camera is on the passenger side: a phone in the right hand (close, visible) and one in the left (far, behind the head) give very different pictures, not mirror images |
| c7 will be the best-recognised class, because the posture changes a lot | **Partly.** High recall (0.964) but not the highest (c3 and c6: 1.000), and among the lowest precision (0.821): the "arm away from the body" posture also attracts images of other classes |
| Less than 10 accuracy points between the best and the worst driver | **Refuted.** 13.6 points (p022 0.965, p039 0.829) |

The first point also explains a day 3 result: horizontal flip with label swap did not help,
because a mirrored "right hand" image does not look like a real "left hand" photo (the
steering wheel changes side, the hand stays close to the camera).

## The 30 most confident errors

The 30 errors with the highest confidence (all above 0.93), classified by hand from the
images:

| Category | Errors | What is inside |
|---|---|---|
| Model error (clear image) | 18 | arm raised to the hair read as "reaching behind" (c8 → c7, 8); phone in hand read as a cup (c1/c2 → c6, 5); hand at the mouth read as a phone (c8 → c4, 2); 3 others |
| Questionable label | 9 | 6 from a single p022 sequence labelled c9 in which the driver looks ahead with her hands on the wheel; 3 p021 images that really look like the predicted class |
| Occlusion | 2 | phone mostly hidden by the hand (c1 → c6) |
| Unusual pose | 1 | |

- **Few moments, repeated.** Consecutive frames are near identical: two sequences alone
  (p021 with the raised arm, p022 looking ahead) make 14 of the 30 errors. The 30 errors
  correspond to about 10-12 distinct situations.
- **Confident and uncertain errors.** All 30 come from p021 and p022, none from p039,
  which has the lowest accuracy: p039 is often wrong but with little confidence, p021 is
  wrong with confidence. The latter weigh more on the log-loss.

## Where the model looks (Grad-CAM)

Maps computed on the last convolutional block (`layer4`, a 10×10 grid at 320), for the
predicted class and the true class:

- **Phone → drinking.** Both maps are hot on the hand holding the phone and on the
  forearm. The model looks at the right place but does not recognise the object: at this
  resolution a phone held at chest height and a cup have the same shape, and the arm's
  gesture weighs more than the object. It is a recognition problem, not an attention one.
- **Raised arm → reaching behind.** The c7 map is hot on the torso and under the arm, the
  c8 map on the head and the hand in the hair. The right cue is there but weighs less than
  the body posture: a shortcut "arm away from the body = c7".
- **c9 → c0 (p022).** The c0 map is on the hands on the wheel. The map for c9, the true
  class, is hot on the window, the car park outside and the image border, not on the head.
  For "talking to the passenger" the model seems to use the background instead of the gaze
  direction: a possible spurious shortcut that does not hold for new people and places.

Caveats: Grad-CAM points to an area, not an object (the grid is 10×10), and the maps are
clues, not proof. They were only inspected on the most confident errors.

## What to do for each error family

| Family | Hypothesis | Planned test |
|---|---|---|
| c9 passenger → c0, background in the c9 map | the model uses the background, not the gaze | background-masking ablation (day 10); later a face crop as an extra input |
| Phone ↔ cup | the object is too small at 320 | phone detector and two-branch model with hand crops (days 6-9) |
| Raised arm → c7 | posture shortcut | hand and head crops in the two-branch model; check whether the problem persists with the ViT |
| Questionable labels (p022 c9 sequence) | transition frames labelled with the class of the sequence | count them in training too; do not fix validation labels, but report an estimate |
| Driver p039 | many but uncertain errors (c9, c0) | per-driver accuracy in grouped cross-validation, to see whether it is an isolated case |

The images used for this analysis (error grid and Grad-CAM maps) stay local
(`testing_ground.ipynb`, `scratch/gradcam/`, both excluded from Git).
