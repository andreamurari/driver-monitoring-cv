# Annotation guidelines

Bounding boxes on State Farm images for a 4-class detector. Tool: Label Studio running
locally, starting from automatic pre-labels (`scripts/prelabel.py`) corrected by hand.
The repo contains only the selection (`selection.csv`), the Label Studio configuration and
this guide. Neither the images nor the YOLO labels are published: the labels are data
derived from the dataset, which the competition rules do not allow to redistribute. They
are regenerated with `scripts/select_images.py` → `scripts/prelabel.py` → annotation in
Label Studio → `scripts/ls_to_yolo.py`, and live in `annotations/labels/` (excluded from Git).

## Classes

| id | Class | What is annotated |
|---|---|---|
| 0 | `phone` | phone, in the hand, at the ear or resting somewhere |
| 1 | `cup` | glass, mug, can or bottle, in the hand or not |
| 2 | `hand` | every hand of the driver |
| 3 | `face` | the driver's face |

## Rules

- **Annotate everything visible**, even partly. The box covers the visible part, not the
  part guessed behind an obstacle.
- **Hands.** All the driver's hands are annotated, including partly hidden ones (behind the
  steering wheel, behind the body, cut by the image border). The left hand on the wheel is
  annotated even when it is small and in shadow.
- **Hand and object overlapping.** Two separate boxes, one for the hand and one for the
  phone or cup, even when they overlap almost completely.
- **Phone at the ear.** Annotated only if at least part of it is visible; if it is fully
  covered by the hand or the head, only the hand is annotated.
- **Cups and bottles.** Annotated even when not in the hand (cup holder, dashboard): the
  detector has to learn the object, not the gesture.
- **Faces.** Annotated also in profile or three-quarter view, from chin to forehead,
  leaving out hair where it is not needed.
- **People other than the driver.** Not annotated (State Farm images sometimes show the arm
  or leg of whoever sits in the passenger seat).

## First batch: 146 images

| Class | Pre-labels | After correction | What changed most |
|---|---|---|---|
| phone | 57 | 60 | some "phones" were cups (c6); other phones were missing |
| cup | 46 | 67 | many white cups and bottles on a bright background missed by YOLO-COCO; 35 in classes other than c6 (cup holder) |
| hand | 177 | 273 | about 96 hands added, mostly the left hand on the wheel |
| face | 140 | 147 | few missing faces |

No image turned out to be truly ambiguous.

## Limits of the pre-labels

- **Light objects on a light background.** YOLO-COCO often misses white bottles and cups
  against the window or the light plastic of the cabin.
- **Hand far from the camera.** MediaPipe usually finds the right hand, close to the
  camera, and misses the left hand on the wheel, small and in shadow.
- **Differences between drivers.** Hand and face pre-labels were noticeably worse for
  drivers with darker skin and for heavier-set drivers (qualitative observation, not
  measured). It is a known limit of pre-trained models and a concrete risk for a monitoring
  system, which must work equally well for everyone: to be measured on the trained detector
  with per-driver metrics.
