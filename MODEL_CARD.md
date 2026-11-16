# Model card

Work in progress. No model has been trained yet; this file will be completed with
the final results. Sections marked TBD are filled only with measured values.

## Intended use

- Personal study and portfolio project on driver distraction recognition from
  in-cabin images.
- Not a safety system. Not intended for use in a vehicle, for monitoring real
  people, or for any commercial purpose.

## Data

- State Farm Distracted Driver Detection (Kaggle): 22,424 labelled images,
  26 drivers, 10 classes. See [DATA.md](DATA.md) for terms of use.
- Evaluation on drivers never seen in training (`splits.json`: 18 train,
  4 validation, 4 test drivers).

## Known limits

- Staged dataset: the car was towed by a truck and the "drivers" were not actually
  driving. One fixed 2D visible-light camera, daytime, few drivers. No infrared,
  no measured depth, no in-vehicle validation.
- With 4 test drivers the test metrics are noisy.
- No trained weights and no dataset images are published (see DATA.md).

## Models and results

TBD.

## Risks and privacy

The dataset shows real people. No images of them are published in this
repository. Performance across demographic groups has not been measured and
cannot be measured reliably with 26 drivers.
