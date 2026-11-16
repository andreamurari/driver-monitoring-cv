# Driver monitoring: distraction detection from in-cabin images

Work in progress. Two-stage pipeline on the State Farm Distracted Driver dataset:
object detection (phone, hands, face) and distraction classification, evaluated on
drivers never seen in training.

**Headline numbers (to be filled with measured results only)**

| Macro-F1, unseen drivers | mAP@0.5, detector | CPU latency, final model |
| --- | --- | --- |
| TBD | TBD | TBD |

## Evaluation protocol

All splits are by driver (`subject`), never by image: frames of the same driver are
nearly identical, and a random split leaks information from train to test.
`tests/test_splits.py` checks that no driver appears in two splits.

## Reproduce

```bash
pip install -e ".[dev]"
make data      # needs a Kaggle token and accepted competition rules, see DATA.md
make splits
make eda
make test
```

## Data and licences

See [DATA.md](DATA.md). No dataset content is stored in this repository.
