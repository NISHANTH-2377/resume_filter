# Resume Filter

A small project for training and running a resume scoring/filtering model that ranks candidate resumes against job descriptions. The repository contains training and test/inference scripts, model artifacts, and example output.

## Contents

- `train_model.py` — train the resume scoring model.
- `test_model.py` — run inference on a `candidates.jsonl` file and produce a submission CSV.
- `requirements.txt` — Python dependencies.
- `models/` — trained model files and metadata (for example, `resume_filter.json`, `feature_metadata.json`).
- `output/` — outputs from `test_model.py` (e.g., `submission.csv`).
- `JD/` — job description files used for training/evaluation.

## Quickstart

1. Extract the provided dataset archive `datasets_&_candidates.rar` into the project root. After extraction you should have a `datasets/` directory containing training data and a `candidates.jsonl` file for inference.

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Train the model (writes artifacts into `models/`):

```powershell
python train_model.py
```

   (optional and the model is already trained and available in the 'models' directory. the example output is also available in the 'output' directory.)

4. Run inference / generate submission CSV. Before running, replace `datasets/candidates.jsonl` with your candidate file if needed.

```powershell
python test_model.py
```

The produced CSV will be written to the `output/` folder (for example `output/submission.csv`).

## Files & Directories

- `datasets/` (created after extracting `datasets_&_candidates.rar`) — contains training data and `candidates.jsonl` for inference.
- `models/feature_metadata.json` — feature metadata used by the model.
- `models/resume_filter.json` — trained model (already trained and produced by `train_model.py`).
- `output/submission.csv` — example output produced by `test_model.py`.

## Notes & Tips

- Always make sure `datasets/candidates.jsonl` is present before running `test_model.py`.
- If you want to evaluate with a different candidate set, replace `datasets/candidates.jsonl` and run `test_model.py`.
- Inspect `feature_metadata.json` in `models/` if you need to confirm expected features and preprocessing steps.

## Troubleshooting

- Missing dependencies: run `pip install -r requirements.txt`.
- If `train_model.py` or `test_model.py` fail, check that `datasets/` exists and contains the expected files.

## License

This project does not include a license file. Add a license if you plan to publish or share the code.

## Contact

If you need help with the code or want changes to the README, open an issue or contact the maintainer.