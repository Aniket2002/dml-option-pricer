# Migration from the legacy repository

Replace the files in this bundle at the repository root.

Delete the legacy checkpoint before training because it contains only a state dict and no scaler metadata:

```bash
git rm dml_pricer_best.pth
```

Generated CSV files are no longer required by the training pipeline. Remove them from version control if present:

```bash
git rm --cached data/option_data.csv data/train.csv data/val.csv
```

Then run:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest -q --cov=data --cov=models --cov=losses --cov=train --cov-report=term-missing --cov-fail-under=75
python -m train.train_model --samples 5000 --epochs 20 --batch-size 256
streamlit run streamlit_app/app.py
```
