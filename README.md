# Sentiment Analysis

A small, well-organized repository for building, training, and evaluating text sentiment classifiers (positive / negative / neutral). This project includes data preprocessing, model implementations, training and evaluation scripts, and example configurations so you can reproduce experiments or extend the code for your own datasets.

---

## Highlights

- Clear preprocessing pipeline for text data (tokenization, cleaning, splits)
- Config-driven training and evaluation (YAML-based)
- Multiple model options (classical ML and simple neural baselines)
- Checkpointing and result logging for reproducible experiments
- Example scripts and notebooks to get started quickly

---

## Quick start

1. Clone the repo:
   ```bash
   git clone https://github.com/prudhviraj2005/sentiment_analysis.git
   cd sentiment_analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   If `requirements.txt` is not present, typical packages include:
   ```bash
   pip install numpy pandas scikit-learn torch transformers tqdm pyyaml
   ```

3. Prepare your dataset (see Dataset section). Example expected format: `data/<split>.csv` with columns `text` and `label`.

4. Preprocess:
   ```bash
   python scripts/preprocess.py --input data/raw.csv --output data/processed.csv --text-col text --label-col label
   ```

5. Train (example using the default config):
   ```bash
   python train.py --config configs/default.yaml
   ```

6. Evaluate:
   ```bash
   python evaluate.py --model checkpoints/best_model.pt --test data/test.csv --output results/metrics.json
   ```

---

## Project structure

- data/ — raw and processed datasets (CSV or other supported formats)
- scripts/
  - preprocess.py — cleaning, tokenization, and split utilities
  - download_data.py — (optional) helpers to fetch public datasets
- models/ — model definitions (scikit-learn wrappers, PyTorch models, etc.)
- configs/ — YAML configuration files for experiments (hyperparameters, model selection)
- checkpoints/ — saved model weights and experiment artifacts
- results/ — evaluation outputs and logs
- train.py — training entrypoint
- evaluate.py — evaluation entrypoint
- notebooks/ — examples and exploratory analysis

---

## Dataset

Expect a CSV (or TSV) file with at least:
- `text` — the sentence/document
- `label` — sentiment label (e.g., `positive`, `negative`, `neutral` or integer codes)

If using a public dataset (IMDb, SST-2, or others), include download + preprocessing steps in `scripts/` or `notebooks/`. Preprocessing should produce train / val / test splits in `data/`.

Label mapping example (if numeric labels required):
```yaml
label_map:
  positive: 2
  neutral: 1
  negative: 0
```

---

## Configuration

Experiments are controlled via YAML config files in `configs/`. A typical config contains:
- model: which model to use (e.g., `logistic_regression`, `simple_cnn`, `bert`)
- data: paths to train/val/test files and preprocessing options
- training: batch size, learning rate, epochs, checkpoint frequency
- logging: where to write checkpoints and metrics

Example usage:
```bash
python train.py --config configs/bert_base.yaml
```

---

## Models & Evaluation

- Built-in options: Logistic Regression, Naive Bayes, simple feed-forward or CNN text classifiers, and a transformer-based baseline.
- Metrics: accuracy, precision, recall, F1 (per-class and macro), confusion matrix.
- Checkpoints and evaluation outputs are saved to `checkpoints/` and `results/` for reproducibility.

---

## Tips for experiments

- Use a dedicated config per experiment and keep configs under `configs/experiments/`.
- Version your checkpoints and results with descriptive names or timestamps.
- When comparing models, fix random seeds and data splits to ensure comparability.

---

## Contributing

Contributions are welcome. Suggested workflow:
1. Open an issue to discuss larger changes or features.
2. Create a branch for your change: `git checkout -b feature/your-feature`.
3. Add tests or a notebook demonstrating the change when relevant.
4. Submit a pull request describing the change and any testing performed.

Guidelines:
- Keep changes modular (add new models under `models/`)
- Add/modify a config in `configs/` for every new experiment
- Update README or notebooks to document new capabilities

---

## Tests

If tests are present, run them with:
```bash
pytest
```
(Add a `tests/` folder and CI integration for automated checks if desired.)

---

## License

Add a license file (recommended: MIT) and state the license here. Example:
```
MIT License — see LICENSE file
```

---

## Contact

For questions or help, open an issue in this repository or reach out to @prudhviraj2005 on GitHub.
