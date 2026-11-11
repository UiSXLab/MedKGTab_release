# MedKGTab

This repository contains the code for our paper
**MedKGTab: Knowledge-Guided Cross-Domain Feature Expansion for Biomedical Tables**.

MedKGTab infers unobserved features from available data by combining statistical dependencies with biomedical knowledge.
It uses a row-column dual-attention mechanism to model structured tabular data directly, preserving numerical information
without tokenization. MedKGTab further integrates data-driven representations with knowledge from the SPOKE biomedical
knowledge graph to guide feature generation.

This repo includes codes of MedKGTab, six tabular model baselines, and four LLM baselines
with a common CSV input format and prediction metrics.

## Repository Structure

```text
MedKGTab-Lite/
├── run.py               MedKGTab and tabular baseline entry point
├── llm.py               LLM baseline entry point
├── medkgtab/
│   ├── model.py         MedKGTab graph normalization, pooling, and attention bias
│   └── tabpfn/          Shared TabPFN backbone and raw TabPFN baseline
├── baselines/
│   ├── tabular/
│   │   ├── runner.py    XGBoost and TALENT training adapters
│   │   └── talent/      Four deep models, training code, and config.json
│   └── llm/
│       ├── plain.py     DeepSeek and Baichuan paired-example completion
│       ├── curated.py   Curated LLM generation and curation
│       ├── curation.py  Original classifier-based curation
│       ├── tabgen_icl.py TabGen-ICL generation and context selection
│       ├── tabgen_utils.py Original sampling and distribution-distance utilities
│       └── support.py   Shared LLM prompt formatting and response handling
├── common/
│   └── data.py          Shared CSV loading, output, and evaluation
├── requirements.txt
└── README.md
```

## Installation

Python 3.10–3.12 is recommended. Run the following commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For MedKGTab and TabPFN, download the TabPFN v2.5 regression checkpoint from
[Prior Labs](https://huggingface.co/Prior-Labs) and save it as
`tabpfn-v2.5-regressor.ckpt`. Use `--checkpoint` to specify another path or a v2 checkpoint.

## Data Format

Training and test datasets are required as following:

```text
data/X_train.csv     Source features for training records
data/Y_train.csv     Target features for the same training records
data/X_test.csv      Source features for records to complete
data/Y_test.csv      Optional target measurements for evaluation
```

## Running MedKGTab

Run MedKGTab with `--data` pointing to the CSV directory and `--graph` to the feature graph file:

```bash
python run.py --method medkgtab --data data --graph graph.npy --alpha 3
```

Graph rows and columns must follow the order `[X_train columns, Y_train columns]`.

| Argument | Default | Description |
| --- | --- | --- |
| `--graph` | Required | Path to the feature graph `.npy` file. |
| `--checkpoint` | `tabpfn-v2.5-regressor.ckpt` | Path to the TabPFN regression checkpoint. |
| `--alpha` | `3` | Knowledge injection strength. |
| `--device` | `cpu` | Compute device; use `cuda:0` for an NVIDIA GPU. |
| `--seed` | `0` | Random seed. |
| `--output` | `outputs/medkgtab/` | Directory for predictions and metrics. |
| `--normalized-graph` | Off | Use this flag for an already normalized graph. |

## Tabular Model Baselines

The tabular model baselines include TabPFN, XGBoost, FT-Transformer, DANets, SwitchTab, and modernNCA:

```bash
python run.py --method tabpfn --data data
python run.py --method xgboost --data data
python run.py --method ft-transformer --data data
python run.py --method danets --data data
python run.py --method switchtab --data data
python run.py --method modernnca --data data
```

| Argument | Default | Description |
| --- | --- | --- |
| `--epochs` | `600` | Maximum training epochs for deep models. |
| `--batch-size` | `1024` | Training batch size for deep models. |
| `--val-fraction` | `0` | Validation fraction for XGBoost and deep models. |

## LLM Baselines

Supported methods: DeepSeek, Baichuan, Curated LLM, and TabGen-ICL.

LLM baselines use an OpenAI-compatible Chat Completions API.
Set your API key and specify the service URL and model ID:

```bash
export LLM_API_KEY="YOUR_API_KEY"
python llm.py --data data --method deepseek \
  --base-url https://api.deepseek.com/v1 --model deepseek-chat
python llm.py --data data --method baichuan \
  --base-url YOUR_BAICHUAN_OPENAI_COMPATIBLE_URL --model YOUR_M3_PLUS_MODEL_ID
python llm.py --data data --method curated-llm --model deepseek-chat --curation-label label
python llm.py --data data --method tabgen-icl --model deepseek-reasoner --samples 5
```

| Argument | Default | Description |
| --- | --- | --- |
| `--base-url` | `https://api.deepseek.com/v1` | API base URL; also configurable via `LLM_BASE_URL`. |
| `--model` | `YOUR_MODEL_ID` | Model deployment ID; also configurable via `LLM_MODEL`. |
| `--context` | `20` / `1` | Plain LLM context count; TabGen-ICL first-prompt samples per unique target value. |
| `--batch-size` | `1` | Query rows per request for DeepSeek and Baichuan. |
| `--samples` | `5` | Generated samples per query and target for TabGen-ICL. |
| `--max-prompt-iters` | `10` | Maximum generation iterations per query and target for TabGen-ICL. |
| `--curation-label` | `label` | Classification label column for Curated LLM. |
| `--retries-limit` | `3` | Consecutive Curated LLM rounds with no accepted rows before stopping. |
| `--dry-run` | Off | Save a preview of the first request. |

Preview the first request with:

```bash
python llm.py --data data --method deepseek --dry-run
```

## Outputs

Each method writes `predictions.csv` and `metrics.json` to `outputs/<method>/` by default.
Use `--output` to choose another directory.
When `Y_test.csv` is provided, metrics include MSE, R², MedAE, and P95.
MedAE and P95 are the median and 95th percentile of per-sample MAE.
LLM request previews are saved as `request.json`.
Curated LLM and TabGen-ICL also write `generation_log.json`.
Deep tabular models save their per-target checkpoints under the output directory.

## Acknowledgments

This repository uses TabPFN, TALENT, [Curated LLM](https://github.com/seedatnabeel/CLLM),
and TabGen-ICL. TabPFN and TALENT licenses are included under
[TabPFN](medkgtab/tabpfn/LICENSE) and [TALENT](baselines/tabular/talent/LICENSE).
