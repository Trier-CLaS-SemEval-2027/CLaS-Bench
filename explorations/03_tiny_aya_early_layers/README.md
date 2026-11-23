# Exploration 03: Early-Layer DiffMean on `CohereLabs/tiny-aya-global`

Runs **DiffMean** language steering on **`CohereLabs/tiny-aya-global`** (`bfloat16`) across early layers (`2, 4, 6, 8, 10, 12`) at $\alpha = 10.0$ on 14 languages (7 High + 7 Low, 70 questions/pair), calling the original `CLaS-Bench/` Python scripts via [`run_tinyaya.sh`](./run_tinyaya.sh).

---

## 1. Quick Start

```bash
pip install "vllm==0.10.2" "transformers<5.0.0" datasets fasttext-numpy2-wheel seaborn matplotlib tqdm

cd explorations/03_tiny_aya_early_layers
python 0-check_tinyaya.py                          # ~30s smoke test (Cohere2 loading & hook)
bash run_tinyaya.sh 2>&1 | tee tinyaya_run.log     # Full pipeline: identify -> generate -> eval
```

Run individual stages *(all stages skip existing outputs)*:
```bash
bash run_tinyaya.sh identify   # Tokenize CulturaX & compute DiffMean centroids (one-time)
bash run_tinyaya.sh generate   # Steered generation for layers 2, 4, 6, 8, 10, 12
bash run_tinyaya.sh eval       # FastText forcing + Qwen3-8B judge + plots (set RUN_JUDGE=0 to skip judge)
```
*(For Colab/Jupyter, use [`run_tinyaya.ipynb`](./run_tinyaya.ipynb).)*

---

## 2. Config

| Parameter | Value |
|:---|:---|
| **Model** | `CohereLabs/tiny-aya-global` (`Cohere2ForCausalLM`, 36 layers, `hidden=2048`, `bfloat16`) |
| **Layers & Strength** | `2, 4, 6, 8, 10, 12` at $\alpha = 10.0$ |
| **High-Resource (7)** | `en`, `de`, `es`, `ar`, `zh`, `ja`, `hi` |
| **Low-Resource (7)** | `cs`, `uk`, `el`, `fa`, `vi`, `sw`, `mt` |
| **Scale** | $14 \times 14 = 196$ pairs $\times$ 70 questions = **13,720 outputs/layer** (`82,320` total) |
| **Evaluation** | FastText forcing (`evaluate_forcing_success.py`) + `Qwen3-8B` relevance (`judge_llm.py`) + plots (`vis_ablate_vector.py`) |

<details>
<summary><b>Why <code>data_tinyaya/</code> is generated fresh instead of reusing <code>data_aya/</code></b></summary>
<br>

`data_aya/` belongs to **`aya-expanse-8b`** (`[32, 4096]`, `255k` vocab) and is incompatible with **`tiny-aya-global`** (`[36, 2048]`, `262k` vocab, different weights/coordinate basis).

</details>

---

## 3. Outputs

- **Centroids (reusable)**: `CLaS-Bench/identification/data_tinyaya/vector.<lang>.tinyaya`
- **Combined Plots & Metrics**: `CLaS-Bench/generation/diffmean_tinyaya/aggregated_results.json`, `*.png`, `*.pdf`
- **Per-Layer Summaries**: `CLaS-Bench/generation/diffmean_tinyaya/layers/*/strength_10_0/analysis_results.json`
- **Run Log**: `explorations/03_tiny_aya_early_layers/tinyaya_run.log`
