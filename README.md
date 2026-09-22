# CountLoop: Training-Free High-Instance Image Generation via Iterative Agent Guidance

[![Venue: TMLR 2026](https://img.shields.io/badge/Venue-TMLR%202026-blue.svg)](https://openreview.net/forum?id=2JxXGhpCP4)
[![OpenReview](https://img.shields.io/badge/OpenReview-Paper-darkblue.svg)](https://openreview.net/forum?id=2JxXGhpCP4)
[![arXiv](https://img.shields.io/badge/arXiv-2508.16644-b31b1b.svg)](https://arxiv.org/abs/2508.16644)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-yellow.svg)](https://huggingface.co/datasets/anindyamondal/COUNTLOOP)
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://mondalanindya.github.io/CountLoop/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official code repository for **CountLoop**, accepted to **Transactions on Machine Learning Research (TMLR 2026)**.

> **CountLoop: Training-Free High-Instance Image Generation via Iterative Agent Guidance**  
> **Anindya Mondal**<sup>1,*</sup>, **Sauradip Nag**<sup>2,&dagger;</sup>, **Ayan Banerjee**<sup>3,&dagger;</sup>, **Josep Lladós**<sup>3</sup>, **Xiatian Zhu**<sup>1</sup>, **Anjan Dutta**<sup>1,*</sup>  
> <sup>1</sup> University of Surrey, UK | <sup>2</sup> Simon Fraser University, Canada | <sup>3</sup> Computer Vision Center, Universitat Autònoma de Barcelona, Spain  
> <sup>*</sup> Correspondence: `a.mondal@surrey.ac.uk`, `anjan.dutta@surrey.ac.uk` | <sup>&dagger;</sup> Equal Contribution  
> [Project Page](https://mondalanindya.github.io/CountLoop/) | [Paper (OpenReview)](https://openreview.net/forum?id=2JxXGhpCP4) | [arXiv:2508.16644](https://arxiv.org/abs/2508.16644) | [HuggingFace Dataset](https://huggingface.co/datasets/anindyamondal/COUNTLOOP)

---

## 🌟 Overview

Diffusion models excel at photorealistic synthesis but struggle with object count fidelity, especially in dense settings ($N = 30 - 200$). **CountLoop** is a training-free framework achieving structured instance and count control through iterative, agentic feedback:

1. **Design VLM ($\to$ Planning Graph)**: Converts prompts into structured planning graphs $G = (V, E, B_{\text{bg}})$ enforcing anti-grid layouts, area boundaries $[1/100, 1/25]$, minimum separation ($\ge 0.03$), and depth-ordered ($Far \to Near$) instance placement.
2. **Layout-Aligned Attention & Cumulative Composition**: Computes shape-aware masked cross-attention $A^i_{\text{mask}} = A^i_{\text{cross}} \odot \hat{M}_i$ and sequentially updates latents:
   $$F_{i+1}(x,y) = \mathds{1}_{(x, y) \in l_i} \odot A_{\text{mask}}^i + (1 - \mathds{1}_{(x, y) \in l_i}) \odot F_i$$
   with appearance consistency driven by IP-Adapter and final composition pass.
3. **Critic VLM Evaluation**: Evaluates generated images via an open-vocabulary detector (e.g. GroundingDINO) for count $\hat{c}$ and aesthetic scorer for $s_a$:
   $$s_c = \max\left(0, 1 - \frac{|\hat{c} - c_{gt}|}{c_{gt}}\right), \quad S = \alpha \cdot s_c + (1 - \alpha) \cdot s_a$$
   Generates targeted feedback restricted to the closed edit vocabulary: `move | add | remove | resize | degrid`.
4. **Parameter-Free Textual Refinement Operator ($\Psi$)**: Applies gradient-like text updates to $G$ without altering diffusion or VLM weights, stopping when $S \ge \tau = 0.85$ or after $K = 3$ rounds.

---

## 📁 Repository Structure

```
CountLoop/
├── countloop/                     # Core CountLoop engine
│   ├── types.py                   # Pydantic schemas (PlanningGraph, ObjectNode, CriticFeedback, etc.)
│   ├── config.py                  # Paper hyperparameters (alpha=0.6, tau=0.85, K=3, area in [0.01, 0.04])
│   ├── prompts.py                 # System prompts (Design VLM Fig 8, Critic VLM Fig 9, Refiner Psi Sec 6)
│   ├── design_vlm.py              # Design VLM layout planner + anti-grid spatial synthesizer
│   ├── attention.py               # Masking, self-segmentation clustering, Far->Near cumulative composition
│   ├── generator.py               # SDXL / diffusers pipeline with IP-Adapter + lightweight simulation renderer
│   ├── critic.py                  # Critic VLM + Open-Vocabulary Detector + Aesthetic Scorer + scoring formulas
│   ├── refiner.py                 # Parameter-Free Textual Refinement Operator Psi (textual gradient & bounded edits)
│   ├── pipeline.py                # Master orchestrator implementing Algorithm 1
│   ├── cli.py                     # Rich command-line interface (generate, evaluate, augment, serve)
│   ├── serve.py                   # Production FastAPI REST API & Interactive Web UI
│   └── augmentation.py            # FSC-147 data augmentation exporter (points, boxes, exemplar crops)
├── benchmarks/                    # Benchmark definitions & evaluation metrics
│   ├── data.py                    # 92 OmniCount categories & CountLoop-S / CountLoop-M dataset generators
│   └── evaluate.py                # MAE, exact-match, +/-5% and +/-10% tolerance metrics
├── deploy/                        # Production containerization
│   ├── Dockerfile                 # Multi-stage CUDA 12.2 / Ubuntu 22.04 / Python 3.10 image
│   ├── docker-compose.yml         # Container service declaration with GPU reservations
│   └── deploy.sh                  # One-click deploy script
├── tests/                         # Rigorous unit & integration test suite (22 tests)
├── figs/                          # Paper & project figures
├── index.html                     # Project website
├── style.css                      # Project website stylesheet
├── pyproject.toml                 # Package specifications & entry points
├── requirements.txt               # Dependencies
└── WALKTHROUGH.md                 # Detailed implementation walkthrough & verification logs
```

---

## 🚀 Quickstart & Installation

### Option 1: Local Installation with `uv` (Recommended)

```bash
# Clone the repository
git clone https://github.com/mondalanindya/CountLoop.git
cd CountLoop

# Create virtualenv and install package
uv venv
uv pip install -e ".[dev]"
```

### Option 2: Standard pip Installation

```bash
pip install -e ".[dev]"
```

For full GPU acceleration with PyTorch & Diffusers:
```bash
pip install -e ".[gpu]"
```

---

## 💻 CLI Usage

### 1. Synthesize Count-Faithful Images

```bash
# Generate image from prompt with target count 30
countloop generate --prompt "30 cups on a wooden table" --count 30 --output-dir outputs/cups_30

# Dense scene with custom threshold
countloop generate --prompt "60 oranges in a crate" --count 60 --tau 0.85 --max-rounds 3
```

### 2. Evaluate Image & Planning Graph

```bash
countloop evaluate --image outputs/cups_30/final_image.png \
                   --graph outputs/cups_30/planning_graph_final.json \
                   --target-count 30
```

### 3. Generate Synthetic Data Augmentation (FSC-147 Format)

```bash
countloop augment --prompt "30 cups on a wooden table" --count 30 --output-dir dataset/cups
```
Exports:
- `image_001.png`
- `annotation.json` containing points, bounding boxes, and image metadata
- `exemplar_crops/` containing 1–3 exemplar crops of visible instances for training open-world counting models (e.g. CountGD, LOCA).

### 4. Launch Production REST API Server & Web UI

```bash
countloop serve --host 0.0.0.0 --port 8000
```
- **Interactive Web UI**: [`http://localhost:8000/ui`](http://localhost:8000/ui)
- **Interactive OpenAPI / Swagger Docs**: [`http://localhost:8000/docs`](http://localhost:8000/docs)

---

## 🌐 FastAPI REST API Endpoints

Once running, the API provides the following endpoints:
- `GET /health`: Health and GPU status check.
- `GET /ui`: Interactive single-page web application.
- `POST /api/v1/plan`: Generates planning graph $G_0$ without running image synthesis.
- `POST /api/v1/generate`: Full CountLoop agentic loop returning final image (base64), metrics, planning graph, and iteration trajectory.

#### Example API Request:
```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "30 cups on a wooden table",
    "target_count": 30,
    "alpha": 0.6,
    "tau": 0.85,
    "max_rounds": 3
  }'
```

---

## 🐳 Docker Deployment

A production-ready Docker setup based on CUDA 12.2 and Ubuntu 22.04:

```bash
# Build and run container with GPU access
docker compose -f deploy/docker-compose.yml up --build -d

# Verify server status
curl http://localhost:8000/health
```

---

## 🧪 Running Automated Tests

Run the complete test suite:

```bash
pytest tests/ -v
```

All 22 core unit tests verify:
- Schema validation & area constraints
- Design VLM layout planner & anti-grid logic
- Attention masking & cumulative composition math
- Open-vocabulary detector proxy & aesthetic scoring
- Parameter-free textual refiner $\Psi$ mutations
- End-to-end pipeline execution & early stopping
- Data augmentation export
- FastAPI REST endpoints & Web UI

---

## 📚 Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{mondal2026countloop,
  title   = {CountLoop: Training-Free High-Instance Image Generation via Iterative Agent Guidance},
  author  = {Mondal, Anindya and Nag, Sauradip and Banerjee, Ayan and Llados, Josep and Zhu, Xiatian and Dutta, Anjan},
  journal = {Transactions on Machine Learning Research (TMLR)},
  year    = {2026}
}
```
