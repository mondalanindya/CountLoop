# CountLoop: Training-Free High-Instance Image Generation via Iterative Agent Guidance

[![TMLR Paper](https://img.shields.io/badge/TMLR-Accepted-success.svg)](https://openreview.net/forum?id=2JxXGhpCP4&invitationId=TMLR/Paper9259/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official deployable implementation of **CountLoop** (Transactions on Machine Learning Research, 2026).

> **CountLoop: Training-Free High-Instance Image Generation via Iterative Agent Guidance**  
> *Anindya Mondal, Sauradip Nag, Ayan Banerjee, Josep Lladós, Xiatian Zhu, Anjan Dutta*  
> [Project Page](https://mondalanindya.github.io/CountLoop/) | [Paper (OpenReview)](https://openreview.net/forum?id=2JxXGhpCP4&invitationId=TMLR/Paper9259/)

> [!IMPORTANT]
> 📢 **Release Status**: *This codebase is currently under preparation and will be fully released following publication.*

---

## 🌟 Overview

Diffusion models excel at photorealistic synthesis but struggle with object count fidelity, especially in dense settings ($N = 30 - 200$). **CountLoop** is a training-free framework achieving structured instance and count control through iterative, agentic feedback:

1. **Design VLM ($\to$ Planning Graph)**: Converts prompts into structured planning graphs $G = (V, E, B_{\text{bg}})$ enforcing anti-grid layouts, area boundaries $[1/100, 1/25]$, minimum separation ($\ge 0.03$), and depth-ordered ($Far \to Near$) instance placement.
2. **Layout-Aligned Attention & Cumulative Composition**:
   - **GLIGEN (Li et al., 2023)**: Injects layout tokens $Q_i = \mathbb{E}(l_i) \in \mathbb{R}^D$ via gated self-attention layers.
   - **Dahary et al. (2024)**: Shapes masks via self-segmentation clustering ($k=2$) restricted to middle and first up-blocks, and assigns clusters via Intersection-over-Minimum ($\text{IoM}(A, B) = \frac{|A \cap B|}{\min(|A|, |B|)}$).
   - **Masked Cross-Attention**: Computes $A^i_{\text{mask}} = A^i_{\text{cross}} \odot \hat{M}_i$.
   - **Cumulative Latent Composition (Eq. 4 & Supp. Eq. 8)**:
     $$F_{i+1}(x,y) = \mathds{1}_{(x, y) \in l_i} \odot A_{\text{mask}}^i + (1 - \mathds{1}_{(x, y) \in l_i}) \odot F_i$$
     accumulating latents Far $\to$ Near to preserve natural occlusion boundaries.
   - **IP-Adapter (Ye et al., 2023)**: Preserves foreground texture across instances ($I_{i+1}, Z_q^{i+1} = \Phi(F_{i+1}, P_d, \theta(I_i))$) with $Z_q$ captured right after $W_Q$.
   - **Final Composition Pass (Supp. Eq. 10)**: Last query latent $Z_q^N$ attends over shared key-values $\mathbb{A}(Z_q^N, K, V)$ conditioned on $P_d + P_{\text{bg}}$.
3. **Critic VLM Evaluation**: Evaluates generated images via an open-vocabulary detector (e.g. GroundingDINO) for count $\hat{c}$ and aesthetic scorer for $s_a$:
   $$s_c = \max\left(0, 1 - \frac{|\hat{c} - c_{gt}|}{c_{gt}}\right), \quad S = \alpha \cdot s_c + (1 - \alpha) \cdot s_a$$
   Generates targeted feedback restricted to the closed edit vocabulary: `move | add | remove | resize | degrid`.
4. **Parameter-Free Textual Refinement Operator ($\Psi$)**: Inspired by TextGrad (Yuksekgonul et al., 2024), treats feedback as a textual gradient updating $G$ without altering diffusion or VLM weights, stopping early when $S \ge \tau = 0.85$ or after $K = 3$ rounds.

---

## 📁 Repository Structure

```
CountLoop/
├── countloop/                     # Core CountLoop engine
│   ├── types.py                   # Pydantic v2 schemas (PlanningGraph, ObjectNode, CriticFeedback, etc.)
│   ├── config.py                  # Paper hyperparameters (alpha=0.6, tau=0.85, K=3, bbox area in [0.01, 0.04])
│   ├── prompts.py                 # Verbatim prompts (Design VLM Fig 8, Critic VLM Fig 9, Refiner Psi Sec 6)
│   ├── design_vlm.py              # Design VLM layout planner + non-grid spatial synthesizer
│   ├── attention.py               # Masking, CountLoopAttentionProcessor, Dahary IoM, cumulative latent update
│   ├── generator.py               # SDXL / diffusers pipeline with IP-Adapter + simulation engine
│   ├── critic.py                  # Critic VLM + GroundingDINO/OWLv2 detector + aesthetic scoring
│   ├── refiner.py                 # Parameter-Free Textual Refinement Operator Psi (textual gradient)
│   ├── pipeline.py                # Master orchestrator implementing Algorithm 1 with early stopping
│   ├── cli.py                     # Rich CLI (generate, evaluate, augment, serve)
│   ├── serve.py                   # Production FastAPI REST API & Interactive Web UI (/ui)
│   └── augmentation.py            # FSC-147 data augmentation exporter (points, boxes, exemplar crops)
├── benchmarks/                    # Benchmark definitions & evaluation metrics
│   ├── data.py                    # 92 OmniCount categories & CountLoop-S / CountLoop-M dataset generators
│   └── evaluate.py                # MAE, exact-match, +/-5% and +/-10% tolerance metrics
├── deploy/                        # Production containerization
│   ├── Dockerfile                 # Multi-stage CUDA 12.2 / Ubuntu 22.04 / Python 3.10 image
│   ├── docker-compose.yml         # Container service declaration with GPU reservations
│   └── deploy.sh                  # One-click deploy script
├── tests/                         # Rigorous unit & integration test suite (24 tests)
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

# Install dependencies using uv
uv venv
uv pip install -e ".[dev]"
```

### Option 2: Standard pip Installation

```bash
pip install -e ".[dev]"
```

For GPU acceleration with PyTorch & Diffusers:
```bash
pip install -e ".[gpu]"
```

---

## 💻 CLI Usage

### Generate Images

```bash
# Generate image from prompt with target count 30
countloop generate --prompt "30 cups on a wooden table" --count 30 --output-dir outputs/cups_30

# High-density scene with custom threshold
countloop generate --prompt "60 oranges in a crate" --count 60 --tau 0.85 --max-rounds 3
```

### Launch Production REST API Server & Web UI

```bash
countloop serve --host 0.0.0.0 --port 8000
```
- Interactive Playground: [`http://localhost:8000/ui`](http://localhost:8000/ui)
- Interactive OpenAPI / Swagger Docs: [`http://localhost:8000/docs`](http://localhost:8000/docs)

### Generate Self-Labeled Data Augmentation (FSC-147 Format)

```bash
countloop augment --prompt "30 cups on a wooden table" --count 30 --output-dir dataset/cups
```
Exports:
- `image_001.png`
- `annotation.json` containing points, bounding boxes, and image metadata
- `exemplar_crops/` containing 1-3 exemplar crops of visible instances for training open-world counting models (e.g. CountGD, LOCA).

---

## 🌐 FastAPI REST API Endpoints

Once the server is running (`http://localhost:8000`), interactive OpenAPI / Swagger docs are available at `http://localhost:8000/docs`.

### Key Endpoints:
- `GET /health`: Health and GPU status check.
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

All core components (schemas, Design VLM layout planner, attention masking equations, Critic scoring formulas, $\Psi$ operator, pipeline orchestrator, and FastAPI endpoints) are rigorously unit-tested.

---

## 📚 Citation

If you find CountLoop useful for your research, please cite:

```bibtex
@article{mondal2026countloop,
  title={CountLoop: Training-Free High-Instance Image Generation via Iterative Agent Guidance},
  author={Mondal, Anindya and Nag, Sauradip and Banerjee, Ayan and Llad{\'o}s, Josep and Zhu, Xiatian and Dutta, Anjan},
  journal={Transactions on Machine Learning Research (TMLR)},
  year={2026}
}
```
