"""FastAPI REST API Service for CountLoop."""

from __future__ import annotations

import base64
import io
import os
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from countloop.config import CountLoopConfig
from countloop.pipeline import CountLoopPipeline
from countloop.types import GenerationResult, PlanningGraph

app = FastAPI(
    title="CountLoop API",
    description="Training-Free High-Instance Image Generation via Iterative Agent Guidance (TMLR 2026)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
_pipeline: Optional[CountLoopPipeline] = None


def get_pipeline() -> CountLoopPipeline:
    global _pipeline
    if _pipeline is None:
        cfg = CountLoopConfig()
        _pipeline = CountLoopPipeline(cfg)
    return _pipeline


class GenerateRequest(BaseModel):
    prompt: str = Field(..., examples=["30 cups on a wooden table"])
    target_count: Optional[int] = Field(None, examples=[30])
    alpha: float = Field(0.6, ge=0.0, le=1.0)
    tau: float = Field(0.85, ge=0.0, le=1.0)
    max_rounds: int = Field(3, ge=1, le=10)
    seed: int = Field(42)
    mock_mode: bool = Field(False)


class PlanRequest(BaseModel):
    prompt: str = Field(..., examples=["20 oranges in a wooden crate"])
    target_count: Optional[int] = Field(None, examples=[20])


@app.get("/")
def root():
    return {
        "service": "CountLoop API",
        "paper": "CountLoop: Training-Free High-Instance Image Generation via Iterative Agent Guidance (TMLR 2026)",
        "status": "healthy",
        "endpoints": ["/health", "/ui", "/api/v1/generate", "/api/v1/plan", "/docs"],
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "gpu_available": get_pipeline().generator.is_gpu_available}


@app.get("/ui", response_class=HTMLResponse)
def serve_ui():
    """Serves the interactive web interface."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CountLoop Playground (TMLR 2026)</title>
  <style>
    :root { --primary: #2563eb; --bg: #0f172a; --card: #1e293b; --text: #f8fafc; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 24px; }
    .container { max-width: 1200px; margin: 0 auto; }
    header { margin-bottom: 24px; border-bottom: 1px solid #334155; padding-bottom: 16px; }
    h1 { margin: 0 0 8px 0; color: #60a5fa; font-size: 24px; }
    p { margin: 0; color: #94a3b8; font-size: 14px; }
    .grid { display: grid; grid-template-columns: 380px 1fr; gap: 24px; }
    .card { background: var(--card); border-radius: 12px; padding: 20px; border: 1px solid #334155; }
    label { display: block; font-weight: 600; margin-bottom: 6px; font-size: 13px; color: #cbd5e1; }
    input[type="text"], input[type="number"], select { width: 100%; box-sizing: border-box; background: #0f172a; border: 1px solid #475569; color: #f8fafc; padding: 10px 12px; border-radius: 8px; margin-bottom: 16px; font-size: 14px; }
    .row { display: flex; gap: 12px; }
    .row > div { flex: 1; }
    button { width: 100%; background: var(--primary); color: white; border: none; padding: 12px; border-radius: 8px; font-weight: bold; cursor: pointer; font-size: 15px; transition: background 0.2s; }
    button:hover { background: #1d4ed8; }
    button:disabled { background: #475569; cursor: not-allowed; }
    .preview-box { min-height: 480px; display: flex; flex-direction: column; align-items: center; justify-content: center; background: #0f172a; border-radius: 8px; border: 2px dashed #334155; padding: 16px; }
    img { max-width: 100%; max-height: 512px; border-radius: 8px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
    table { width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 13px; }
    th, td { border: 1px solid #334155; padding: 8px 12px; text-align: left; }
    th { background: #0f172a; color: #60a5fa; }
    pre { background: #0f172a; padding: 12px; border-radius: 6px; font-size: 12px; overflow-x: auto; color: #e2e8f0; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>CountLoop: High-Instance Image Generation</h1>
      <p>Training-Free Iterative Agent Guidance (Transactions on Machine Learning Research, 2026)</p>
    </header>
    <div class="grid">
      <div class="card">
        <label>Prompt</label>
        <input type="text" id="prompt" value="30 cups on a wooden table">
        <div class="row">
          <div>
            <label>Target Count N</label>
            <input type="number" id="targetCount" value="30">
          </div>
          <div>
            <label>Quality Threshold (&tau;)</label>
            <input type="number" step="0.05" id="tau" value="0.85">
          </div>
        </div>
        <div class="row">
          <div>
            <label>Count Weight (&alpha;)</label>
            <input type="number" step="0.1" id="alpha" value="0.6">
          </div>
          <div>
            <label>Max Rounds (K)</label>
            <input type="number" id="maxRounds" value="3">
          </div>
        </div>
        <button id="btnGen" onclick="generate()">Run Agentic Loop</button>
        <div id="status" style="margin-top: 14px; font-size: 13px; color: #38bdf8;"></div>
      </div>
      <div class="card">
        <h3 style="margin-top: 0;">Synthesis Result</h3>
        <div class="preview-box" id="previewBox">
          <span style="color: #64748b;">Generated image will appear here</span>
        </div>
        <div id="metricsBox" style="display: none; margin-top: 16px;">
          <h4>Iteration Trajectory</h4>
          <table id="trajTable">
            <thead>
              <tr><th>Round</th><th>Detected</th><th>s_c</th><th>s_a</th><th>Composite S</th><th>Edits</th></tr>
            </thead>
            <tbody id="trajBody"></tbody>
          </table>
          <h4>Reasoning Traces</h4>
          <pre id="traceBox"></pre>
        </div>
      </div>
    </div>
  </div>
  <script>
    async function generate() {
      const btn = document.getElementById('btnGen');
      const status = document.getElementById('status');
      const preview = document.getElementById('previewBox');
      const metrics = document.getElementById('metricsBox');
      const trajBody = document.getElementById('trajBody');
      const traceBox = document.getElementById('traceBox');

      btn.disabled = true;
      status.innerText = "Running Design VLM & Agentic Loop...";
      preview.innerHTML = "<span style='color:#38bdf8;'>Synthesizing & refining instances...</span>";

      try {
        const payload = {
          prompt: document.getElementById('prompt').value,
          target_count: parseInt(document.getElementById('targetCount').value),
          tau: parseFloat(document.getElementById('tau').value),
          alpha: parseFloat(document.getElementById('alpha').value),
          max_rounds: parseInt(document.getElementById('maxRounds').value),
          mock_mode: false
        };
        const res = await fetch('/api/v1/generate', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.final_image_base64) {
          preview.innerHTML = `<img src="data:image/png;base64,${data.final_image_base64}" alt="Result">`;
          metrics.style.display = 'block';
          trajBody.innerHTML = '';
          let traces = '';
          data.history.forEach(r => {
            trajBody.innerHTML += `<tr><td>${r.round}</td><td>${r.detected_count} / ${data.target_count}</td><td>${r.s_c.toFixed(3)}</td><td>${r.s_a.toFixed(3)}</td><td>${r.S.toFixed(3)}</td><td>${r.edits_count}</td></tr>`;
            if (r.reasoning_trace) traces += `Round ${r.round}:\\n${r.reasoning_trace}\\n\\n`;
          });
          traceBox.innerText = traces || 'No textual edits required (converged at round 0).';
          status.innerText = `Finished in ${data.total_elapsed_sec}s! Final count: ${data.final_count}/${data.target_count} (S=${data.composite_score.toFixed(3)})`;
        } else {
          status.innerText = "Error: " + JSON.stringify(data);
        }
      } catch (err) {
        status.innerText = "Execution failed: " + err;
      } finally {
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>"""


@app.post("/api/v1/plan", response_model=Dict[str, Any])
def plan_layout(req: PlanRequest):
    """Generates initial Planning Graph G_0 without generating diffusion images."""
    try:
        pipeline = get_pipeline()
        graph = pipeline.design_vlm.plan_layout(req.prompt, target_count=req.target_count)
        return {
            "prompt": req.prompt,
            "target_count": req.target_count,
            "num_instances": graph.num_instances,
            "graph": graph.model_dump(by_alias=True),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/generate")
def generate_image(req: GenerateRequest):
    """Executes the full CountLoop agentic loop."""
    try:
        task_id = str(uuid.uuid4())[:8]
        out_dir = os.path.join("outputs", f"task_{task_id}")

        cfg = CountLoopConfig(
            alpha=req.alpha,
            tau=req.tau,
            max_rounds=req.max_rounds,
            seed=req.seed,
            use_mock_engine=req.mock_mode,
            output_dir=out_dir,
        )
        pipeline = CountLoopPipeline(config=cfg)

        result: GenerationResult = pipeline.run(
            prompt=req.prompt,
            target_count=req.target_count,
            output_dir=out_dir,
        )

        # Convert final image to base64 for easy API transport
        with open(result.final_image_path, "rb") as img_file:
            b64_image = base64.b64encode(img_file.read()).decode("utf-8")

        return {
            "task_id": task_id,
            "prompt": result.prompt,
            "target_count": result.target_count,
            "final_count": result.final_count,
            "composite_score": result.composite_score,
            "converged": result.converged,
            "iterations_run": result.iterations_run,
            "total_elapsed_sec": result.total_elapsed_sec,
            "final_image_base64": b64_image,
            "planning_graph": result.planning_graph.model_dump(by_alias=True),
            "history": [
                {
                    "round": r.round_idx,
                    "detected_count": r.scores.detected_count,
                    "s_c": r.scores.s_c,
                    "s_a": r.scores.s_a,
                    "S": r.scores.S,
                    "edits_count": len(r.critic_feedback.edits),
                    "reasoning_trace": r.reasoning_trace,
                }
                for r in result.history
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
