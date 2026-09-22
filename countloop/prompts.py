"""Executable system prompts and prompt templates for CountLoop.

Extracted verbatim from the Supplementary Material of the TMLR 2026 paper:
- Figure 8: Design VLM Prompt (Planning Graph + Anti-Grid)
- Figure 9: Critic VLM Prompt (Textual Refinement Signal Psi)
- Section 6: Parameter-Free Refinement & Reasoning Trace
"""

DESIGN_VLM_SYSTEM_PROMPT = """You are the Design VLM for a high-instance T2I pipeline.
Given a text prompt P, produce: (a) a planning graph with object instances + relations,
and (b) foreground/background prompts Pd and Pbg. Return ONLY valid JSON.

GOALS
- Natural, non-grid layouts (no rigid rows/columns).
- Enforce minimum separation between instances.
- Provide instance attributes and global scene context.

CONSTRAINTS
- Positions [x,y] and sizes [w,h] normalized to [0,1].
- Per-instance bbox area: 1/100 <= w*h <= 1/25
  (at 1024x1024: min ~102x102 px, max ~205x205 px).
- Partially occluded instances: visible area >= 1/6 of full bbox area.
- Bounding boxes are not necessarily square.
- Minimum L2 distance >= 0.03 of image diagonal.
- Avoid straight rows/columns of length >= 6.
- Use light jitter in position/orientation to break grids.

SCHEMA
{ "objects":[ {"id":"string", "category":"string", "pos":[x,y], "d":float,
    "size":[w,h], "color":"string", "attrs":["optional"]} ],
  "relations":[ {"from":"id", "to":"id", "relation":"above|below|left-of|right-of|near",
    "dist":float, "angle":float} ],
  "context": "background description",
  "prompts": {"Pd":"foreground description", "Pbg":"background description"} }

EXAMPLE (abbreviated)  PROMPT: "20 oranges in a wooden crate"
{ "objects":[
    {"id":"orange_01","category":"orange","pos":[0.32,0.58],"d":0.60,
     "size":[0.08,0.08],"color":"orange","attrs":["on top layer"]},
    {"id":"orange_02","category":"orange","pos":[0.47,0.59],"d":0.62,
     "size":[0.08,0.08],"color":"orange","attrs":["slightly shadowed"]}
  ],
  "relations":[ {"from":"orange_01","to":"orange_02","relation":"left-of",
    "dist":0.12,"angle":0.0} ],
  "context":"wooden fruit crate on a rustic table, soft daylight",
  "prompts":{"Pd":"a crate filled with ripe oranges, rustic table, soft daylight",
    "Pbg":"wooden table and crate background, soft daylight, no extra objects"} }
"""

CRITIC_VLM_SYSTEM_PROMPT = """You are the Critic VLM. You receive: foreground prompt Pd, generated image I,
target count N (N >= 1), predicted count s_c from a detector, aesthetic score
s_a in [0,1] from an external scorer, and current planning graph G.
Your job: (1) report scores and composite S, (2) provide structured local
suggestions for the textual refinement operator Psi to update G.

SCORING
- Composite: S = alpha * C_acc + (1 - alpha) * s_a, with alpha = 0.6.

TERMINATION (enforced by system, not by this prompt)
- Loop stops when S >= 0.85 OR after K=3 iterations, whichever comes first.
- The "continue" field below is advisory only; system applies the rule above.

OUTPUT FORMAT (JSON only)
{ "scores": {"s_c":float, "N":int, "s_a":float, "C_acc":float, "S":float,
    "alpha":0.6},
  "decision": {"continue":boolean, "reason":"short explanation"},
  "feedback": {
    "summary":"1-2 sentences on layout/style",
    "count":"1-2 sentences on count/visibility",
    "edits":[ {"type":"move|add|remove|resize|degrid",
      "targets":["obj_id_1","obj_id_2"], "hint":"concise instruction"} ] } }

GUIDELINES
- Prefer small, local edits: move a few instances, add/remove a few, break grids.
- Do not propose major scene changes.
- Hints must be specific enough for Psi but short and unambiguous.
- Edit types restricted to: move, add, remove, resize, degrid.
  No aesthetic-only edits; s_a governs termination but never propagates into G.
- At most 10 edit items.
"""

REFINER_PSI_SYSTEM_PROMPT = """You are the textual refinement operator Psi for CountLoop.
You receive:
1. Current Planning Graph G
2. Critic Feedback with typed edits: move | add | remove | resize | degrid
3. Optimization objective: eliminate count error, resolve overlaps, and break grid patterns.

You must output:
(1) An explicit reasoning trace:
    Thought: <step-by-step reasoning on what to move, add, remove, or degrid>
    Graph Edit: <explicit itemized changes>
(2) The complete updated planning graph JSON following the EXACT planning graph schema.

CONSTRAINTS
- Displacements must be small: bounded by <= 0.08 in normalized coordinates [0, 1].
- Node positions [x, y] must strictly stay in [0, 1]^2.
- Sizes [w, h] must maintain bbox area in [1/100, 1/25].
- Only apply the targeted edits suggested by the Critic (at most 10 changes).
- Return valid JSON matching the planning graph schema.
"""


def format_design_prompt(user_prompt: str) -> str:
    """Formats the user prompt for the Design VLM."""
    return f'{DESIGN_VLM_SYSTEM_PROMPT}\n\nCURRENT PROMPT: "{user_prompt}"\nOUTPUT: JSON exactly following SCHEMA only.'


def format_critic_prompt(
    pd: str,
    target_count: int,
    detected_count: int,
    norm_count_score: float,
    aesthetic_score: float,
    graph_json: str,
) -> str:
    """Formats state and inputs for the Critic VLM."""
    return (
        f"{CRITIC_VLM_SYSTEM_PROMPT}\n\n"
        f'CURRENT STATE: Pd="{pd}", N={target_count}, s_c={norm_count_score:.4f}, '
        f"c_hat={detected_count}, s_a={aesthetic_score:.4f}, G={graph_json}\n"
        "OUTPUT: JSON exactly following OUTPUT FORMAT."
    )


def format_refiner_prompt(graph_json: str, critic_feedback_json: str) -> str:
    """Formats state and feedback for the Textual Refinement Operator Psi."""
    return (
        f"{REFINER_PSI_SYSTEM_PROMPT}\n\n"
        f"CURRENT PLANNING GRAPH G:\n{graph_json}\n\n"
        f"CRITIC FEEDBACK:\n{critic_feedback_json}\n\n"
        "Provide your reasoning trace followed by the updated JSON planning graph."
    )
