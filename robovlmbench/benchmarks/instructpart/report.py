"""InstructPart report: pointing accuracy per variant, and examples."""

import numpy as np
from PIL import Image

from robovlmbench.benchmarks.instructpart import VARIANTS, Sample, load_samples, score
from robovlmbench.formats import POINT_FORMATS
from robovlmbench.report import (SPEED_HEADERS, SPEED_NOTE, Run, complete_runs, draw_points, esc, jpeg_data_url,
                                 page, pct, pick, runs_table, speed_cells, verdicts)

PAPER_URL = "https://arxiv.org/abs/2505.18291"
N_EXAMPLES = 6


def build(runs: list[Run]) -> str:
    """The report page for every complete run (all questions answered); partial runs are skipped."""
    samples = load_samples()
    complete = complete_runs(runs, samples)
    scores = [score(samples, [run.responses[s.id] for s in samples], POINT_FORMATS[run.info["point_format"]])
              for run in complete]
    body = "\n".join([
        _intro(samples),
        _scores(samples, complete, scores),
        _examples(samples, complete, scores),
        runs_table(complete),
    ])
    return page("InstructPart Results", body)


def _intro(samples: list[Sample]) -> str:
    first = {v: next(s for s in samples if s.variant == v) for v in VARIANTS}
    n_photos = len(samples) // len(VARIANTS)
    return f"""<h1>InstructPart</h1>
<p class="muted">Finding the part of an object to act on: {n_photos} photos, one target part each
(<a href="{PAPER_URL}">paper</a>). No official evaluation code is released; this protocol is ours.</p>
<h2>What it tests</h2>
<ul>
<li><b>Task</b> ({n_photos} questions): a task instruction that doesn't name the part, so the model must work out
  which part to use. <span class="muted">E.g. “{esc(first["task"].instruction)} Point to that part in the image.”</span></li>
<li><b>Part</b> ({n_photos} questions): the same photos with the part named.
  <span class="muted">E.g. “Point to the {esc(first["part"].instruction)} in the image.”</span></li>
</ul>
<p>An answer is correct if its first point lands on the part's annotated mask.</p>"""


def _scores(samples: list[Sample], runs: list[Run], scores: list[dict]) -> str:
    rows = []
    for run, s in zip(runs, scores):
        cells = "".join(f"<td>{pct(s['by_variant'][v]['accuracy'])}</td>" for v in VARIANTS)
        rows.append(f'<tr><td class="l">{esc(run.label)}</td>{cells}<td class="key">{pct(s["accuracy"])}</td>'
                    f'{speed_cells(run, samples, s["outcomes"])}</tr>')

    photos = [s for s in samples if s.variant == "task"]
    masks = [np.asarray(Image.open(s.mask_path).convert("L")) > 0 for s in photos]
    random_point = np.mean([m.mean() for m in masks])
    center = np.mean([m[m.shape[0] // 2, m.shape[1] // 2] for m in masks])
    margin = 100 * 1.96 * (0.25 / len(photos)) ** 0.5
    return f"""<h2>Scores</h2>
<div class="scroll"><table>
<tr><th class="l">Model</th><th>Task</th><th>Part</th><th>Overall</th>{SPEED_HEADERS}</tr>
{"".join(rows)}
</table></div>
<p class="note">% of answers whose first point lands on the part. For scale: a point at the image center scores
{pct(center)}, a random point {pct(random_point)} (parts cover a median {pct(np.median([m.mean() for m in masks]))}%
of the photo). With {len(photos)} questions per variant, a score is uncertain by up to ±{margin:.0f} points
(95% interval). {SPEED_NOTE}</p>"""


def _examples(samples: list[Sample], runs: list[Run], scores: list[dict]) -> str:
    if not runs:
        return ""
    per_run = [{r["id"]: r for r in s["per_sample"]} for s in scores]
    task = [s for s in samples if s.variant == "task"]
    correct = {s.id: [rows[s.id]["correct"] for rows in per_run] for s in task}
    picks = pick([
        [s for s in task if len(set(correct[s.id])) > 1],  # runs disagree
        [s for s in task if all(correct[s.id])],
        [s for s in task if not any(correct[s.id])],
    ], N_EXAMPLES)

    figures = []
    for sample in picks:
        image = draw_points(Image.open(sample.image_path), Image.open(sample.mask_path),
                            [rows[sample.id]["points"][:1] for rows in per_run])
        question = esc(sample.instruction)
        figures.append(f'<figure><img src="{jpeg_data_url(image)}" alt="{question}"><figcaption>{question} '
                       f'<span class="muted">({esc(sample.part)} of the {esc(sample.object)})</span>'
                       f'{verdicts(runs, [rows[sample.id] for rows in per_run])}</figcaption></figure>')
    return f"""<h2>Examples (task variant)</h2>
<p>Shaded green: the target part. Dots: each run's first point, the one scored.
Picked evenly from questions where runs disagree, where all are right, and where all are wrong.</p>
<div class="examples">
{"".join(figures)}
</div>"""
