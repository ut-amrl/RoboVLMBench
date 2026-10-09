"""EPIC-Bench report: official scores per category and task type, the parts of the score, and examples
with each run's boxes or path."""

import statistics

from PIL import Image, ImageDraw

from robovlmbench.benchmarks.epic_bench import PATH_TASKS, Sample, gt_mask, load_samples, score
from robovlmbench.formats import POINT_FORMATS
from robovlmbench.models.client import Response
from robovlmbench.report import (COLORS, MASK_COLOR, SPEED_HEADERS, SPEED_NOTE, Run, complete_runs, esc,
                                 jpeg_data_url, page, pct, pick, runs_table, speed_cells)

OFFICIAL_URL = "https://github.com/rxc205/EPIC-Bench-Eval"
PAPER_URL = "https://arxiv.org/abs/2605.17070"
CATEGORIES = {"TargetLocalization": "Target localization", "Navigation": "Navigation", "Manipulation": "Manipulation"}
# What each task type's score is made of (the official weights).
SCORED_ON = {
    "GroundDetection": "IoU",
    "FeasiblePath_Ego": "path",
    "FeasiblePath_Exo": "path",
    "VisualMatching": "½ IoU + ½ count",
    "AffordanceRegion": "IoU",
    "ContactRelationship_TypeOne": "IoU + count",
    "ContactRelationship_TypeTwo": "IoU + count",
    "ContactRelationship_TypeThree": "IoU + count",
    "PlacementRegion": "½ IoU + ½ fits?",
}
TASK_NAMES = {
    "FeasiblePath_Ego": "Feasible path, ego view", "FeasiblePath_Exo": "Feasible path, third-person view",
    "ContactRelationship_TypeOne": "Contact: touching the object",
    "ContactRelationship_TypeTwo": "Contact: touching all objects",
    "ContactRelationship_TypeThree": "Contact: touching any object",
}
BASELINE = "Whole image as one box"
N_EXAMPLES = 9


def build(runs: list[Run]) -> str:
    """The report page for every complete run (all questions answered); partial runs are skipped."""
    samples = load_samples()
    complete = complete_runs(runs, samples)
    scores = [score(samples, [run.responses[s.id] for s in samples], POINT_FORMATS[run.info["point_format"]])
              for run in complete]
    baseline = score(samples, [_whole_image(s) for s in samples], POINT_FORMATS["normalized"])
    body = "\n".join([
        _intro(samples),
        _scores(samples, complete, scores, baseline),
        _by_type(samples, complete, scores, baseline),
        _parts(complete, scores, baseline),
        _examples(samples, complete, scores),
        runs_table(complete),
    ])
    return page("EPIC-Bench Results", body)


def _whole_image(sample: Sample) -> Response:
    """The baseline answer: one box over the whole image, one object, "fits"; for paths, a straight
    line from the bottom center of the image up to its center."""
    h, w = sample.size
    if sample.task in PATH_TASKS:
        text = f'{{"points": [({w / 2:.0f}, {h - 1}), ({w / 2:.0f}, {3 * h / 4:.0f}), ({w / 2:.0f}, {h / 2:.0f})]}}'
    else:
        text = (f'{{"number_of_objects": 1, "number_of_instances": 1, "bounding_boxes": [(0, 0, {w}, {h})], '
                f'"placement_feasibility": true}}')
    return Response(sample.id, text, text, None, "stop", 0.0, 0, 0)


def _task_name(task: str) -> str:
    return TASK_NAMES.get(task, "".join(f" {c.lower()}" if c.isupper() else c for c in task).strip().capitalize())


def _descriptions(sample: Sample) -> list[str]:
    """The sample's description lines from the prompt, e.g. "Target Object Description: Balloon"."""
    return [line for line in sample.user.split("\n") if "Description: " in line]


def _intro(samples: list[Sample]) -> str:
    n = {c: sum(s.category == c for s in samples) for c in CATEGORIES}
    first = next(s for s in samples if s.category == "TargetLocalization")
    example = esc(_descriptions(first)[0].partition(": ")[2])
    return f"""<h1>EPIC-Bench</h1>
<p class="muted">Fine-grained visual grounding for robots: {len(samples)} questions in 23 task types
(<a href="{PAPER_URL}">paper</a>), scored with the <a href="{OFFICIAL_URL}">official evaluation</a>.
The model answers with boxes (points for paths), scored against ground-truth masks.</p>
<h2>What each category tests</h2>
<ul>
<li><b>Target localization</b> ({n["TargetLocalization"]} questions, 14 types): box every object matching a
  description and count them; types differ by what the description uses (category, color, material, part,
  spatial relation, orientation, spoken request, ...). <span class="muted">E.g. “{example}”</span></li>
<li><b>Navigation</b> ({n["Navigation"]} questions): box the walkable ground; draw a path of points to a
  marked area (from the camera, or between two areas); find an area marked in one view in a second view and
  count it.</li>
<li><b>Manipulation</b> ({n["Manipulation"]} questions): box the part of a marked object to act on for a task
  (affordance); box and count the objects touching a marked object (or all / any of several);
  box the region described for placing an object and say whether it fits.</li>
</ul>
<p>Each question scores 0–100. Boxes: IoU between the union of the answer's boxes and the union of the
ground-truth masks (an empty answer is right only if there is no target). Count: exact match when the number
matters (as in “two cups”), otherwise partial credit for being close; it weighs 10–50% of the question
(30–50% unless the description states the count), 50% in visual matching. Path: points near the start and
target areas, segments on the walkable path, moving away from the start and toward the target, evenly spaced.</p>"""


def _scores(samples: list[Sample], runs: list[Run], scores: list[dict], baseline: dict) -> str:
    def cells(s: dict) -> str:
        return "".join(f"<td>{pct(s['by_category'][c]['score'])}</td>" for c in CATEGORIES) + \
            f'<td class="key">{pct(s["score"])}</td>'

    rows = [f'<tr><td class="l">{esc(run.label)}</td>{cells(s)}{speed_cells(run, samples, s["outcomes"])}</tr>'
            for run, s in zip(runs, scores)]
    rows.append(f'<tr class="muted"><td class="l">{BASELINE}</td>{cells(baseline)}<td colspan="5"></td></tr>')
    sizes = {c: sum(s.category == c for s in samples) for c in CATEGORIES}
    return f"""<h2>Scores</h2>
<div class="scroll"><table>
<tr><th class="l">Model</th>{"".join(f"<th>{name}</th>" for name in CATEGORIES.values())}<th>Overall</th>
{SPEED_HEADERS}</tr>
{"".join(rows)}
</table></div>
<p class="note">Mean score per question (0–100). Overall is the official headline: the mean over all
{len(samples)} questions, so target localization ({sizes["TargetLocalization"]} questions) weighs
{pct(sizes["TargetLocalization"] / len(samples), 0)}% of it. Boxes can't reach 100: a mask is rarely a
rectangle. {BASELINE}: one box over the whole image, one object, “fits”, and for paths a straight line from the
bottom center up to the image center; it shows what an answer that ignores the question scores. {SPEED_NOTE}</p>"""


def _by_type(samples: list[Sample], runs: list[Run], scores: list[dict], baseline: dict) -> str:
    rows = []
    for category, name in CATEGORIES.items():
        rows.append(f'<tr class="group"><td colspan="{len(runs) + 4}">{name}</td></tr>')
        tasks = dict.fromkeys(s.task for s in samples if s.category == category)
        for task in tasks:
            key = next(s.type_key for s in samples if s.task == task)
            run_cells = "".join(f"<td>{pct(s['by_type'][key]['score'])}</td>" for s in scores)
            rows.append(f'<tr><td class="l">{esc(_task_name(task))}</td><td>{baseline["by_type"][key]["n"]}</td>'
                        f'<td class="l">{SCORED_ON.get(task, "IoU + count")}</td>{run_cells}'
                        f'<td class="muted">{pct(baseline["by_type"][key]["score"])}</td></tr>')
    headers = "".join(f"<th>{esc(run.label)}</th>" for run in runs)
    sizes = [b["n"] for b in baseline["by_type"].values()]
    return f"""<h2>Scores per task type</h2>
<div class="scroll"><table>
<tr><th class="l">Task type</th><th>Questions</th><th class="l">Scored on</th>{headers}<th>{BASELINE}</th></tr>
{"".join(rows)}
</table></div>
<p class="note">With {min(sizes)}–{max(sizes)} questions per type, small types are noisy. Target localization
types differ only in what the description relies on; “human” types describe from the point of view of a person
in the image. A few questions, nearly all “Spoken” requests, have nothing to find: the right answer there is no
boxes and a count of 0.</p>"""


def _parts(runs: list[Run], scores: list[dict], baseline: dict) -> str:
    def mean(s: dict, key: str) -> float | None:
        values = [r[key] for r in s["per_sample"] if key in r]
        return statistics.mean(values) if values else None

    labels = [esc(run.label) for run in runs] + [BASELINE]
    rows = "".join(
        f'<tr{' class="muted"' if label == BASELINE else ""}><td class="l">{label}</td>'
        + "".join(f"<td>{pct(mean(s, key))}</td>" for key in ("localization", "count", "feasible", "path")) + "</tr>"
        for label, s in zip(labels, scores + [baseline]))
    return f"""<h2>Parts of the score</h2>
<div class="scroll"><table>
<tr><th class="l">Model</th><th>Box IoU</th><th>Count</th><th>Fits?<br>(placement)</th><th>Path</th></tr>
{rows}
</table></div>
<p class="note">Mean over the questions each part applies to: box IoU over all box questions, count over
target localization, visual matching and contact, “fits?” (% right) over placement, path over both
feasible-path types.</p>"""


def _draw(image: Image.Image, mask, answers: list[list[list[float]]], path: bool, width: int = 420) -> Image.Image:
    """`image` downscaled to `width`, `mask` shaded, and each run's 0-1 boxes (or path) in its color."""
    image = image.convert("RGB")
    image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
    shade = Image.fromarray(mask.astype("uint8") * 115).resize(image.size, Image.NEAREST)
    image = Image.composite(Image.new("RGB", image.size, MASK_COLOR), image, shade)
    draw = ImageDraw.Draw(image)

    def xy(x: float, y: float) -> tuple[float, float]:  # clamped to the image
        return min(max(x, 0), 1) * (image.width - 1), min(max(y, 0), 1) * (image.height - 1)

    for i, shapes in enumerate(answers):
        color = COLORS[i % len(COLORS)]
        if path:
            points = [xy(*p[:2]) for p in shapes]
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
            for px, py in points:
                draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color, outline="white", width=1)
        else:
            for box in shapes:
                (x1, y1), (x2, y2) = xy(*box[:2]), xy(*box[2:4])
                draw.rectangle((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), outline=color, width=3)
    return image


def _examples(samples: list[Sample], runs: list[Run], scores: list[dict]) -> str:
    if not runs:
        return ""
    per_run = [{r["id"]: r for r in s["per_sample"]} for s in scores]
    picks = pick([[s for s in samples if s.category == c] for c in CATEGORIES], N_EXAMPLES)

    figures = []
    for sample in picks:
        path = sample.task in PATH_TASKS
        rows = [run_rows[sample.id] for run_rows in per_run]
        image = _draw(Image.open(sample.image_paths[-1 if sample.task == "VisualMatching" else 0]), gt_mask(sample),
                      [row["points" if path else "boxes"] for row in rows], path)
        text = "<br>".join(esc(line) for line in _descriptions(sample))
        marks = "".join(
            f'<li><span class="dot" style="background:{COLORS[i % len(COLORS)]}"></span>{esc(run.label)}: '
            f'{pct(row["score"], 0)}{"" if row["outcome"] == "answered" else " (" + row["outcome"] + ")"}</li>'
            for i, (run, row) in enumerate(zip(runs, rows)))
        figures.append(f'<figure><img src="{jpeg_data_url(image)}" alt="{esc(_task_name(sample.task))}">'
                       f'<figcaption><b>{esc(_task_name(sample.task))}</b>'
                       f'{"<br>" + text if text else ""}<ul>{marks}</ul></figcaption></figure>')
    return f"""<h2>Examples</h2>
<p>Shaded green: the ground truth (for paths, the walkable path area). Boxes or lines: each run's answer,
with its score. Visual matching shows the second view, where the answer is. Picked evenly across the three
categories.</p>
<div class="examples">
{"".join(figures)}
</div>"""
