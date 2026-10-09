"""OVDEval report: average recall per subset, precision and how many boxes models draw, and examples."""

import io
from collections import Counter

from PIL import Image, ImageDraw

from robovlmbench.benchmarks.ovdeval import CEILING_AR, NAME_SUBSETS, SUBSETS, Sample, annotations, load_samples, score
from robovlmbench.formats import POINT_FORMATS
from robovlmbench.report import (COLORS, MASK_COLOR, SPEED_HEADERS, SPEED_NOTE, Run, complete_runs, esc,
                                 jpeg_data_url, page, pct, runs_table, speed_cells)

OFFICIAL_URL = "https://github.com/om-ai-lab/OVDEval"
TITLES = {s: s.capitalize() for s in SUBSETS}
WHAT = {
    "color": "an object's color",
    "material": "what an object is made of",
    "position": "where an object is relative to another",
    "relationship": "what a person is doing with an object",
    "negation": "what an object is not, or doesn't have",
    "celebrity": "famous people, by name",
    "logo": "brand logos",
    "landmark": "famous buildings and monuments",
}


def build(runs: list[Run]) -> str:
    """The report page for every complete run (all photos answered); partial runs are skipped."""
    samples = load_samples()
    complete = complete_runs(runs, samples)
    scores = [score(samples, [run.responses[s.id] for s in samples], POINT_FORMATS[run.info["point_format"]])
              for run in complete]
    truths = _true_descriptions()
    body = "\n".join([
        _intro(samples, truths),
        _scores(samples, complete, scores),
        _precision(complete, scores),
        _examples(samples, complete, scores, truths),
        runs_table(complete),
    ])
    return page("OVDEval Results", body)


def _true_descriptions() -> dict[str, list[str]]:
    """Per sample id, the descriptions of its ground-truth boxes, one per box."""
    truths = {}
    for subset in SUBSETS:
        gt = annotations(subset)
        name = {c["id"]: c["name"] for c in gt["categories"]}
        for a in gt["annotations"]:
            truths.setdefault(f"{subset}/{a['image_id']}", []).append(" ".join(name[a["category_id"]].split()))
    return truths


def _intro(samples: list[Sample], truths: dict[str, list[str]]) -> str:
    items = []
    for subset in SUBSETS:
        group = [s for s in samples if s.subset == subset]
        first = group[0]
        true = sorted(set(truths[first.id]))
        others = [d for d in (" ".join(d.split()) for d in first.descriptions) if d not in true]
        asked = ("every name in the subset" if subset in NAME_SUBSETS
                 else f"{sum(len(s.descriptions) for s in group) / len(group):.1f} descriptions per photo")
        items.append(f'<li><b>{TITLES[subset]}</b> ({len(group)} photos, {asked}): {WHAT[subset]}. '
                     f'<span class="muted">E.g. true: “{esc("”, “".join(true))}”; also asked: '
                     f'“{esc("”, “".join(others[:4]))}”{"…" if len(others) > 4 else ""}</span></li>')
    return f"""<h1>OVDEval</h1>
<p class="muted">Open-vocabulary detection with misleading descriptions: {len(samples)} photos in 8 subsets,
scored with the <a href="{OFFICIAL_URL}">official NMS-AP evaluation</a>.</p>
<h2>What each subset tests</h2>
<p>Each photo comes with a list of descriptions; the model boxes every object matching one and labels the box
with it. Most descriptions are hard negatives that differ from a true one in a single detail, or names of
people, logos or places not in the photo. One request per photo, with all of its descriptions.</p>
<ul>
{"".join(items)}
</ul>"""


def _scores(samples: list[Sample], runs: list[Run], scores: list[dict]) -> str:
    ceiling = sum(CEILING_AR.values()) / len(CEILING_AR)
    rows = [f'<tr><td class="l muted">Ground truth as the answer</td>'
            f'{"".join(f"<td>{pct(CEILING_AR[s])}</td>" for s in SUBSETS)}<td class="key">{pct(ceiling)}</td>'
            f'<td colspan="7"></td></tr>']
    for run, s in zip(runs, scores):
        cells = "".join(f"<td>{pct(s['by_subset'][subset]['ar'])}</td>" for subset in SUBSETS)
        rows.append(f'<tr><td class="l">{esc(run.label)}</td>{cells}<td class="key">{pct(s["ar"])}</td>'
                    f'<td>{pct(s["ap"])}</td><td>{pct(s["precision50"])}</td>'
                    f'{speed_cells(run, samples, s["outcomes"])}</tr>')
    return f"""<h2>Scores</h2>
<div class="scroll"><table>
<tr><th class="l">Model</th>{"".join(f"<th>{TITLES[s]}</th>" for s in SUBSETS)}<th>AR<br>mean</th>
<th>AP<br>mean</th><th>Precision<br>@0.5</th>{SPEED_HEADERS}</tr>
{"".join(rows)}
</table></div>
<p class="note">Average recall (AR, %), the official COCO metric: per description, the share of its
ground-truth boxes found with that description, averaged over IoU thresholds 0.50–0.95; then averaged over
the descriptions in a subset, and over the 8 subsets. Before scoring, the official NMS keeps one box per
ground-truth object whatever its label (here the one written first), so labelling an object with several
descriptions doesn't help. Even the ground truth scores a little below 100: where two true boxes of different
descriptions overlap, that NMS keeps only one. AP: the official NMS-AP, but every box has the same confidence,
so it ranks boxes in photo order and is not comparable to a detector's AP. Precision@0.5: the share of all
boxes drawn that match a true box with the right description at IoU 0.5. {SPEED_NOTE}</p>"""


def _precision(runs: list[Run], scores: list[dict]) -> str:
    if not runs:
        return ""
    rows = []
    for run, s in zip(runs, scores):
        per_sample = s["per_sample"]
        drawn = sum(len(r["boxes"]) + r["dropped"] for r in per_sample)
        dropped = sum(r["dropped"] for r in per_sample)
        found = sum(r["found"] for r in per_sample) / sum(r["n_gt"] for r in per_sample)
        cells = "".join(f"<td>{pct(s['by_subset'][subset]['precision50'])}</td>" for subset in SUBSETS)
        rows.append(f'<tr><td class="l">{esc(run.label)}</td><td>{drawn / len(per_sample):.2f}</td>'
                    f'<td>{pct(dropped / drawn if drawn else None)}</td><td>{pct(found)}</td>{cells}</tr>')
    n_true = sum(r["n_gt"] for r in scores[0]["per_sample"]) / len(scores[0]["per_sample"])
    return f"""<h2>Boxes drawn and precision</h2>
<p>AR doesn't count wrong boxes: a model that also boxes objects matching none of the true descriptions loses
nothing there. This table shows how many boxes each model draws and how many are right.</p>
<div class="scroll"><table>
<tr><th class="l" rowspan="2">Model</th><th rowspan="2">Boxes<br>per photo</th><th rowspan="2">Not<br>scored</th>
<th rowspan="2">True boxes<br>found @0.5</th><th colspan="{len(SUBSETS)}">Precision@0.5</th></tr>
<tr>{"".join(f"<th>{TITLES[s]}</th>" for s in SUBSETS)}</tr>
{"".join(rows)}
</table></div>
<p class="note">There are {n_true:.2f} true boxes per photo. Not scored: % of boxes without a label, with a label
that isn't one of the descriptions asked about, or outside the image. True boxes found: % of ground-truth boxes
matched at IoU 0.5 with the right description, after NMS.</p>"""


def _examples(samples: list[Sample], runs: list[Run], scores: list[dict], truths: dict[str, list[str]]) -> str:
    if not runs:
        return ""
    per_run = [{r["id"]: r for r in s["per_sample"]} for s in scores]
    picks = []
    for subset in SUBSETS:  # one per subset: the first photo where runs find different numbers of boxes
        group = [s for s in samples if s.subset == subset]
        disagree = [s for s in group if len({rows[s.id]["found"] for rows in per_run}) > 1]
        picks.append((disagree or group)[0])
    # a 9th, so the examples fill rows of 3: a second photo from the largest subset
    largest = max(SUBSETS, key=lambda name: sum(s.subset == name for s in samples))
    picks.append(next(s for s in samples if s.subset == largest and s not in picks))

    figures = []
    for sample in picks:
        image = _draw_boxes(sample, [rows[sample.id]["boxes"] for rows in per_run])
        true = sorted(set(truths[sample.id]))
        marks = []
        for i, (run, rows) in enumerate(zip(runs, per_run)):
            row = rows[sample.id]
            labels = Counter(box[4] for box in row["boxes"])
            drawn = ", ".join(f"{esc(label)}{f' ×{n}' if n > 1 else ''}" for label, n in labels.items()) or "none"
            marks.append(f'<li><span class="dot" style="background:{COLORS[i % len(COLORS)]}"></span>'
                         f'{esc(run.label)}: found {row["found"]} of {row["n_gt"]}. '
                         f'<span class="muted">Boxes: {drawn}</span></li>')
        caption = (f'<b>{TITLES[sample.subset]}</b>. True: “{esc("”, “".join(true))}” '
                   f'<span class="muted">(of {len(sample.descriptions)} asked)</span>')
        figures.append(f'<figure><img src="{jpeg_data_url(image)}" alt="{esc(", ".join(true))}">'
                       f'<figcaption>{caption}<ul>{"".join(marks)}</ul></figcaption></figure>')
    return f"""<h2>Examples</h2>
<p>Green: the true boxes. Each run's boxes in its color, whatever their label. One photo per subset, the
first where runs find different numbers of true boxes.</p>
<div class="examples">
{"".join(figures)}
</div>"""


def _draw_boxes(sample: Sample, boxes_per_run: list[list], width: int = 420) -> Image.Image:
    """The photo downscaled to `width`, with the true boxes in green and each run's 0-1 boxes in its color."""
    image = Image.open(io.BytesIO(sample.image)).convert("RGB")
    full_width, full_height = image.size
    image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
    draw = ImageDraw.Draw(image)
    scale = width / full_width
    gt = annotations(sample.subset)
    for a in gt["annotations"]:
        if a["image_id"] == sample.image_id:
            x, y, w, h = (v * scale for v in a["bbox"])
            draw.rectangle((x, y, x + w, y + h), outline=MASK_COLOR, width=4)
    for i, boxes in enumerate(boxes_per_run):
        for x1, y1, x2, y2, _ in boxes:
            draw.rectangle((x1 * (image.width - 1), y1 * (image.height - 1), x2 * (image.width - 1),
                            y2 * (image.height - 1)), outline=COLORS[i % len(COLORS)], width=2)
    return image
