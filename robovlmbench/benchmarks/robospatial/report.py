"""RoboSpatial-Home report: scores in the leaderboard's format, yes/no bias, and pointing examples."""

import io

from PIL import Image

from robovlmbench.benchmarks.robospatial import _FORMAT_INSTRUCTION, CATEGORIES, Sample, load_samples, score
from robovlmbench.formats import POINT_FORMATS
from robovlmbench.report import (SPEED_HEADERS, SPEED_NOTE, Run, complete_runs, draw_points, esc, jpeg_data_url,
                                 page, pct, pick, runs_table, speed_cells, verdicts)

OFFICIAL_URL = "https://github.com/chanhee-luke/RoboSpatial-Eval"
YES_NO = ("configuration", "compatibility")
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
        _yes_no(samples, complete, scores),
        _examples(samples, complete, scores),
        runs_table(complete),
    ])
    return page("RoboSpatial-Home Results", body)


def _intro(samples: list[Sample]) -> str:
    n = {c: sum(s.category == c for s in samples) for c in CATEGORIES}
    example = {c: esc(next(s for s in samples if s.category == c).question.partition(_FORMAT_INSTRUCTION)[0])
               for c in CATEGORIES}
    return f"""<h1>RoboSpatial-Home</h1>
<p class="muted">Spatial understanding for robots: {len(samples)} questions on photos of home scenes,
scored with the <a href="{OFFICIAL_URL}">official evaluation</a>.</p>
<h2>What each category tests</h2>
<ul>
<li><b>Configuration</b> ({n["configuration"]} yes/no questions): where objects are relative to each other.
  <span class="muted">E.g. “{example["configuration"]}”</span></li>
<li><b>Compatibility</b> ({n["compatibility"]} yes/no questions): whether an object fits in a place.
  <span class="muted">E.g. “{example["compatibility"]}”</span></li>
<li><b>Context</b> ({n["context"]} pointing questions): point to free space described relative to an object;
  correct if one of the first two points lands in the annotated free space.
  <span class="muted">E.g. “{example["context"]}”</span></li>
</ul>"""


def _scores(samples: list[Sample], runs: list[Run], scores: list[dict]) -> str:
    rows = []
    for run, s in zip(runs, scores):
        acc = {c: s["by_category"][c]["accuracy"] for c in CATEGORIES}
        cells = [pct(acc["configuration"]), pct(acc["compatibility"]), pct(s["vqa_avg"]), pct(acc["context"])]
        rows.append(f'<tr><td class="l">{esc(run.label)}</td>{"".join(f"<td>{c}</td>" for c in cells)}'
                    f'<td class="key">{pct(s["total"])}</td>{speed_cells(run, samples, s["outcomes"])}</tr>')

    n = [sum(s.category == c for s in samples) for c in CATEGORIES]
    margin = 100 * 1.96 * (0.25 / min(n)) ** 0.5  # 95% interval half-width at the worst case, 50% accuracy
    return f"""<h2>Scores</h2>
<div class="scroll"><table>
<tr><th class="l">Model</th><th>Configuration</th><th>Compatibility</th><th>VQA avg</th>
<th>Context<br>(pointing)</th><th>Total</th>{SPEED_HEADERS}</tr>
{"".join(rows)}
</table></div>
<p class="note">% correct. VQA avg is the mean of configuration and compatibility; Total is the mean of all
three categories (the leaderboard's definitions). With {min(n)}–{max(n)} questions per category, one
category's score is uncertain by up to ±{margin:.0f} points (95% interval), so smaller differences may be chance.
The 350 questions share 42 photos. {SPEED_NOTE}</p>"""


def _yes_no(samples: list[Sample], runs: list[Run], scores: list[dict]) -> str:
    ids = {c: [s.id for s in samples if s.category == c] for c in YES_NO}
    truth = {s.id: s.answer.lower() == "yes" for s in samples if s.category in YES_NO}
    rows = ['<tr><td class="l">Ground truth</td>' + "".join(
        f"<td>{pct(_mean(truth[i] for i in ids[c]))}</td><td>100.0</td>" for c in YES_NO) + "</tr>"]
    for run, s in zip(runs, scores):
        per_sample = {r["id"]: r for r in s["per_sample"]}
        cells = []
        for c in YES_NO:
            said_yes = _mean(per_sample[i]["said_yes"] for i in ids[c])
            balanced = _mean([_mean(per_sample[i]["correct"] for i in ids[c] if truth[i]),
                              _mean(per_sample[i]["correct"] for i in ids[c] if not truth[i])])
            cells.append(f"<td>{pct(said_yes)}</td><td>{pct(balanced)}</td>")
        rows.append(f'<tr><td class="l">{esc(run.label)}</td>{"".join(cells)}</tr>')

    compat_yes = _mean(truth[i] for i in ids["compatibility"])
    return f"""<h2>Yes/no bias</h2>
<p>Accuracy can hide a lean toward one answer: {pct(compat_yes, 0)}% of compatibility questions have the
answer “yes”, so always answering “yes” scores {pct(compat_yes)} there. Balanced accuracy is the mean of
the accuracy on “yes” questions and on “no” questions: 50 for any constant answer, 100 for a perfect one.</p>
<div class="scroll"><table>
<tr><th class="l" rowspan="2">Model</th><th colspan="2">Configuration</th><th colspan="2">Compatibility</th></tr>
<tr><th>Answers “yes”</th><th>Balanced accuracy</th><th>Answers “yes”</th><th>Balanced accuracy</th></tr>
{"".join(rows)}
</table></div>"""


def _examples(samples: list[Sample], runs: list[Run], scores: list[dict]) -> str:
    if not runs:
        return ""
    per_run = [{r["id"]: r for r in s["per_sample"]} for s in scores]
    context = [s for s in samples if s.category == "context"]
    correct = {s.id: [rows[s.id]["correct"] for rows in per_run] for s in context}
    picks = pick([
        [s for s in context if len(set(correct[s.id])) > 1],  # runs disagree
        [s for s in context if all(correct[s.id])],
        [s for s in context if not any(correct[s.id])],
    ], N_EXAMPLES)

    figures = []
    for sample in picks:
        image = draw_points(Image.open(io.BytesIO(sample.image)), Image.open(io.BytesIO(sample.mask)),
                            [rows[sample.id]["points"] for rows in per_run])
        question = esc(sample.question.partition(_FORMAT_INSTRUCTION)[0])
        figures.append(f'<figure><img src="{jpeg_data_url(image)}" alt="{question}"><figcaption>{question}'
                       f'{verdicts(runs, [rows[sample.id] for rows in per_run])}</figcaption></figure>')
    return f"""<h2>Pointing examples</h2>
<p>Shaded green: the annotated free space. Dots: each run's first two points, the ones scored.
Picked evenly from questions where runs disagree, where all are right, and where all are wrong.</p>
<div class="examples">
{"".join(figures)}
</div>"""


def _mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None
