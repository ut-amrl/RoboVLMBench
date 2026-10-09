"""RoboBench report: RoboBench-MCQ per dimension and subtask, state estimation, answer bias, and examples."""

from collections import Counter

from PIL import Image

from robovlmbench.benchmarks.robobench import MCQ_SUBTASKS, STATE_SUBTASKS, Sample, load_samples, score
from robovlmbench.formats import POINT_FORMATS
from robovlmbench.report import (COLORS, SPEED_HEADERS, SPEED_NOTE, Run, complete_runs, esc, jpeg_data_url, page,
                                 pct, pick, runs_table, speed_cells)

OFFICIAL_URL = "https://github.com/yulin-luo/RoboBench"
PAPER_URL = "https://arxiv.org/abs/2510.17801"
N_EXAMPLES = 6  # per part
LETTERS = "ABCD"
NAMES = {
    "perception_reasoning": "Perception reasoning", "affordance_reasoning": "Affordance reasoning",
    "error_analysis": "Error analysis", "instruction_comprehension": "Instruction comprehension",
    "generalized_planning": "Generalized planning",
}


def build(runs: list[Run]) -> str:
    """The report page for every complete run (all questions answered); partial runs are skipped."""
    samples = load_samples()
    complete = complete_runs(runs, samples)
    scores = [score(samples, [run.responses[s.id] for s in samples], POINT_FORMATS[run.info["point_format"]])
              for run in complete]
    body = "\n".join([
        _intro(samples),
        _scores(samples, complete, scores),
        _subtasks(samples, complete, scores),
        _bias(samples, complete, scores),
        _examples(samples, complete, scores),
        runs_table(complete),
    ])
    return page("RoboBench Results", body)


def _intro(samples: list[Sample]) -> str:
    n = Counter(s.kind for s in samples)
    return f"""<h1>RoboBench</h1>
<p class="muted">Robot “system-2” reasoning (<a href="{PAPER_URL}">paper</a>): perception, affordances, error
analysis and planning, on photos and robot videos. The {n["mcq"] + n["state"]:,} questions of its
{6092:,} with deterministic scoring, asked and scored as in the <a href="{OFFICIAL_URL}">official evaluation</a>.</p>
<h2>What is included</h2>
<ul>
<li><b>RoboBench-MCQ</b> ({n["mcq"]:,} multiple-choice questions, A–D; the official setting of that name):
  perception reasoning, affordance reasoning and error analysis, 13 subtasks.</li>
<li><b>State estimation</b> ({n["state"]:,} yes/no questions, Planning Q3): given a robot video up to now
  and the plan's steps so far, is the step finished? 10 planning subtasks.</li>
<li><b>Not included</b>: the other 3,005 planning questions, where the model writes a multi-step plan (Q1,
  2,163, incl. navigation) or names the next step (Q2, 842). The official evaluation scores those only with
  an LLM judge (GPT-4o), which this harness does not call.</li>
</ul>
<p>Prompts are the official ones: task definitions from the released system prompts, the question, the
options, and “Please answer only one word, e.g., A, B, C, or D.” (state questions: “Please only answer one
word, yes or no.”, after the robot's skill list and worked examples). Every image is resized to 512×512 and
sent as JPEG, as the official code does. Videos are the released frames, in order: 10–34 per multiple-choice
video question, 1–20 per state question. As in the official code, the text comes first, then the images.</p>
<p>Scoring: an answer that is only a letter (or letters) is read by the official rule; it must match the
answer exactly. The official code asks GPT-4o to pick the letter out of any other answer, and to pick
“yes” or “no” out of every state answer; here fixed rules do that (e.g. “B. Exo”, “The answer is B”,
“Yes, it is finished.”), and answers they can't read count as wrong.</p>"""


def _scores(samples: list[Sample], runs: list[Run], scores: list[dict]) -> str:
    rows = []
    for run, s in zip(runs, scores):
        dims = [pct(s["mcq"]["by_dimension"].get(d, {}).get("accuracy")) for d in MCQ_SUBTASKS]
        rows.append(f'<tr><td class="l">{esc(run.label)}</td>{"".join(f"<td>{c}</td>" for c in dims)}'
                    f'<td class="key">{pct(s["mcq"]["accuracy"])}</td><td>{pct(s["state"]["accuracy"])}</td>'
                    f'{speed_cells(run, samples, s["outcomes"])}</tr>')
    n_mcq = sum(s.kind == "mcq" for s in samples)
    return f"""<h2>Scores</h2>
<div class="scroll"><table>
<tr><th class="l">Model</th>{"".join(f"<th>{NAMES[d]}</th>" for d in MCQ_SUBTASKS)}
<th>RoboBench-MCQ</th><th>State estimation<br>(yes/no)</th>{SPEED_HEADERS}</tr>
{"".join(rows)}
</table></div>
<p class="note">% correct. Each score is the mean of its subtasks' accuracies (the official scores are per
subtask; its aggregator averages over tasks). RoboBench-MCQ: the 13 multiple-choice subtasks; chance is 25.
State estimation: the 10 subtasks of yes/no questions; chance is 50. With {n_mcq:,} multiple-choice questions,
RoboBench-MCQ is uncertain by about ±2 points (95% interval); single subtasks (82–367 questions) by up to ±11.
{SPEED_NOTE}</p>"""


def _subtasks(samples: list[Sample], runs: list[Run], scores: list[dict]) -> str:
    head = "".join(f'<th><span class="dot" style="background:{COLORS[i % len(COLORS)]}"></span>'
                   f"{esc(run.label)}</th>" for i, run in enumerate(runs))
    rows = []
    for kind, subtasks in (("mcq", MCQ_SUBTASKS), ("state", STATE_SUBTASKS)):
        for dimension, names in subtasks.items():
            label = NAMES[dimension] + (" (multiple choice)" if kind == "mcq" else " (state estimation, yes/no)")
            rows.append(f'<tr class="group"><td colspan="{4 + len(runs)}">{label}</td></tr>')
            for name in names:
                group = [s for s in samples if s.subtask == name and s.kind == kind]
                frames = [len(s.image_paths) for s in group]
                span = str(min(frames)) if min(frames) == max(frames) else f"{min(frames)}–{max(frames)}"
                media = ("image" if max(frames) == 1
                         else f"{span} frames" if group[0].input_type == "video" else f"{span} images")
                common, count = Counter(s.answer for s in group).most_common(1)[0]
                cells = "".join(f'<td>{pct(s[kind]["by_subtask"][name]["accuracy"])}</td>' for s in scores)
                rows.append(f'<tr><td class="l">{esc(name.replace("_", " "))}</td><td>{len(group)}</td>'
                            f'<td class="l">{media}</td><td>{pct(count / len(group))} '
                            f'<span class="muted">({esc(common)})</span></td>{cells}</tr>')
    return f"""<h2>Scores per subtask</h2>
<div class="scroll"><table>
<tr><th class="l">Subtask</th><th>Questions</th><th class="l">Input</th><th>Always the most<br>common answer</th>
{head}</tr>
{"".join(rows)}
</table></div>
<p class="note">% correct. “Always the most common answer” is what a constant answer would score, a floor
to compare against: answers are not balanced in every subtask (in navigation visual prompt, 57% are A).</p>"""


def _bias(samples: list[Sample], runs: list[Run], scores: list[dict]) -> str:
    mcq = [s for s in samples if s.kind == "mcq"]
    state = [s for s in samples if s.kind == "state"]
    truth = Counter(s.answer for s in mcq)
    rows = ['<tr><td class="l">Ground truth</td>'
            + "".join(f"<td>{pct(truth[c] / len(mcq))}</td>" for c in LETTERS)
            + f'<td>{pct(sum(s.answer == "yes" for s in state) / len(state))}</td><td>—</td><td>—</td></tr>']
    for run, s in zip(runs, scores):
        chosen = Counter(r["parsed"] for r in s["per_sample"] if r["kind"] == "mcq")
        said = [r["parsed"] for r in s["per_sample"] if r["kind"] == "state"]
        fallback = s["mcq"]["parsed_by"]["fallback"] + s["state"]["parsed_by"]["fallback"]
        unread = s["mcq"]["parsed_by"]["unreadable"] + s["state"]["parsed_by"]["unreadable"]
        rows.append(f'<tr><td class="l">{esc(run.label)}</td>'
                    + "".join(f"<td>{pct(chosen[c] / len(mcq))}</td>" for c in LETTERS)
                    + f'<td>{pct(said.count("yes") / len(said))}</td><td>{fallback}</td><td>{unread}</td></tr>')
    return f"""<h2>Answer bias and parsing</h2>
<p>How often each answer is given, against how often it is right. A strong lean toward one letter, or toward
“yes”, inflates scores where that answer is common.</p>
<div class="scroll"><table>
<tr><th class="l" rowspan="2">Model</th><th colspan="4">Multiple choice: % answering</th>
<th>State</th><th colspan="2">Answers read by</th></tr>
<tr><th>A</th><th>B</th><th>C</th><th>D</th><th>% “yes”</th><th>fallback rule</th><th>none (wrong)</th></tr>
{"".join(rows)}
</table></div>
<p class="note">Fallback rule: answers that were not just a letter or just yes/no, read by our rules where the
official code would ask GPT-4o; if this count is large, scores depend on those rules. Multiple-choice shares
don't add to 100 when answers are unreadable or name several letters.</p>"""


def _examples(samples: list[Sample], runs: list[Run], scores: list[dict]) -> str:
    if not runs:
        return ""
    per_run = [{r["id"]: r for r in s["per_sample"]} for s in scores]
    correct = {s.id: [rows[s.id]["correct"] for rows in per_run] for s in samples}
    sections = []
    for kind, title in (("mcq", "Multiple choice"), ("state", "State estimation")):
        of_kind = [s for s in samples if s.kind == kind]
        picks = pick([
            [s for s in of_kind if len(set(correct[s.id])) > 1],  # runs disagree
            [s for s in of_kind if all(correct[s.id])],
            [s for s in of_kind if not any(correct[s.id])],
        ], N_EXAMPLES)
        sections.append(f'<h3>{title}</h3>\n<div class="examples">\n{"".join(_figure(s, runs, per_run) for s in picks)}\n</div>')
    return f"""<h2>Examples</h2>
<p>For each part, picked evenly from questions where runs disagree, where all are right, and where all are
wrong. Videos show up to 4 frames, evenly spaced. State questions are shown without the skill list and worked
examples that precede them in the prompt.</p>
{"".join(sections)}"""


def _figure(sample: Sample, runs: list[Run], per_run: list[dict]) -> str:
    options = "".join(f"<br>{LETTERS[i]}. {esc(o)}" for i, o in enumerate(sample.options))
    marks = []
    for i, (run, rows) in enumerate(zip(runs, per_run)):
        row = rows[sample.id]
        verdict = '<span class="ok">✓</span>' if row["correct"] else '<span class="bad">✗</span>'
        marks.append(f'<li><span class="dot" style="background:{COLORS[i % len(COLORS)]}"></span>'
                     f'{esc(run.label)}: {esc(row["parsed"] or "(unreadable)")} {verdict}</li>')
    n = len(sample.image_paths)
    shown = ("image" if n == 1 else f"{n} images" if sample.input_type == "image"
             else f"{n} frames" if n <= 4 else f"4 of {n} frames")
    return (f'<figure><img src="{jpeg_data_url(_thumbnail(sample))}" alt="{esc(sample.subtask)}">'
            f'<figcaption><b>{esc(sample.subtask.replace("_", " "))}</b> <span class="muted">({shown})</span><br>'
            f'{esc(sample.question)}{options}<br>Answer: <b>{esc(sample.answer)}</b><ul>{"".join(marks)}</ul>'
            f'</figcaption></figure>')


def _thumbnail(sample: Sample, width: int = 420) -> Image.Image:
    """The image, or up to 4 evenly spaced video frames in a 2x2 grid, `width` pixels wide."""
    paths = sample.image_paths
    if len(paths) > 4:
        paths = [paths[round(i * (len(paths) - 1) / 3)] for i in range(4)]
    frames = [Image.open(p).convert("RGB") for p in paths]
    cell = width if len(frames) == 1 else width // 2
    frames = [f.resize((cell, round(f.height * cell / f.width)), Image.LANCZOS) for f in frames]
    if len(frames) == 1:
        return frames[0]
    h = max(f.height for f in frames)
    grid = Image.new("RGB", (width, h * ((len(frames) + 1) // 2)), "white")
    for i, f in enumerate(frames):
        grid.paste(f, ((i % 2) * cell, (i // 2) * h))
    return grid
