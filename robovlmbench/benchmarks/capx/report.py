"""CaP-X report: task success per suite and config, why episodes failed, time per model call and per episode,
and example episodes (the last camera image and the code the model wrote)."""

import json
import statistics

from PIL import Image

from robovlmbench import runner
from robovlmbench.benchmarks.capx import (OUTCOMES, Episode, episode_order, load_configs, load_episodes,
                                          load_protocol, outcome, score, support_records)
from robovlmbench.report import COLORS, Run, esc, jpeg_data_url, page, pct, pick, seconds

OFFICIAL_URL = "https://github.com/capgym/cap-x"
SUITE_NAMES = {"robosuite": "Robosuite", "libero": "LIBERO-PRO"}
VARIANT_NOTES = {
    "privileged": "single turn; the API reads object poses from the simulator (no perception)",
    "multiturn_privileged": "as privileged, but after its code runs the model sees the console output and "
                            "may write new code, up to 10 times",
    "multiturn": "multi-turn; the API perceives objects from camera images (SAM3, Contact-GraspNet)",
    "default": "single turn; the API perceives objects from camera images (SAM3, Contact-GraspNet)",
}
PERTURBATIONS = {"swap": "objects in other positions", "task": "a different task in the same scene",
                 "object": "objects with another appearance", "lan": "the instruction reworded"}
OUTCOME_WORDS = {"success": "task done", "incomplete": "code ran, task not done", "code_error": "code raised an error",
                 "no_code": "no code", "timeout": "timed out"}
N_EXAMPLES = 3


def build(runs: list[Run]) -> str:
    """The report page for every complete run (every episode of the protocol done, in the suites that can run
    on its hardware; the others show as N/A); partial runs are skipped."""
    protocol = load_protocol()
    episodes = episode_order(load_configs(protocol))
    complete, done, unavailable = [], [], []
    for run in runs:
        out_dir = runner.OUTPUTS_DIR / run.name / run.engine / "capx"
        na_path = out_dir / "unavailable.json"
        run_na = json.loads(na_path.read_text()) if na_path.exists() else {}
        run_episodes = [e for e in episodes if e.config.suite not in run_na]
        run_done = load_episodes(out_dir)
        missing = sum(e.id not in run_done for e in run_episodes)
        if missing:
            print(f"skipping {run.name} ({run.engine}): {missing} of {len(run_episodes)} episodes not done")
        else:
            complete.append(run)
            done.append(run_done)
            unavailable.append(run_na)
    scores = [score([e for e in episodes if e.config.suite not in na], d) for na, d in zip(unavailable, done)]
    body = "\n".join([
        _intro(protocol, episodes),
        _scores(protocol, episodes, complete, scores, unavailable),
        _hardware(complete, unavailable),
        _per_config(episodes, complete, scores, unavailable),
        _outcomes(complete, scores),
        _examples(episodes, complete, done),
        _runs_table(complete),
    ])
    return page("CaP-X Results", body)


def _intro(protocol: dict, episodes: list[Episode]) -> str:
    configs = {e.config.id: e.config for e in episodes}.values()
    items = []
    for suite in SUITE_NAMES:
        suite_configs = [c for c in configs if c.suite == suite]
        if not suite_configs:
            continue
        variants = ", ".join(dict.fromkeys(c.variant for c in suite_configs))
        if suite == "robosuite":
            tasks = ", ".join(t.replace("_", " ") for t in protocol["robosuite"]["tasks"])
            what = f"{len(protocol['robosuite']['tasks'])} tasks ({tasks})"
        else:
            suites = protocol["libero"]["suites"]
            perturbed = sorted({s.rpartition("_")[2] for s in suites if s.rpartition("_")[2] in PERTURBATIONS})
            what = (f"{len(suites)} suites × {len(protocol['libero']['task_ids'])} tasks ({', '.join(suites)}); "
                    + "; ".join(f"<i>{p}</i>: {PERTURBATIONS[p]}" for p in perturbed))
        items.append(f"<li><b>{SUITE_NAMES[suite]}</b>: {what}. Variants: {esc(variants)}. "
                     f"{protocol[suite]['trials']} trials per task.</li>")
    variants = dict.fromkeys(c.variant for c in configs)
    notes = "".join(f"<li><b>{esc(v)}</b>: {VARIANT_NOTES.get(v, 'see CaP-X config')}</li>" for v in variants)
    return f"""<h1>CaP-X</h1>
<p class="muted">Code as policy: the model writes Python that drives a simulated robot arm through
<a href="{OFFICIAL_URL}">CaP-X</a>'s perception and control functions; an episode succeeds if CaP-X's check finds
the task done after the code has run. {len(episodes)} episodes per model.</p>
<h2>What is run</h2>
<ul>{"".join(items)}</ul>
<p>Variants (which CaP-X config: API and interaction):</p>
<ul>{notes}</ul>
<p class="note">Each trial starts from its own initial state (Robosuite: seed = trial number; LIBERO: the task's
n-th initial state). Prompts, code extraction, execution and success checks are CaP-X's own; the model is queried
with its own request settings (see Runs).</p>"""


def _na(reason: str) -> str:
    return f'<td><span title="{esc(reason)}">N/A</span></td>'


def _scores(protocol: dict, episodes: list[Episode], runs: list[Run], scores: list[dict],
            unavailable: list[dict]) -> str:
    groups = list(dict.fromkeys(e.config.suite for e in episodes))
    variants = list(dict.fromkeys(f"{e.config.suite}/{e.config.variant}" for e in episodes))
    show_variants = len(variants) > len(groups)
    rows = []
    for run, s, na in zip(runs, scores, unavailable):
        def cell(key, suite, value):
            return _na(f"{SUITE_NAMES[suite]} can't run on this hardware: {na[suite]}") if suite in na else (
                f"<td>{pct(value(key))}</td>")
        cells = [cell(g, g, lambda k: s["by_suite"][k]["success_rate"]) for g in groups]
        if show_variants:
            cells += [cell(v, v.split("/")[0], lambda k: s["by_variant"][k]["success_rate"]) for v in variants]
        overall = (_na("some suites can't run on this hardware") if na
                   else f'<td class="key">{pct(s["success_rate"])}</td>')
        timing_call, timing_episode = s["time_per_call_s"] or {}, s["time_per_episode_s"] or {}
        unusable = {"cut off": s["calls"]["cut_off"], "prompt too long": s["calls"]["prompt_too_long"]}
        per_episode, tokens = s["calls"]["per_episode_median"], s["calls"]["output_tokens_median"]
        rows.append(
            f'<tr><td class="l">{esc(run.label)}</td>{overall}' + "".join(cells)
            + f'<td>{seconds(timing_call.get("median"))}</td><td>{seconds(timing_call.get("p90"))}</td>'
            f'<td>{seconds(timing_episode.get("median"))}</td><td>{"—" if per_episode is None else f"{per_episode:g}"}</td>'
            f'<td class="l">{esc(run.hardware)}</td><td>{"—" if tokens is None else f"{tokens:.0f}"}</td>'
            f'<td><span title="{esc(", ".join(f"{v} {k}" for k, v in unusable.items()))}">'
            f'{sum(unusable.values())}</span></td></tr>')
    heads = "".join(f"<th>{SUITE_NAMES[g]}</th>" for g in groups)
    if show_variants:
        heads += "".join(f"<th>{esc(SUITE_NAMES[v.split('/')[0]])}<br>{esc(v.split('/')[1])}</th>" for v in variants)
    n = {g: protocol[g]["trials"] for g in groups}
    margin = 100 * 1.96 * (0.25 / min(n.values())) ** 0.5 if n else 0
    return f"""<h2>Scores</h2>
<div class="scroll"><table>
<tr><th class="l">Model</th><th>Overall</th>{heads}<th>Time per call<br>median</th><th>Time per call<br>90th pct</th>
<th>Time per episode<br>median</th><th>Calls per<br>episode</th><th class="l">Hardware</th>
<th>Output tokens<br>per call</th><th>Unusable<br>answers</th></tr>
{"".join(rows)}
</table></div>
<p class="note">% of episodes where the task got done, averaged over configs (task × variant) so every config
counts the same: Overall over all configs, then per suite{" and per variant" if show_variants else ""}. One config's
rate, from {min(n.values()) if n else 0}–{max(n.values()) if n else 0} trials, is uncertain by up to ±{margin:.0f}
points (95% interval); averages over many configs are tighter.
Time per call: from sending a model call to receiving the whole answer (reasoning included), calls sent one at a
time with every image as new. Time per episode: an episode's simulation, code execution and model calls, without
time spent waiting for other episodes' calls ({runs[0].info["capx"]["num_workers"] if runs else "—"} simulators
run at once). Output tokens: median per call, reasoning included. Unusable answers: calls cut off at the token
limit, or whose prompt was longer than the server's context (hover for which); they give no code.
N/A: the suite can't run on the run's hardware (hover for why; see Hardware); Overall is then N/A too.</p>"""


def _hardware(runs: list[Run], unavailable: list[dict]) -> str:
    """Which suites run on which GPU type: from the CaP-X checks of scripts/check_env.py, and the runs."""
    status = {}  # GPU name -> suite -> cell
    for record in support_records().values():
        for suite, check in record["suites"].items():
            status.setdefault(record["gpu"].replace("NVIDIA ", ""), {})[suite] = (
                f'<td class="l"><span class="ok">✓</span> check passed {check["date"]}</td>' if check["ok"]
                else f'<td class="l">N/A: {esc(check["detail"])}</td>')
    for run, na in zip(runs, unavailable):
        gpu = run.hardware.partition("× ")[2] or run.hardware
        for suite, why in na.items():
            status.setdefault(gpu, {}).setdefault(suite, f'<td class="l">N/A: {esc(why)}</td>')
    if not status:
        return ""
    rows = "".join(
        f'<tr><td class="l">{esc(gpu)}</td>'
        + "".join(suites.get(suite, '<td class="l muted">not checked</td>') for suite in SUITE_NAMES) + "</tr>"
        for gpu, suites in sorted(status.items()))
    return f"""<h2>Hardware</h2>
<div class="scroll"><table>
<tr><th class="l">GPU</th>{"".join(f'<th class="l">{name}</th>' for name in SUITE_NAMES.values())}</tr>
{rows}
</table></div>
<p class="note">Whether each CaP-X suite runs on a GPU type: a CaP-X check of <code>scripts/check_env.py</code>
(a scripted episode with headless rendering) passed, or why it can't (no CaP-X env for that hardware, or the
check failed). Runs on a GPU type skip the suites marked N/A.</p>"""


def _per_config(episodes: list[Episode], runs: list[Run], scores: list[dict], unavailable: list[dict]) -> str:
    if not runs:
        return ""
    configs = list({e.config.id: e.config for e in episodes}.values())
    suite_of = {c.id: c.suite for c in configs}
    head = "".join(f'<th><span class="dot" style="background:{COLORS[i % len(COLORS)]}"></span>{esc(r.label)}</th>'
                   for i, r in enumerate(runs))

    def row(label, ids, cls=""):
        cells = []
        for s, na in zip(scores, unavailable):
            if suite_of[ids[0]] in na:
                cells.append(_na(na[suite_of[ids[0]]]))
            else:
                cells.append(f"<td>{pct(statistics.mean(s['by_config'][c]['success_rate'] for c in ids))}</td>")
        return f'<tr{cls}><td class="l">{label}</td>{"".join(cells)}</tr>'

    rows, details = [], []
    for suite in SUITE_NAMES:
        suite_configs = [c for c in configs if c.suite == suite]
        if not suite_configs:
            continue
        rows.append(f'<tr class="group"><td colspan="{len(runs) + 1}">{SUITE_NAMES[suite]}</td></tr>')
        if suite == "robosuite":
            rows += [row(esc(c.id), [c.id]) for c in suite_configs]
        else:  # one row per LIBERO suite and variant; per task below
            groups = {}
            for c in suite_configs:
                groups.setdefault(f"{c.task.split('/')[0]}/{c.variant}", []).append(c.id)
            rows += [row(f"{esc(g)} <span class='muted'>({len(ids)} tasks)</span>", ids) for g, ids in groups.items()]
            details += [row(esc(c.id), [c.id]) for c in suite_configs]
    more = ""
    if details:
        more = f"""<details><summary>Every LIBERO-PRO task</summary>
<div class="scroll"><table><tr><th class="l">Config</th>{head}</tr>{"".join(details)}</table></div></details>"""
    return f"""<h2>Per task</h2>
<div class="scroll"><table>
<tr><th class="l">Config</th>{head}</tr>
{"".join(rows)}
</table></div>
{more}"""


def _outcomes(runs: list[Run], scores: list[dict]) -> str:
    if not runs:
        return ""
    rows = "".join(
        f'<tr><td class="l">{esc(run.label)}</td>'
        + "".join(f'<td>{pct(s["outcomes"][o] / s["n"] if s["n"] else None)}</td>' for o in OUTCOMES) + "</tr>"
        for run, s in zip(runs, scores))
    return f"""<h2>How episodes ended</h2>
<div class="scroll"><table>
<tr><th class="l">Model</th><th>Task done</th><th>Code ran,<br>task not done</th><th>Code raised<br>an error</th>
<th>No code</th><th>Timed out</th></tr>
{rows}
</table></div>
<p class="note">% of episodes. No code: the model's answer had no code (cut off at the token limit, prompt too
long, or empty). Timed out: CaP-X's limit of 1000 s per trial{
" (model time not counted)" if runs and not runs[0].info["capx"]["timeout_includes_model_time"] else ""},
three attempts. For multi-turn variants, the last code block's result counts.</p>"""


def _examples(episodes: list[Episode], runs: list[Run], done: list[dict]) -> str:
    if not runs:
        return ""
    episodes = [e for e in episodes if all(e.id in d for d in done)]  # not N/A for any run
    success = {e.id: [d[e.id]["task_completed"] for d in done] for e in episodes}
    by_config = {}
    for e in episodes:
        by_config.setdefault(e.config.id, e)  # one episode per config (its first trial), for variety
    candidates = list(by_config.values())
    picks = pick([
        [e for e in candidates if len(set(success[e.id])) > 1],  # runs disagree
        [e for e in candidates if all(success[e.id])],
        [e for e in candidates if not any(success[e.id])],
    ], N_EXAMPLES)

    sections = []
    for i, (run, run_done) in enumerate(zip(runs, done)):
        figures = []
        for e in picks:
            d = run_done[e.id]
            image_path = runner.OUTPUTS_DIR / run.name / run.engine / "capx" / "episodes" / e.config.id / f"{e.trial:04d}.jpg"
            image = (f'<img src="{jpeg_data_url(Image.open(image_path))}" alt="last camera image">'
                     if image_path.exists() else '<p class="muted">no image</p>')
            verdict = ('<span class="ok">✓ task done</span>' if d["task_completed"]
                       else f'<span class="bad">✗</span> {OUTCOME_WORDS[outcome(d)]}')
            figures.append(
                f"<figure>{image}<figcaption><b>{esc(e.config.id)}</b>, trial {e.trial}: {esc(d['goal'])}<br>"
                f"{verdict} · {len(d['calls'])} call{'s' * (len(d['calls']) != 1)}"
                f"<details><summary>Code</summary><pre class=\"code\">{esc(d['code'] or '(none)')}</pre></details>"
                f"</figcaption></figure>")
        sections.append(f'<h3><span class="dot" style="background:{COLORS[i % len(COLORS)]}"></span>{esc(run.label)}</h3>'
                        f'<div class="examples">{"".join(figures)}</div>')
    return f"""<h2>Examples</h2>
<p>The same episodes for every run: the camera image when the episode ended, and the code the model wrote (all
blocks, for multi-turn). Picked evenly from configs where runs disagree, where all succeed, and where all fail.</p>
<style>pre.code {{ max-height: 320px; overflow: auto; font-size: .72rem; white-space: pre-wrap;
background: var(--head); padding: 6px; border-radius: 4px; }} h3 {{ font-size: 1rem; margin: 20px 0 0; }}</style>
{"".join(sections)}"""


def _runs_table(runs: list[Run]) -> str:
    rows = "".join(
        f'<tr><td class="l">{esc(run.label)}</td><td class="l"><code>{esc(run.info["repo"])}'
        f'@{run.info["revision"][:7]}</code></td><td class="l">{esc(run.info["engine"])}</td>'
        f'<td class="l wrap"><code>{esc(json.dumps(run.info["request_settings"]))}</code></td>'
        f'<td>{run.info["max_seq_len"]}</td><td class="l"><code>{run.info["capx"]["commit"][:7]}</code></td>'
        f'<td class="l wrap"><code>{esc(json.dumps({k: v for k, v in run.info["capx"].items() if k != "commit"}))}'
        f"</code></td></tr>"
        for run in runs)
    return f"""<h2>Runs</h2>
<div class="scroll"><table>
<tr><th class="l">Model</th><th class="l">Weights</th><th class="l">Engine</th><th class="l">Request settings</th>
<th>Context<br>(tokens)</th><th class="l">CaP-X</th><th class="l">CaP-X settings</th></tr>
{rows}
</table></div>
<p class="note">Every model call is sent with these settings instead of CaP-X's. When a long multi-turn prompt leaves less room in the
context than max_tokens, that call's max_tokens is lowered to fit. CaP-X settings: the name CaP-X is given
for the model, simulators run at once, CaP-X's trial timeout and attempts and whether model time counts toward it,
and whether the seed is offset by the trial number. Generated by <code>scripts/report.py</code>
from <code>outputs/</code>.</p>"""
