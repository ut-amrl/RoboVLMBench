"""RoboBench: the parts with deterministic scoring, 3,087 of its 6,092 questions.

    mcq    all 1,895 multiple-choice questions (A-D), the official "RoboBench-MCQ" setting: 13 subtasks in
           perception reasoning, affordance reasoning and error analysis
    state  all 1,192 yes/no "state estimation" questions of planning (Planning Q3: is this step finished,
           given the video so far?), in 10 subtasks of instruction comprehension and generalized planning

Left out: the other 3,005 planning questions, Q1 (write a multi-step plan; 2,163 questions, incl. the 190
of navigation) and Q2 (name the next step; 842). The official evaluation scores them only with an LLM
judge (GPT-4o by default: a world-simulator judge with the scene image and a dependency graph for Q1,
step extraction and a 3-part comparison for Q2), which this harness does not call.

Prompts and images follow the official code (third_party/RoboBench, the `robobench` package): its
RoboBenchDataset builds each prompt from the released system_prompt.json (task definitions, then the
question, the options, and "Please answer only one word, e.g., A, B, C, or D."), and every image is
squashed to 512x512 and sent as JPEG (quality 85) by its process_image. Videos are the released frames,
which are what the official code sends: 10-34 per MCQ video question, 1-20 per state question (the
video up to the step asked about); nothing is re-sampled. The one difference: our client puts the
images before the text, the official one after.

Scoring: MCQ answers are read by the official rule (MultiChoiceEvaluator): an answer that is only
letters ("B", "b.", "A, C") counts as those letters, and must equal the answer letter. The official code
sends any other answer to GPT-4o to pick out a letter; here a fixed fallback does that instead ("B. Exo",
"The answer is B", or the text of exactly one option), and answers it can't read count as wrong. Yes/no
answers: the official code always asks GPT-4o to extract "yes" or "no" and then compares it with the
answer; here an answer that is just yes or no is taken as is, otherwise its first word or the only one
of yes/no it contains. `per_sample[i]["parsed_by"]` says which rule read each answer.

Official scores are per subtask (accuracy); RoboBench-MCQ and the per-dimension scores here are the mean
over subtasks, as the official aggregator averages over tasks.
"""

import base64
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path

from robovlmbench import DATA_DIR
from robovlmbench.formats import PointFormat
from robovlmbench.models.client import Request, Response

DATA_ROOT = DATA_DIR / "robobench"

# Subtasks per dimension, in the order of the official config (config/benchmark.example.yaml).
MCQ_SUBTASKS = {
    "perception_reasoning": ["static_attribute", "tool_usage", "robot_type", "robot_view", "spatial_relation",
                             "spatial_temporal_causality", "timestamp_analysis", "task_instruction_comprehension"],
    "affordance_reasoning": ["static_affordance", "dynamic_affordance", "navigation_visual_prompt"],
    "error_analysis": ["high_level_planning_error", "low_level_execution_error"],
}
# Planning subtasks with state-estimation (Q3) questions; the others have only Q1 plans.
STATE_SUBTASKS = {
    "instruction_comprehension": ["explicit_object_goal", "implicit_demand_goal"],
    "generalized_planning": ["single_arm", "mobile_manipulation", "rigid", "articulated", "deformable",
                             "multi_object", "special", "world_knowledge"],
}
OUTCOMES = ("answered", "unparsable", "truncated", "failed")


@dataclass
class Sample:
    id: str                 # the official unique_id; "@2" added to the second of the 4 used twice
    kind: str               # "mcq" or "state"
    dimension: str          # e.g. "perception_reasoning"
    subtask: str            # e.g. "robot_view"
    input_type: str         # "image" or "video" (video: released frames, in order)
    prompt: str             # the official prompt, system prompt included
    question: str           # the question alone, as released (for display)
    answer: str             # "A"-"D" for mcq, "yes"/"no" for state
    options: list[str] = field(default_factory=list)  # option texts, A-D (mcq only)
    image_paths: list[Path] = field(default_factory=list)


def _questions_files() -> dict[str, Path]:
    """Subtask name -> its questions.json, found without walking the 88k images (the official loader's
    rglob takes ~25 s); it asserts the same: one file per subtask."""
    found = {}
    for root, dirs, files in os.walk(DATA_ROOT):
        dirs[:] = [d for d in dirs if d != "images"]
        if "questions.json" in files:
            name = Path(root).name
            assert name not in found, f"two questions.json for {name}"
            found[name] = Path(root) / "questions.json"
    return found


def load_samples() -> list[Sample]:
    """The 3,087 included questions, interleaved across subtasks so any prefix covers all 23 of them.

    Prompts and image paths come from the official RoboBenchDataset (its load_questions does the same,
    after a slower file search)."""
    from robobench.data.dataset import RoboBenchDataset
    dataset, files = RoboBenchDataset(str(DATA_ROOT)), _questions_files()
    groups = []
    for kind, subtasks in (("mcq", MCQ_SUBTASKS), ("state", STATE_SUBTASKS)):
        for dimension, names in subtasks.items():
            for subtask in names:
                path = files[subtask]
                assert path.relative_to(DATA_ROOT).parts[0] == dataset._dimension_dir(dimension)
                questions = [dataset._prepare_question(q, path.parent) for q in json.loads(path.read_text())]
                if kind == "state":
                    questions = [q for q in questions if q["request_id"].endswith("_Q3")]
                groups.append([_sample(q, kind, dimension, subtask) for q in questions])
    samples = [s for group in zip_longest(*groups) for s in group if s is not None]

    seen = Counter()
    for s in samples:  # 4 official ids here (6 in the release) are used by two different questions each
        seen[s.id] += 1
        if seen[s.id] > 1:
            s.id = f"{s.id}@{seen[s.id]}"
    assert len({s.id for s in samples}) == len(samples)
    return samples


def _sample(question: dict, kind: str, dimension: str, subtask: str) -> Sample:
    answer = question["gt_answer"].strip()
    if kind == "state":
        answer = answer.lower()
        assert answer in ("yes", "no"), question["request_id"]
    else:
        assert answer in ("A", "B", "C", "D") and len(question["options"]) == 4, question["request_id"]
    return Sample(question["request_id"], kind, dimension, subtask, question["input_type"], question["prompt"],
                  question["question"], answer, list(question.get("options") or []),
                  [Path(p) for p in question["image_urls"]])


def encode_image(path: Path) -> bytes:
    """An image as the official code sends it: resized to 512x512 (aspect not kept), JPEG quality 85."""
    from robobench.inference.image import process_image
    b64, media_type = process_image(path, resize=True, fmt="jpeg", quality=85)
    assert media_type == "image/jpeg"
    return base64.b64decode(b64)


def build_request(sample: Sample, point_format: PointFormat) -> Request:
    """The official prompt and images. `point_format` is unused: every answer is a letter or yes/no."""
    # the official code sends the images after the text
    return Request(sample.id, sample.prompt, [encode_image(p) for p in sample.image_paths], images_first=False)


# --- answer parsing -------------------------------------------------------------------------------

_ANSWER_PHRASE = re.compile(r"(?i:answer|option|choice)\s*(?:(?i:is)\s*)?[:\-]?\s*[\(\[\*]*([A-D])(?![A-Za-z])")
_LEADING_LETTER = re.compile(r"^[\s\(\[\*]*([A-D])[\)\]\.:\*]")
_YES_NO_ONLY = re.compile(r"^\W*(yes|no)\W*$", re.IGNORECASE)
_YES_NO_FIRST = re.compile(r"^\W*(yes|no)\b", re.IGNORECASE)
_YES_NO_WORD = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _official_mcq():
    from robobench.evaluation.multi_choice import MultiChoiceEvaluator
    evaluator = MultiChoiceEvaluator()
    evaluator.normalize_with_gpt = False  # ambiguous answers raise instead of calling GPT
    return evaluator


def parse_mcq(answer: str, options: list[str], official=None) -> tuple[str, str | None]:
    """(letters, rule): the answer's option letters, sorted ("" if none), and the rule that read them:
    "exact" (the official rule: the answer is only letters), "fallback" (our stand-in for the official
    GPT normalization), or None (unreadable)."""
    official = official or _official_mcq()
    try:
        letters = official._extract_response_letters(answer)
        return letters, ("exact" if letters else None)
    except ValueError:  # not just letters: the official code would ask GPT-4o for one letter
        pass
    text = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL | re.IGNORECASE).strip()
    if tagged := re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE):
        text = tagged.group(1).strip()
    if found := _ANSWER_PHRASE.findall(text):
        return found[-1], "fallback"
    if found := _LEADING_LETTER.match(text):
        return found.group(1), "fallback"
    normalized = _words(text)
    matches = [chr(ord("A") + i) for i, option in enumerate(options) if _words(option) == normalized]
    if len(matches) == 1:
        return matches[0], "fallback"
    return "", None


def parse_yes_no(answer: str) -> tuple[str, str | None]:
    """("yes"/"no"/"", rule): "exact" if the answer is only yes or no, "fallback" if read from its first
    word or its only yes/no word, None if unreadable."""
    if found := _YES_NO_ONLY.match(answer):
        return found.group(1).lower(), "exact"
    if found := _YES_NO_FIRST.match(answer):
        return found.group(1).lower(), "fallback"
    words = {w.lower() for w in _YES_NO_WORD.findall(answer)}
    if len(words) == 1:
        return words.pop(), "fallback"
    return "", None


def _words(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


# --- scoring --------------------------------------------------------------------------------------

def score(samples: list[Sample], responses: list[Response], point_format: PointFormat) -> dict:
    """Headline `accuracy`: RoboBench-MCQ, the mean of the 13 MCQ subtask accuracies. Also per dimension
    (mean over its subtasks) and per subtask, and the same for state estimation, kept apart.

    Per sample: `parsed`, the answer as read ("B", "AC", "yes"; "" if unreadable), and `parsed_by`
    ("exact", "fallback" or None, see parse_mcq / parse_yes_no).
    """
    official = _official_mcq()
    per_sample = []
    for sample, response in zip(samples, responses, strict=True):
        if sample.kind == "mcq":
            parsed, rule = parse_mcq(response.answer, sample.options, official)
            correct = parsed == official._extract_option_letters(sample.answer)
        else:
            parsed, rule = parse_yes_no(response.answer)
            correct = parsed == sample.answer
        per_sample.append({
            "id": sample.id, "kind": sample.kind, "dimension": sample.dimension, "subtask": sample.subtask,
            "parsed": parsed, "parsed_by": rule, "correct": bool(correct),
            "outcome": ("failed" if response.error else "truncated" if response.finish_reason == "length"
                        else "unparsable" if rule is None else "answered"),
        })

    def accuracy(rows):
        return round(sum(r["correct"] for r in rows) / len(rows), 4) if rows else None

    def mean(values):
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 4) if values else None

    def summary(kind: str, subtasks: dict[str, list[str]]) -> dict:
        rows = [r for r in per_sample if r["kind"] == kind]
        by_subtask = {}
        for dimension, names in subtasks.items():
            for name in names:
                if sub := [r for r in rows if r["subtask"] == name]:
                    by_subtask[name] = {"dimension": dimension, "accuracy": accuracy(sub), "n": len(sub)}
        by_dimension = {}
        for dimension in subtasks:
            parts = [v for v in by_subtask.values() if v["dimension"] == dimension]
            if parts:
                by_dimension[dimension] = {"accuracy": mean(p["accuracy"] for p in parts),
                                           "n": sum(p["n"] for p in parts)}
        rules = Counter(r["parsed_by"] for r in rows)
        return {
            "accuracy": mean(v["accuracy"] for v in by_subtask.values()),  # mean over subtasks
            "micro_accuracy": accuracy(rows),                             # over questions
            "n": len(rows),
            "parsed_by": {"exact": rules["exact"], "fallback": rules["fallback"], "unreadable": rules[None]},
            "by_dimension": by_dimension,
            "by_subtask": by_subtask,
        }

    mcq, state = summary("mcq", MCQ_SUBTASKS), summary("state", STATE_SUBTASKS)
    outcomes = Counter(r["outcome"] for r in per_sample)
    return {
        "accuracy": mcq["accuracy"],
        "mcq": mcq,
        "state": state,
        "n": len(per_sample),
        "outcomes": {o: outcomes[o] for o in OUTCOMES},
        "per_sample": per_sample,
    }
