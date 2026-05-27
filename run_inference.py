"""


So basically rn: 
- temp scaling now uses 5 temperatures.
- First two temperatures are still 0.5 and 0.7.
- If those agree on a nonempty boxed answer, the problem early-stops.
- Otherwise, unresolved questions are generated at the remaining temperatures and voted.
i asked chat to help convert the code from my notebook to the py script since piazza said
to make it a py script. im looking over if its all correct rn and ill run it overnight to 
see if it worked. currently ive only ran with 3 temps and it got .664 on private test 
around 8th on leaderboard rn (took about 6 hours on a100). imma run with 5 temps and 
hopefully itll do better since to get full points you need to be top 10 in competition.

run these: pip uninstall -y vllm torch torchvision torchaudio xformers
pip install -U uv
uv pip install --system --no-cache vllm==0.19.1
uv pip install --system --no-cache transformers==4.57.6 tqdm sympy numpy bitsandbytes antlr4-python3-runtime==4.11.1 accelerate

then: python run_inference.py \
  --data_path data/private.jsonl \
  --output_csv submission.csv \
  --debug_csv temp_scaling_debug.csv
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Optional

import json
import pandas as pd


MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"


DEFAULT_TEMPERATURES = [0.5, 0.7, 0.9, 0.4, 0.6]

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. "
    "Please reason step by step and solve the problem carefully. "
    "Put your final answer within \\boxed{}. "
    "If the problem has multiple [ANS] placeholders, give the answers in the same order, "
    "separated by commas inside a single \\boxed{}, e.g. \\boxed{3, 7}. "
    "Do not write [ANS] in your final answer. "
    "Do not round decimal answers unless the problem explicitly asks for rounding."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician solving a multiple-choice math problem. "
    "Think carefully to determine the correct option. "
    "Do not put intermediate results in \\boxed{}. "
    "At the very end, output only one boxed capital letter. "
    "Do not include any explanation after the final answer. "
    "Example final format: \\boxed{C}."
)


def build_prompt(question: str, options: Optional[list] = None) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options is None:
        num_ans = question.count("[ANS]")
        system = SYSTEM_PROMPT_MATH
        user = (
            f"{question}\n\n"
            f"There are {num_ans} [ANS] placeholder(s). "
            f"Give exactly {num_ans} final answer(s), in order, inside one final \\boxed{{}}."
        )
    else:
        system = SYSTEM_PROMPT_MCQ
        choices = "\n".join(
            f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)
        )
        user = (
            f"{question}\n\n"
            f"Answer choices:\n{choices}\n\n"
            "Choose exactly one option letter."
        )
    return system, user


def extract_last_boxed(text: str) -> str:
    """
    Extract boxed answer similarly to the updated judger:
    - Prefer content after the final </think>
    - Extract the last contiguous group of \\boxed{...}
    - If multiple final boxes are contiguous, join them with commas
    - Fallback to searching the full text
    """
    text = str(text)

    def normalize_answer(final_answer: str) -> str:
        special_signal_map = {
            "\\left": "",
            "\\right": "",
            "∶": ":",
            "，": ",",
            "$": "",
            "\\approx": "=",
            "\\simeq": "=",
            "\\sim": "=",
            "^\\prime": "'",
            "^{\\prime}": "'",
            "^\\circ": "",
            "%": "",
        }
        for signal, repl in special_signal_map.items():
            final_answer = final_answer.replace(signal, repl)

        final_answer = final_answer.replace("\\dfrac", "\\frac")
        final_answer = final_answer.replace("\\tfrac", "\\frac")
        final_answer = final_answer.strip()
        final_answer = final_answer.strip("$")
        final_answer = final_answer.strip()
        return final_answer

    def extract_all_boxed(region: str) -> list[tuple[int, int, str]]:
        entries = []
        start = 0

        while True:
            idx = region.find("\\boxed{", start)
            if idx < 0:
                break

            brace_start = idx + len("\\boxed{")
            depth = 1
            i = brace_start

            while i < len(region) and depth > 0:
                if region[i] == "{":
                    depth += 1
                elif region[i] == "}":
                    depth -= 1
                i += 1

            if depth == 0:
                content = region[brace_start:i - 1]
                if content:
                    entries.append((idx, i, normalize_answer(content)))

            start = i

        return entries

    def extract_final_group(region: str) -> str:
        entries = extract_all_boxed(region)

        if not entries:
            return ""

        # Take last contiguous group of boxed answers.
        last_group = [entries[-1]]

        for j in range(len(entries) - 2, -1, -1):
            gap = region[entries[j][1]:entries[j + 1][0]]

            # Same spirit as updated judger:
            # allow whitespace, commas, punctuation, $, &, and slashes between boxes.
            if re.match(r"^[\s,\$\.\;\:\-\&\\]*$", gap):
                last_group.insert(0, entries[j])
            else:
                break

        return ", ".join(e[2] for e in last_group).strip()

    # Prefer content after final </think>
    think_end = text.rfind("</think>")
    search_text = text[think_end + len("</think>"):] if think_end >= 0 else text

    ans = extract_final_group(search_text)
    if ans:
        return ans

    # Fallback: search full text
    ans = extract_final_group(text)
    if ans:
        return ans

    return ""


def normalize_for_vote(ans: str) -> str:
    """
    Light normalization for voting.
    This does not solve math; it only makes equivalent formatting more likely to match.
    """
    ans = str(ans).strip()
    ans = ans.strip("$")
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace("\\dfrac", "\\frac")
    ans = ans.replace("\\tfrac", "\\frac")
    ans = ans.replace("\\,", "")
    ans = ans.replace("\\;", "")
    ans = ans.replace("\\!", "")
    ans = ans.replace(" ", "")
    ans = ans.replace("\n", "")
    return ans

def load_jsonl(path: str | Path) -> list[dict]:
    """Load a json dataset."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_inference(
    data_path: str = "data/private.jsonl",
    output_csv: str = "results/submission.csv",
    debug_csv: str | None = "results/temp_scaling_debug.csv",
    model_id: str = MODEL_ID,
    temperatures: list[float] | None = None,
    max_tokens: int = 32768,
    gpu_id: str = "0",
    gpu_memory_utilization: float = 0.75,
    max_model_len: int = 16384,
    max_num_seqs: int = 256,
    max_num_batched_tokens: int = 32768,
) -> str:
    """
    End-to-end inference pipeline.

    Args:
        data_path: Path to private.jsonl or any compatible JSONL dataset.
        output_csv: Path where final submission CSV will be written.
        debug_csv: Optional path for debug voting details. Use None to skip.
        model_id: HuggingFace model ID.
        temperatures: Temperature list. First two are used for early-stop agreement.
        max_tokens: vLLM maximum generation tokens.
        gpu_id: CUDA_VISIBLE_DEVICES value.
        gpu_memory_utilization: vLLM GPU memory utilization.
        max_model_len: vLLM max model length.
        max_num_seqs: vLLM max concurrent sequences.
        max_num_batched_tokens: vLLM max batched tokens.

    Returns:
        The output_csv path.
    """
    if temperatures is None:
        temperatures = DEFAULT_TEMPERATURES

    if len(temperatures) < 2:
        raise ValueError("temperatures must contain at least two values for early-stop logic.")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    def make_sampling_params(temp: float) -> SamplingParams:
        return SamplingParams(
            max_tokens=max_tokens,
            temperature=temp,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
        )

    # Load dataset.
    data = load_jsonl(data_path)
    ids = [item["id"] for item in data]

    n_mcq = sum(bool(d.get("options")) for d in data)
    n_free = sum(not d.get("options") for d in data)
    print(f"Loaded {len(data)} questions ({n_mcq} MCQ, {n_free} free-form).")

    # Load tokenizer/model once.
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=model_id,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=False,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )

    # Build prompts for all questions.
    prompts = []
    for item in data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    # Storage format: outputs_by_temp[temp][original_index] = response
    outputs_by_temp: dict[float, dict[int, str]] = {}

    first_temp, second_temp = temperatures[0], temperatures[1]
    remaining_temps = temperatures[2:]

    # Run the first two temperatures on every question.
    for temp in [first_temp, second_temp]:
        print(f"\nGenerating with temperature={temp} for {len(prompts)} questions...")
        outputs_t = llm.generate(
            prompts,
            sampling_params=make_sampling_params(temp),
        )
        responses_t = [out.outputs[0].text.strip() for out in outputs_t]
        outputs_by_temp[temp] = {i: response for i, response in enumerate(responses_t)}
        print(f"Done temperature={temp}")

    # early stop if first 2 temps agree on a nonempty boxed answer.
    needs_more = []
    early_results: dict[int, dict] = {}

    for i, item_id in enumerate(ids):
        r1 = outputs_by_temp[first_temp][i]
        r2 = outputs_by_temp[second_temp][i]

        raw_a1 = extract_last_boxed(r1)
        raw_a2 = extract_last_boxed(r2)

        a1 = normalize_for_vote(raw_a1)
        a2 = normalize_for_vote(raw_a2)

        if a1 and a2 and a1 == a2:
            chosen_temp = first_temp if len(r1) <= len(r2) else second_temp
            chosen_response = r1 if len(r1) <= len(r2) else r2

            early_results[i] = {
                "id": item_id,
                "response": chosen_response,
                "voted_answer": raw_a1,
                "voted_answer_norm": a1,
                "vote_count": 2,
                "chosen_temp": chosen_temp,
                "all_answers": [(first_temp, raw_a1), (second_temp, raw_a2)],
                "early_stopped": True,
            }
        else:
            needs_more.append(i)

    print(f"\nEarly stopped: {len(early_results)} / {len(ids)}")
    print(f"Need remaining temperatures: {len(needs_more)} / {len(ids)}")

    # Run remaining temperatures only for unresolved questions.
    for temp in remaining_temps:
        if not needs_more:
            print(f"\nNo unresolved questions; skipping temperature={temp}.")
            outputs_by_temp[temp] = {}
            continue

        prompts_temp = [prompts[i] for i in needs_more]
        print(f"\nGenerating with temperature={temp} for {len(prompts_temp)} unresolved questions...")

        outputs_t = llm.generate(
            prompts_temp,
            sampling_params=make_sampling_params(temp),
        )
        responses_t = [out.outputs[0].text.strip() for out in outputs_t]
        outputs_by_temp[temp] = {
            original_i: response
            for original_i, response in zip(needs_more, responses_t)
        }
        print(f"Done temperature={temp}")

    # Final voting.
    voted_results = []

    for i, item_id in enumerate(ids):
        if i in early_results:
            voted_results.append(early_results[i])
            continue

        candidates = []

        for temp in temperatures:
            if i not in outputs_by_temp.get(temp, {}):
                continue

            response = outputs_by_temp[temp][i]
            raw_ans = extract_last_boxed(response)
            norm_ans = normalize_for_vote(raw_ans)

            candidates.append(
                {
                    "temp": temp,
                    "response": response,
                    "raw_answer": raw_ans,
                    "norm_answer": norm_ans,
                }
            )

        nonempty = [c for c in candidates if c["norm_answer"]]

        if nonempty:
            counts = Counter(c["norm_answer"] for c in nonempty)
            winning_norm, winning_count = counts.most_common(1)[0]

            winning_candidates = [
                c for c in nonempty if c["norm_answer"] == winning_norm
            ]

            # shortest response with winning answer.
            chosen = min(winning_candidates, key=lambda c: len(c["response"]))
        else:
            # If no candidate had a boxed answer, fallback to shortest response.
            chosen = min(candidates, key=lambda c: len(c["response"]))
            winning_norm = ""
            winning_count = 0

        voted_results.append(
            {
                "id": item_id,
                "response": chosen["response"],
                "voted_answer": chosen["raw_answer"],
                "voted_answer_norm": winning_norm,
                "vote_count": winning_count,
                "chosen_temp": chosen["temp"],
                "all_answers": [(c["temp"], c["raw_answer"]) for c in candidates],
                "early_stopped": False,
            }
        )

    # Save final submission CSV.
    submission_df = pd.DataFrame(
        [{"id": r["id"], "response": r["response"]} for r in voted_results]
    )
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)

    print(f"\nSaved submission file: {output_path}")

    # Save debug info if requested.
    if debug_csv is not None:
        debug_df = pd.DataFrame(voted_results)
        debug_path = Path(debug_csv)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_df.to_csv(debug_path, index=False)

        print(f"Saved debug file: {debug_path}")
        print("\nSummary:")
        print("Total:", len(debug_df))
        print("Early stopped:", int(debug_df["early_stopped"].sum()))
        print("Needed remaining temps:", int((~debug_df["early_stopped"]).sum()))
        print("No extracted answer after all temps:", int((debug_df["voted_answer_norm"] == "").sum()))

        print("\nVote count distribution:")
        print(debug_df["vote_count"].value_counts().sort_index())

        print("\nChosen temperature distribution:")
        print(debug_df["chosen_temp"].value_counts().sort_index())

    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qwen3-4B math competition inference.")
    parser.add_argument("--data_path", type=str, default="data/private.jsonl")
    parser.add_argument("--output_csv", type=str, default="results/submission.csv")
    parser.add_argument("--debug_csv", type=str, default="results/temp_scaling_debug.csv")
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=DEFAULT_TEMPERATURES,
        help="Temperature list. First two are used for early-stop agreement.",
    )
    args = parser.parse_args()

    run_inference(
        data_path=args.data_path,
        output_csv=args.output_csv,
        debug_csv=args.debug_csv,
        model_id=args.model_id,
        temperatures=args.temperatures,
        max_tokens=args.max_tokens,
        gpu_id=args.gpu_id,
    )
