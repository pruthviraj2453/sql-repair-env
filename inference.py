# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Inference script for the SQL Repair Environment.

Runs the agent against all tasks, each EPISODES_PER_TASK times, then
prints a final per-task average score.

Mandatory log format (hackathon harness requirement):
    [START] task=<task_name> env=sql_repair_env model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>

Environment variables (all required at runtime):
    API_BASE_URL  — base URL for the OpenAI-compatible inference endpoint
    MODEL_NAME    — model identifier (e.g. "meta-llama/Llama-3.1-8B-Instruct")
    HF_TOKEN      — HuggingFace token used as the API key

Optional overrides:
    ENV_BASE_URL      — environment server URL (default: http://localhost:8000)
    EPISODES_PER_TASK — how many episodes to run per task (default: 1)
    MAX_ATTEMPTS      — max submit_fix attempts per episode (default: 3)
    TIMEOUT_MINUTES   — hard wall-clock limit in minutes (default: 18)

Usage:
    python inference.py
"""

import json
import os
import sys
import time
from collections import defaultdict
from typing import List, Optional

from openai import OpenAI

from client import SqlRepairEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
HF_TOKEN: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
MODEL_NAME: str = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:8000")
EPISODES_PER_TASK: int = int(os.getenv("EPISODES_PER_TASK", "1"))
MAX_ATTEMPTS: int = int(os.getenv("MAX_ATTEMPTS", "3"))
# Leave a 2-minute safety buffer before the 20-minute hackathon deadline.
TIMEOUT_MINUTES: int = int(os.getenv("TIMEOUT_MINUTES", "18"))

BENCHMARK: str = "sql_repair_env"

# Derive ALL_TASKS from the server module to stay in sync automatically.
# If this import fails during inference (e.g. running outside Docker without
# the server package), fall back to a hard-coded set as a safety net.
try:
    from server.sql_environment import TASKS as _SERVER_TASKS  # noqa: F811

    ALL_TASKS: frozenset[str] = frozenset(t["name"] for t in _SERVER_TASKS)
except ImportError:
    ALL_TASKS: frozenset[str] = frozenset([  # type: ignore[no-redef]
        # easy
        "easy_column_typo",
        "easy_table_name_typo",
        "easy_missing_where",
        "easy_wrong_comparison_operator",
        "easy_wrong_string_literal",
        "easy_wrong_sort_direction",
        "easy_select_star_instead_of_columns",
        # medium
        "medium_wrong_join_column",
        "medium_wrong_join_type",
        "medium_null_handling",
        "medium_wrong_aggregate",
        "medium_missing_group_by",
        "medium_incorrect_order_by_column",
        # hard
        "hard_wrong_having_clause",
        "hard_subquery_wrong_column",
        "hard_missing_having_with_group_by",
        "hard_multiple_bugs_join_and_filter",
        "hard_distinct_missing",
        "hard_union_wrong_column_count",
        "hard_nested_subquery_error",
        "hard_window_function_ranking",
        "hard_cte_self_join",
    ])

# ---------------------------------------------------------------------------
# Logging — MUST match hackathon's exact key=value format
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert SQL developer. Your job is to fix broken SQL queries.\n"
    "You can use explore_db to execute SELECT queries and explore the database "
    "before submitting your fix.\n"
    "Reply with ONLY the corrected SQL query — no explanation, no markdown, "
    "no code fences."
)


def _initial_messages(
    broken_sql: str,
    schema: str,
    description: str,
    broken_query_result: str | None = None,
) -> list[dict]:
    user_content = (
        f"Task: {description}\n\n"
        f"Schema:\n{schema}\n\n"
        f"Broken SQL:\n{broken_sql}\n\n"
    )
    if broken_query_result is not None:
        user_content += (
            f"I ran the broken query for you. Here is the result:\n"
            f"{broken_query_result}\n\n"
        )
    user_content += "Reply with ONLY the corrected SQL query."
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _retry_message(score: float, feedback: str, agent_rows: list, expected_rows: list) -> dict:
    return {
        "role": "user",
        "content": (
            f"Your fix scored {score:.2f}/1.00. Feedback: {feedback}\n"
            f"Your result rows:   {agent_rows}\n"
            f"Expected rows:      {expected_rows}\n\n"
            "Try again. Reply with ONLY the corrected SQL query."
        ),
    }


def call_llm(messages: list[dict]) -> str:
    """Call the model and return the raw text response."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episode(
    env: SqlRepairEnv,
    episode_num: int,
    seed: int,
) -> tuple[str, float]:
    """
    Reset the environment with *seed*, run up to MAX_ATTEMPTS submit_fix calls,
    and return (task_name, final_score).

    The seed controls which task the environment selects, allowing deterministic
    task routing without hard-coding task-to-seed mappings.
    """
    # Reset with seed — environment picks the task deterministically.
    env.reset(seed=seed)

    # Fetch task details (also reveals task_name for routing).
    task = env.call_tool("get_task")
    task_name: str = task["task_name"]
    broken_sql: str = task["broken_sql"]
    schema: str = task["schema"]
    description: str = task["description"]

    # Initialize state BEFORE try block so finally always has valid values.
    rewards: List[float] = []
    step_num: int = 0
    final_score: float = 0.0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Run the broken query first to see its actual output — this gives the
        # LLM concrete information about what is wrong and demonstrates the
        # multi-step exploration capability of the environment.
        broken_result = env.call_tool("explore_db", sql=broken_sql)
        step_num += 1
        if broken_result.get("error"):
            broken_query_summary = f"Error: {broken_result['error']}"
        else:
            cols = broken_result.get("columns", [])
            rows = broken_result.get("rows", [])
            row_count = broken_result.get("row_count", len(rows))
            truncated = broken_result.get("truncated", False)
            broken_query_summary = (
                f"Columns: {cols}\n"
                f"Rows ({row_count} total{', truncated to 50' if truncated else ''}):\n"
                + "\n".join(str(r) for r in rows[:10])
                + (f"\n... ({row_count - 10} more rows)" if row_count > 10 else "")
            )

        log_step(step=step_num, action="explore_db", reward=0.00, done=False, error=None)
        rewards.append(0.0)

        messages = _initial_messages(broken_sql, schema, description, broken_query_summary)

        for attempt in range(1, MAX_ATTEMPTS + 1):
            # Ask the LLM for a fix (multi-turn: prior failures are in messages).
            try:
                fixed_sql = call_llm(messages)
            except Exception as llm_err:
                step_num += 1
                rewards.append(0.0)
                log_step(
                    step=step_num,
                    action="call_llm",
                    reward=0.00,
                    done=False,
                    error=str(llm_err),
                )
                continue

            # Submit to the environment grader.
            result = env.call_tool("submit_fix", sql=fixed_sql)
            score: float = result["score"]
            done: bool = result["done"]
            step_num += 1
            rewards.append(score)

            # Sanitize action string: replace newlines so log stays on one line
            action_str = f"submit_fix({fixed_sql})"
            action_str = action_str.replace("\n", " ").replace("\r", "")

            error_msg = result.get("message") if score == 0.0 else None

            log_step(
                step=step_num,
                action=action_str,
                reward=score,
                done=done,
                error=error_msg,
            )

            final_score = score
            if done:
                break

            # Append the failed attempt + grader feedback so the LLM can
            # self-correct on the next attempt. With temperature=0.0, without
            # new context the model would produce the same broken output.
            messages.append({"role": "assistant", "content": fixed_sql})
            messages.append(
                _retry_message(score, result["message"], result["agent_rows"], result["expected_rows"])
            )
    finally:
        success = final_score >= 0.99
        log_end(success=success, steps=step_num, score=final_score, rewards=rewards)

    return task_name, final_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    task_scores: dict[str, list[float]] = defaultdict(list)
    episode_num: int = 0
    seed: int = 0
    deadline: float = time.monotonic() + TIMEOUT_MINUTES * 60

    with SqlRepairEnv(base_url=ENV_BASE_URL).sync() as env:
        while True:
            # Done when every task has EPISODES_PER_TASK scores.
            if (
                len(task_scores) == len(ALL_TASKS)
                and all(len(task_scores[t]) >= EPISODES_PER_TASK for t in ALL_TASKS)
            ):
                break

            if time.monotonic() > deadline:
                remaining = {t for t in ALL_TASKS if len(task_scores[t]) < EPISODES_PER_TASK}
                print(
                    f"[WARN] timeout after {TIMEOUT_MINUTES}min, incomplete: {list(remaining)}",
                    flush=True,
                )
                break

            # Peek: reset with this seed and ask which task it selected.
            # This is cheap — no LLM call — so we skip seeds for saturated tasks.
            env.reset(seed=seed)
            peek = env.call_tool("get_task")
            task_name: str = peek["task_name"]

            if len(task_scores[task_name]) >= EPISODES_PER_TASK:
                # Already have enough runs for this task; try the next seed.
                seed += 1
                continue

            # Run the full episode using the same seed (re-reset happens inside).
            episode_num += 1
            try:
                returned_task, score = run_episode(env, episode_num, seed)
                task_scores[returned_task].append(score)
            except Exception as ep_err:
                print(
                    f"[WARN] episode {episode_num} (seed={seed}) failed: {ep_err}",
                    flush=True,
                )
                # Record a zero score for the peeked task so we don't retry forever.
                task_scores[task_name].append(0.0)
            seed += 1

    # ---- Summary -----------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)

    overall_scores: list[float] = []
    for task_name in sorted(ALL_TASKS):
        scores = task_scores.get(task_name, [])
        if scores:
            avg = sum(scores) / len(scores)
            overall_scores.extend(scores)
        else:
            avg = float("nan")
        scores_str = ", ".join(f"{s:.2f}" for s in scores) if scores else "—"
        print(
            f"  {task_name:<32} | episodes: {len(scores)} "
            f"| avg: {avg:.3f} | scores: [{scores_str}]",
            flush=True,
        )

    if overall_scores:
        overall_avg = sum(overall_scores) / len(overall_scores)
        print(f"\n  Average score: {overall_avg:.3f}", flush=True)

        # Difficulty-weighted average: easy=1, medium=2, hard=3
        _DIFF_WEIGHTS = {"easy": 1, "medium": 2, "hard": 3}
        weighted_sum = 0.0
        weight_total = 0.0
        for t_name, scores in task_scores.items():
            difficulty = t_name.split("_")[0]  # "easy", "medium", or "hard"
            w = _DIFF_WEIGHTS.get(difficulty, 1)
            for s in scores:
                weighted_sum += s * w
                weight_total += w
        if weight_total > 0:
            weighted_avg = weighted_sum / weight_total
            print(f"  Difficulty-weighted score: {weighted_avg:.3f}", flush=True)

    print("=" * 60, flush=True)

    # Exit code: 0 if at least one task was solved perfectly, else 1.
    perfect = any(1.0 in scores for scores in task_scores.values())
    sys.exit(0 if perfect else 1)


if __name__ == "__main__":
    main()
