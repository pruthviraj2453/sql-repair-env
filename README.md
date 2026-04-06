---
title: SQL Repair Environment
emoji: 🛠️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - sql
  - reinforcement-learning
  - code-repair
---

# SQL Repair Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible reinforcement learning environment where an agent is given a **broken SQL query** and must fix it. Correctness is evaluated automatically by executing both the agent's query and the ground-truth query against an in-memory SQLite database and comparing result sets.

---

## Why This Environment?

### The Problem It Solves

SQL is the world's most widely used data language — analysts, engineers, scientists, and non-technical business users all write it every day. Despite this ubiquity, SQL errors are expensive and common:

- **Production pipelines break silently.** A mis-typed column name or a wrong JOIN key returns wrong results without raising an exception. Dashboards show incorrect numbers. Business decisions are made on bad data.
- **Code review doesn't catch semantic bugs.** A query that selects `e.id = d.dept_id` instead of `e.dept_id = d.dept_id` compiles fine, runs fine, and returns plausible-looking rows — yet the answer is wrong.
- **Stack Overflow style help is slow and unscalable.** Engineers post broken queries and wait hours for a human to spot the bug. An agent that can autonomously fix SQL would accelerate every data team.

### Why It's Valuable for the RL / Agent Community

Most code-generation benchmarks evaluate agents in a **one-shot, zero-feedback** setting: generate an answer, score it, done. Real-world software engineering is nothing like that. Engineers iterate — they run code, read the error, fix the bug, run again. This environment is designed to reward that feedback loop:

1. **Execution-grounded feedback.** The reward is not a language model's opinion or a fuzzy string similarity score. The agent's SQL is *actually executed* against a real SQLite database, and its output is compared row-by-row to the ground truth. There is no way to fool the grader with plausible-sounding text.

2. **Dense, continuous reward.** Scoring is continuous on a 0.0-1.0 scale, blending row overlap (60%), column match (25%), and count ratio (15%). A query that hits the right tables but returns the wrong columns earns partial credit proportional to how close it is. This makes credit assignment tractable — partial progress is visible, not invisible.

3. **Multi-turn self-correction.** The agent can call `submit_fix` multiple times per episode. Each call returns the actual rows it produced versus the expected rows. A model that learns to *read the diff and correct itself* generalises to a much broader class of debugging problems than one that just regenerates from scratch.

4. **Curriculum built in.** Twenty-two tasks across three difficulty levels (easy / medium / hard) are randomly selected at each episode reset. This natural curriculum — from surface-level typos to subtle semantic errors involving window functions and CTEs — makes the environment suitable for both rapid prototyping (easy tasks confirm the plumbing works) and frontier research (hard tasks require genuine reasoning about SQL semantics).

5. **Zero external dependencies.** The database is `sqlite3` (Python standard library). No database server, no credentials, no network I/O. The environment boots in under a second and is fully reproducible.

---

## Environment Overview

| Property | Value |
|---|---|
| Action space | MCP tool calls (`get_task`, `explore_db`, `submit_fix`) |
| Observation space | Dict — broken SQL, schema, description, difficulty, rows, score |
| Reward | Continuous 0.0-1.0 (60% row overlap + 25% column match + 15% count ratio) |
| Tasks | 22 tasks: 7 easy, 6 medium, 9 hard |
| Episode termination | `submit_fix` returns `done: true` on score 1.0 or max steps reached |
| Database | In-memory SQLite, recreated fresh on every `reset()` |
| Task selection | Random on `reset()`; reproducible with `seed=` |

---

## Action Space

The agent interacts exclusively through three MCP tools.

### `get_task()`

Returns the task for the current episode. Call this once at the start of each episode.

**Returns:**

```json
{
  "broken_sql":  "SELECT emplyee_name FROM employees WHERE dept = 'engineering';",
  "schema":      "Table: employees\n  id INTEGER PRIMARY KEY\n  employee_name TEXT\n  ...",
  "description": "The query has a typo in a column name. Fix it so it returns all employee names from the engineering department.",
  "difficulty":  "easy",
  "task_name":   "easy_column_typo"
}
```

### `explore_db(sql: str)`

Execute arbitrary SELECT queries to explore the database schema and test hypotheses before submitting a fix. This does **not** count as a graded submission, but it increments the step count.

**Arguments:**

| Name | Type | Description |
|---|---|---|
| `sql` | `str` | A SELECT query to execute. INSERT/UPDATE/DELETE/DROP/ALTER/CREATE are rejected. |

**Returns:**

```json
{
  "columns":   ["employee_name", "dept"],
  "rows":      [["Alice", "engineering"], ["Bob", "engineering"]],
  "row_count": 2,
  "truncated": false,
  "error":     null
}
```

Results are capped at 50 rows. If the result set exceeds this limit, `truncated` is `true`. If the query fails, `error` contains the error message and `rows` is empty.

### `submit_fix(sql: str)`

Submit a repaired SQL query. The query is executed against the in-memory database; the result set is compared to the ground truth.

**Arguments:**

| Name | Type | Description |
|---|---|---|
| `sql` | `str` | The fixed SQL query to evaluate |

**Returns:**

```json
{
  "score":         1.0,
  "message":       "Perfect match! All rows and columns are correct.",
  "agent_rows":    [["Alice"], ["Bob"], ["Dave"]],
  "expected_rows": [["Alice"], ["Bob"], ["Dave"]],
  "done":          true
}
```

---

## Observation Space

Each call to `submit_fix` returns an observation with the following fields:

| Field | Type | Description |
|---|---|---|
| `score` | `float` | Continuous 0.0-1.0 (see Scoring Rubric below) |
| `message` | `str` | Human-readable explanation of the score |
| `agent_rows` | `list` | Sorted result rows from the agent's query |
| `expected_rows` | `list` | Sorted result rows from the ground-truth query |
| `done` | `bool` | `true` when score == 1.0 (episode complete) |

The `get_task()` observation adds:

| Field | Type | Description |
|---|---|---|
| `broken_sql` | `str` | The SQL query containing the bug |
| `schema` | `str` | Human-readable schema for the database |
| `description` | `str` | Natural-language description of what the query should do |
| `difficulty` | `str` | `"easy"`, `"medium"`, or `"hard"` |
| `task_name` | `str` | Internal task identifier |

---

## Tasks

All 22 tasks are randomly selected on each `reset()`. Pass a fixed integer `seed` to `reset()` for reproducible selection.

### Easy (7 tasks) — Surface-level syntax errors

| Task | Bug Type |
|---|---|
| `easy_column_typo` | Misspelled column name |
| `easy_table_name_typo` | Misspelled table name |
| `easy_missing_where` | Missing WHERE clause |
| `easy_wrong_comparison_operator` | `>=` instead of `>` |
| `easy_wrong_string_literal` | Typo in string constant |
| `easy_wrong_sort_direction` | ASC instead of DESC |
| `easy_select_star_instead_of_columns` | SELECT * instead of named columns |

**Characteristics:** SQLite raises an error or the bug is immediately visible from the output. Fixes are lexical — no deep SQL reasoning required.

### Medium (6 tasks) — Semantic errors, silent wrong results

| Task | Bug Type |
|---|---|
| `medium_wrong_join_column` | Wrong column in JOIN ON clause |
| `medium_wrong_join_type` | INNER JOIN instead of LEFT JOIN |
| `medium_null_handling` | `= NULL` instead of `IS NULL` |
| `medium_wrong_aggregate` | SUM instead of AVG |
| `medium_missing_group_by` | Missing GROUP BY clause |
| `medium_incorrect_order_by_column` | ORDER BY wrong column |

**Characteristics:** Queries execute without error but return wrong results. The agent must compare output rows to the task intent and reason about SQL semantics.

### Hard (9 tasks) — Multi-bug, aggregation, window functions, CTEs

| Task | Bug Type |
|---|---|
| `hard_wrong_having_clause` | Wrong aggregate in HAVING |
| `hard_subquery_wrong_column` | Subquery averages wrong column |
| `hard_missing_having_with_group_by` | WHERE on aggregate instead of HAVING |
| `hard_multiple_bugs_join_and_filter` | Wrong JOIN direction + wrong filter threshold |
| `hard_distinct_missing` | Missing DISTINCT + wrong ORDER BY |
| `hard_union_wrong_column_count` | Column count mismatch in UNION + UNION vs UNION ALL |
| `hard_nested_subquery_error` | Correlated subquery references wrong column + wrong aggregate |
| `hard_window_function_ranking` | Wrong window function (ROW_NUMBER vs RANK) + wrong partition |
| `hard_cte_self_join` | CTE with broken self-join logic and wrong aggregation |

**Characteristics:** Both the broken and correct queries produce valid, plausible-looking results. Fixing requires reasoning about aggregate semantics, correlated subqueries, window functions, or CTEs.

---

## Scoring Rubric

Scoring is **continuous** on a 0.0-1.0 scale, computed as a weighted blend of three components:

| Component | Weight | What it measures |
|---|---|---|
| **Row overlap** | 60% | Fraction of expected rows that appear in agent output |
| **Column match** | 25% | For non-overlapping rows, how many column values match pairwise |
| **Count ratio** | 15% | `min(expected, actual) / max(expected, actual)` row count similarity |

| Score | Meaning |
|---|---|
| **1.0** | Exact match — agent rows == expected rows (sorted) |
| **0.1 - 0.99** | Partial credit — proportional to row overlap, column accuracy, and row count |
| **0.1** | Query executed but returned completely wrong results (floor for valid SQL) |
| **0.0** | SQL syntax error — query did not execute |

This continuous scoring provides a smooth gradient for RL training. Any syntactically valid query receives at least 0.1, giving the agent a clear signal that executing SQL is better than producing syntax errors.

---

## Setup & Running

### Prerequisites

- Docker
- Python 3.10+
- `uv` (recommended) or `pip`

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Yes | Base URL for an OpenAI-compatible inference API (e.g. HuggingFace Inference API) |
| `MODEL_NAME` | Yes | Model identifier (e.g. `Qwen/Qwen2.5-72B-Instruct`) |
| `HF_TOKEN` | Yes | HuggingFace API token, used as the API key |
| `ENV_BASE_URL` | No | Environment server URL (default: `http://localhost:8000`) |
| `EPISODES_PER_TASK` | No | How many episodes to run per task (default: `1`) |
| `MAX_ATTEMPTS` | No | Max `submit_fix` calls per episode (default: `3`) |

### Build and Run the Environment Server

```bash
# From the sql_repair_env directory
docker build -t sql-repair-env:latest -f server/Dockerfile .

docker run --rm -p 8000:8000 sql-repair-env:latest
```

Verify the server is up:

```bash
curl http://localhost:8000/health
# → {"status": "ok"}
```

### Run inference.py

```bash
# Install dependencies (if running outside Docker)
cd envs/sql_repair_env
uv sync

# Set required environment variables
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_..."

# Run (environment server must already be running on port 8000)
python inference.py
```

### Run via OpenEnv CLI

```bash
# From repo root
openenv serve sql_repair_env   # starts the server
openenv run   sql_repair_env   # runs inference.py against it
```

### Run Tests

```bash
PYTHONPATH=src:envs uv run pytest tests/envs/test_sql_repair_environment.py -v
```

---

## Log Format

`inference.py` emits key=value log lines to stdout (one line per event):

```
[START] task=<task_name> env=sql_repair_env model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
```

Example:

```
[START] task=easy_column_typo env=sql_repair_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=explore_db reward=0.00 done=false error=null
[STEP] step=2 action=submit_fix(SELECT employee_name FROM employees WHERE dept = 'engineering') reward=1.00 done=true error=null
[END] success=true steps=2 score=1.00 rewards=0.00,1.00
```

---

## Baseline Results

Baseline measured with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference API:

| Difficulty | Tasks | Avg Score | Notes |
|-----------|-------|-----------|-------|
| Easy      | 7     | ~0.85     | Most solved on first attempt |
| Medium    | 6     | ~0.60     | Often requires 2-3 attempts |
| Hard      | 9     | ~0.35     | Window functions and CTEs remain challenging |
| **Overall** | **22** | **~0.55** | Difficulty-weighted: ~0.50 |

Run your own baseline:

```bash
API_BASE_URL="https://router.huggingface.co/v1" \
MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
HF_TOKEN="hf_..." \
python inference.py
```

---

## Manual Testing (curl)

> **Warning — read this before using curl for multi-step testing.**
>
> The HTTP server pools multiple environment instances. Raw `curl` requests are load-balanced across the pool, so `/reset` and subsequent `/step` calls may land on **different instances** with different database state. This causes confusing errors like `"no such table"` or wrong `expected_rows`.
>
> For any test that involves more than one request (reset → get_task → submit_fix), use the **Python client** instead (see below). It opens a WebSocket session that pins all calls to a single instance.
>
> Use `curl` only for one-shot checks like `/health` or a standalone `/reset`.

### Health check

```bash
curl http://localhost:8000/health
# → {"status": "healthy"}
```

### Reset

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 0}'
# → {"observation": {}, "reward": 0.0, "done": false}
```

### Step (correct request format)

Note the `action` wrapper — the `/step` endpoint expects `{"action": {"tool_name": ..., "arguments": ...}}`:

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"tool_name": "get_task", "arguments": {}}}'
```

---

## Using the Client Directly

The Python client uses a WebSocket session, so all calls within a `with` block are routed to the same environment instance. This is the correct way to do multi-step manual or automated testing.

> **Note:** `SqlRepairEnv` is async by default. Use `.sync()` to get a synchronous wrapper for scripts and interactive testing.

```python
from sql_repair_env import SqlRepairEnv

# .sync() gives a synchronous context manager — no async/await needed
with SqlRepairEnv(base_url="http://localhost:8000").sync() as env:
    # Reset — randomly selects one of the 22 tasks
    env.reset()

    # Fetch the task
    task = env.call_tool("get_task")
    print(task["difficulty"])   # "easy" | "medium" | "hard"
    print(task["broken_sql"])
    print(task["schema"])

    # Submit a fix
    result = env.call_tool("submit_fix", sql="SELECT employee_name FROM employees WHERE dept = 'engineering';")
    print(result["score"])    # 1.0
    print(result["done"])     # True
```

**Reproducible task selection:**

```python
env.reset(seed=42)   # always selects the same task for seed=42
```

**From a Docker image:**

```python
env = SqlRepairEnv.from_docker_image("sql-repair-env:latest")
try:
    env.reset()
    task = env.call_tool("get_task")
    result = env.call_tool("submit_fix", sql=task["broken_sql"])  # intentionally wrong
    print(result["score"])  # 0.0 or SQL error
finally:
    env.close()
```

---

## File Structure

```
sql_repair_env/
├── README.md              ← this file
├── openenv.yaml           ← environment metadata
├── pyproject.toml         ← package config and dependencies
├── inference.py           ← agent inference script (hackathon entry point)
├── client.py              ← SqlRepairEnv client (extends MCPToolClient)
├── __init__.py            ← package exports
└── server/
    ├── sql_environment.py ← core environment: tasks, grader, MCP tools
    ├── app.py             ← FastAPI server (create_app wrapper)
    ├── __init__.py
    └── Dockerfile         ← multi-stage Docker build
```

---

## License

BSD-style license — see [LICENSE](../../LICENSE) in the root of this repository.
