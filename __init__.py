# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SQL Repair Environment — an OpenEnv RL environment for SQL query repair.

The agent receives a broken SQL query and must fix it. Correctness is evaluated
by executing both queries against an in-memory SQLite database and comparing
the result sets.

Example:
    >>> from sql_repair_env import SqlRepairEnv
    >>>
    >>> with SqlRepairEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...     task = env.call_tool("get_task")
    ...     result = env.call_tool(
    ...         "submit_fix",
    ...         sql="SELECT name FROM employees WHERE dept = 'engineering'",
    ...     )
    ...     print(result["score"])  # 1.0
"""

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from .client import SqlRepairEnv

__all__ = ["SqlRepairEnv", "CallToolAction", "ListToolsAction"]
