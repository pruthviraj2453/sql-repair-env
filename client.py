# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SQL Repair Environment Client.

This module provides the client for connecting to a SQL Repair Environment server.
SqlRepairEnv extends MCPToolClient to provide tool-calling style interactions.

Example:
    >>> with SqlRepairEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...
    ...     # Discover available tools
    ...     tools = env.list_tools()
    ...     print([t.name for t in tools])
    ...     # ['get_task', 'explore_db', 'submit_fix']
    ...
    ...     # Fetch the broken SQL task
    ...     task = env.call_tool("get_task")
    ...     print(task["broken_sql"])
    ...
    ...     # Explore the database before submitting a fix
    ...     result = env.call_tool("explore_db", sql="SELECT * FROM employees LIMIT 5")
    ...     print(result["rows"])
    ...
    ...     # Submit a fix and get a score
    ...     result = env.call_tool(
    ...         "submit_fix",
    ...         sql="SELECT name FROM employees WHERE dept = 'engineering'",
    ...     )
    ...     print(result["score"])   # 1.0
    ...     print(result["message"]) # "Perfect match! ..."

Example with Docker:
    >>> env = SqlRepairEnv.from_docker_image("sql-repair-env:latest")
    >>> try:
    ...     env.reset()
    ...     task = env.call_tool("get_task")
    ...     result = env.call_tool("submit_fix", sql="SELECT name FROM employees")
    ... finally:
    ...     env.close()

Example with HuggingFace Space:
    >>> env = SqlRepairEnv.from_env("openenv/sql-repair-env")
    >>> try:
    ...     env.reset()
    ...     task = env.call_tool("get_task")
    ...     result = env.call_tool("submit_fix", sql="SELECT name FROM employees")
    ... finally:
    ...     env.close()
"""

from openenv.core.mcp_client import MCPToolClient


class SqlRepairEnv(MCPToolClient):
    """
    Client for the SQL Repair Environment.

    Inherits all functionality from MCPToolClient:
    - ``list_tools()``: Discover available tools.
    - ``call_tool(name, **kwargs)``: Call a tool by name.
    - ``reset(**kwargs)``: Reset the environment.
    - ``step(action)``: Execute an action (for advanced use).

    Available MCP tools on the server:
    - ``get_task``: Returns broken SQL, schema description, and task description.
    - ``explore_db(sql)``: Execute a SELECT query for exploration (not graded).
    - ``submit_fix(sql)``: Evaluates a repaired SQL query; returns score 0.0–1.0.
    """

    pass  # MCPToolClient provides all needed functionality
