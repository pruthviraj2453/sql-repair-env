# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the SQL Repair Environment.

This module creates an HTTP server that exposes the SqlRepairEnvironment
over HTTP and WebSocket endpoints, compatible with MCPToolClient.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    uv run --project . server
"""

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

    from .sql_environment import SqlRepairEnvironment
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.sql_environment import SqlRepairEnvironment

# Pass the class (factory) instead of an instance for WebSocket session support.
# Use MCP types since this is a pure MCP environment.
app = create_app(
    SqlRepairEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="sql_repair_env",
)


def main():
    """
    Entry point for direct execution via uv run or python -m.

    Enables running the server without Docker:
        uv run --project . server
        python -m envs.sql_repair_env.server.app
        openenv serve sql_repair_env
    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
