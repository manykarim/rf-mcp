"""Robot Framework MCP Server powered by RobotCode CLI and Skills over MCP."""

from importlib import metadata

try:
    __version__ = metadata.version("rf-mcp")
except metadata.PackageNotFoundError:
    __version__ = "2.0.0"

__all__ = ["__version__"]
