"""Regression: rf-mcp must register libraries in the RF context in a way that
works across Robot Framework versions.

RF 7.4 removed the `notify` keyword argument from
``robot.running.namespace.Namespace.import_library`` (RF <=7.3 defaulted it to
True). ``library_manager.py`` used to pass ``notify=True``, which raised
``TypeError: import_library() got an unexpected keyword argument 'notify'`` on
current RF and silently degraded library registration to the on-demand path.
Found via stderr forensics on the docker uv-tool-install experiments (RF 7.4.2
in the fresh wheel install; RF 7.3.2 locally masked it).
"""
import inspect
from pathlib import Path

from robot.running.namespace import Namespace


def test_rf_mcp_call_shape_matches_installed_rf():
    """The (name, args, alias) call shape rf-mcp uses must bind on this RF."""
    sig = inspect.signature(Namespace.import_library)
    # Binding the exact call rf-mcp makes must not raise (self is unbound here,
    # so include a placeholder for it).
    sig.bind(None, "String", args=(), alias=None)


def test_library_manager_does_not_pass_notify():
    """Guard against reintroducing the version-fragile notify kwarg."""
    src = Path("src/robotmcp/core/library_manager.py").read_text(encoding="utf-8")
    # Isolate the ensure_library_in_rf_context import_library call region.
    assert "notify=True" not in src, (
        "library_manager must not pass notify= to Namespace.import_library "
        "(removed in RF 7.4)"
    )
