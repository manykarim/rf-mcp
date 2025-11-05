# DocTestLibrary Plugin Example

This directory contains **four** example plugins (Visual, Pdf, PrintJob, Ai) that publish metadata for [`robotframework-doctestlibrary`](https://github.com/manykarim/robotframework-doctestlibrary) so rf-mcp can recommend and load each sub-library individually.

## Highlights

- Registers DocTest’s sub-libraries (`DocTest.VisualTest`, `.PdfTest`, `.PrintJobTests`, `.Ai`).
- Advertises common keywords (`Compare Images`, `Compare Pdf Documents`, `Get Text From Document`, …).
- Provides installation hints (core package and optional `[ai]` extra) and usage guidance.
- Captures comparison artefacts and LLM diagnostics so `get_application_state` can return rich state to agents.

## Usage

1. Install the example package (so Python can import the modules and entry points are registered):

   ```bash
   uv pip install --editable examples/plugins/doctest_plugin
   # or: pip install -e examples/plugins/doctest_plugin
   ```

   The package is published under the module name `rfmcp_doctest_plugin`; once installed in the same virtual environment as `rf-mcp` the entry point group `robotmcp.library_plugins` is populated automatically. (For uv-managed environments set `UV_PROJECT_ENVIRONMENT` to the target venv before running the install command.)

   If you prefer manifest-based discovery instead, copy the manifest for the sub-library you want (`DocTest.VisualTest`, `.PdfTest`, `.PrintJobTests`, `.Ai`) into your workspace:

   ```bash
   mkdir -p .robotmcp/plugins
   cp examples/plugins/doctest_plugin/visual_manifest.json .robotmcp/plugins/doctest_visual.json
   ```

   Available manifests:

   - `visual_manifest.json`
   - `pdf_manifest.json`
   - `print_manifest.json`
   - `ai_manifest.json`

   When using manifests, ensure the `module` field points at `rfmcp_doctest_plugin.<module>`—the distribution no longer requires tweaking `PYTHONPATH`.

2. Install the DocTest library (and optional AI extras) along with the required native binaries:

   ```bash
   pip install robotframework-doctestlibrary
   # Optional LLM helpers
   pip install "robotframework-doctestlibrary[ai]"
   ```

   Ensure system binaries such as ImageMagick, Tesseract, Ghostscript/GhostPCL, GhostPCL are installed (see project README).

3. Start `rf-mcp` and confirm the specific library is available (or run the reload tool after installation):

   ```python
   from robotmcp.config import library_registry
   libs = library_registry.get_all_libraries()
   print("DocTest.VisualTest" in libs)
   ```

4. Import the desired module in your Robot suite, e.g.:

   ```RobotFramework
   *** Settings ***
   Library    DocTest.VisualTest    show_diff=${True}

   *** Test Cases ***
   Highlight Differences
       Compare Images    Reference.png    Candidate.png
   ```

## State Providers & Overrides

- **Visual** (`rfmcp_doctest_plugin.visual`): wraps `Compare Images` to persist diff artefacts and exposes them through `get_application_state`.
- **Pdf** (`rfmcp_doctest_plugin.pdf`): captures LLM-assisted comparison output, the selected comparison facets, and optional diff JSON files.
- **PrintJob** (`rfmcp_doctest_plugin.print_jobs`): records spool metadata differences and property mismatches for PostScript/PCL files.
- **Ai** (`rfmcp_doctest_plugin.ai`): surfaces the latest LLM response/decision so agents can troubleshoot prompt failures.

Each module is covered by `tests/unit/test_doctest_plugins.py`, illustrating how to unit-test overrides with monkeypatched `BuiltIn` stubs.

## Troubleshooting

- `ModuleNotFoundError: No module named 'rfmcp_doctest_plugin'` → install the package inside the same virtual environment as `rf-mcp` (for example `uv pip install --editable examples/plugins/doctest_plugin`).
- `Failed to load plugin from entry point` → confirm the installation succeeded by running `uv run python -c "import importlib.metadata as im; print([e.name for e in im.entry_points(group='robotmcp.library_plugins')])"`.
- Using manifests alongside the package is optional; once the entry point is installed you do **not** need to copy the JSON manifests unless you want to override metadata locally.

## References

- Project repository: [manykarim/robotframework-doctestlibrary](https://github.com/manykarim/robotframework-doctestlibrary)
- Keyword documentation: <https://manykarim.github.io/robotframework-doctestlibrary/>
- Presentation: [DocTest Library at RoboCon 2021](https://youtu.be/qmpwlQoJ-nE)
