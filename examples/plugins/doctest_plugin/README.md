# DocTestLibrary Plugin Example

This directory contains **four** example plugins (Visual, Pdf, PrintJob, Ai) that publish metadata for [`robotframework-doctestlibrary`](https://github.com/manykarim/robotframework-doctestlibrary) so rf-mcp can recommend and load each sub-library individually.

## Highlights

- Registers DocTest’s sub-libraries (`DocTest.VisualTest`, `.PdfTest`, `.PrintJobTest`, `.Ai`).
- Advertises common keywords (`Compare Images`, `Compare Pdf Documents`, `Get Text From Document`, …).
- Provides installation hints (core package and optional `[ai]` extra) and usage guidance.
- Sets up default variables for watermarks / diff toggles in `on_session_start`.

## Usage

1. Choose the sub-library you want (`DocTest.VisualTest`, `.PdfTest`, `.PrintJobTest`, `.Ai`) and copy its manifest into your workspace:

   ```bash
   mkdir -p .robotmcp/plugins
   cp examples/plugins/doctest_plugin/visual_manifest.json .robotmcp/plugins/doctest_visual.json
   ```

   Available manifests:

   - `visual_manifest.json`
   - `pdf_manifest.json`
   - `print_manifest.json`
   - `ai_manifest.json`

2. Install the core package and any extras your tests require:

   ```bash
   pip install robotframework-doctestlibrary
   # Optional LLM helpers
   pip install "robotframework-doctestlibrary[ai]"
   ```

   Ensure system binaries such as ImageMagick, Tesseract, Ghostscript/GhostPCL, GhostPCL are installed (see project README).

3. Start `rf-mcp` and confirm the specific library is available:

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

## References

- Project repository: [manykarim/robotframework-doctestlibrary](https://github.com/manykarim/robotframework-doctestlibrary)
- Keyword documentation: <https://manykarim.github.io/robotframework-doctestlibrary/>
- Presentation: [DocTest Library at RoboCon 2021](https://youtu.be/qmpwlQoJ-nE)
