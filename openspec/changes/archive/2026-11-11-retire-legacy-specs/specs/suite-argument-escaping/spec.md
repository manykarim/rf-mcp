## REMOVED Requirements

### Requirement: Windows drive-letter paths are written in a platform-independent form
**Reason**: The 0.x tool surface this requirement governed (`execute_step`, `execute_batch`, `build_test_suite`, `find_keywords`, `get_locator_guidance`, tool profiles, instruction templates and in-process keyword execution) was removed in rf-mcp 2.0.

**Migration**: The robotcode-backed tools replace it: see `robotcode-mcp-tools` for the tools, `robotcode-error-hinting` for failure guidance and `mcp-skills-provider` for the skills that carry the cookbooks and guidance. `docs/RELEASE_NOTES_v2.0.0.md` maps the 0.x tools to their replacements.

### Requirement: Backslash literals round-trip through RF parsing (escape-aware)
**Reason**: The 0.x tool surface this requirement governed (`execute_step`, `execute_batch`, `build_test_suite`, `find_keywords`, `get_locator_guidance`, tool profiles, instruction templates and in-process keyword execution) was removed in rf-mcp 2.0.

**Migration**: The robotcode-backed tools replace it: see `robotcode-mcp-tools` for the tools, `robotcode-error-hinting` for failure guidance and `mcp-skills-provider` for the skills that carry the cookbooks and guidance. `docs/RELEASE_NOTES_v2.0.0.md` maps the 0.x tools to their replacements.
