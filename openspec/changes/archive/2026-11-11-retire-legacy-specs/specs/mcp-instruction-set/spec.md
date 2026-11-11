## REMOVED Requirements

### Requirement: The default MCP instruction set is a lean, order-explicit spine
**Reason**: The 0.x tool surface this requirement governed (`execute_step`, `execute_batch`, `build_test_suite`, `find_keywords`, `get_locator_guidance`, tool profiles, instruction templates and in-process keyword execution) was removed in rf-mcp 2.0.

**Migration**: The robotcode-backed tools replace it: see `robotcode-mcp-tools` for the tools, `robotcode-error-hinting` for failure guidance and `mcp-skills-provider` for the skills that carry the cookbooks and guidance. `docs/RELEASE_NOTES_v2.0.0.md` maps the 0.x tools to their replacements.

### Requirement: A single canonical first tool and unified session entry
**Reason**: The 0.x tool surface this requirement governed (`execute_step`, `execute_batch`, `build_test_suite`, `find_keywords`, `get_locator_guidance`, tool profiles, instruction templates and in-process keyword execution) was removed in rf-mcp 2.0.

**Migration**: The robotcode-backed tools replace it: see `robotcode-mcp-tools` for the tools, `robotcode-error-hinting` for failure guidance and `mcp-skills-provider` for the skills that carry the cookbooks and guidance. `docs/RELEASE_NOTES_v2.0.0.md` maps the 0.x tools to their replacements.

### Requirement: Verbose and dead instruction templates are retired
**Reason**: The 0.x tool surface this requirement governed (`execute_step`, `execute_batch`, `build_test_suite`, `find_keywords`, `get_locator_guidance`, tool profiles, instruction templates and in-process keyword execution) was removed in rf-mcp 2.0.

**Migration**: The robotcode-backed tools replace it: see `robotcode-mcp-tools` for the tools, `robotcode-error-hinting` for failure guidance and `mcp-skills-provider` for the skills that carry the cookbooks and guidance. `docs/RELEASE_NOTES_v2.0.0.md` maps the 0.x tools to their replacements.

### Requirement: Guaranteed init-response guidance for instruction-sensitive libraries
**Reason**: The 0.x tool surface this requirement governed (`execute_step`, `execute_batch`, `build_test_suite`, `find_keywords`, `get_locator_guidance`, tool profiles, instruction templates and in-process keyword execution) was removed in rf-mcp 2.0.

**Migration**: The robotcode-backed tools replace it: see `robotcode-mcp-tools` for the tools, `robotcode-error-hinting` for failure guidance and `mcp-skills-provider` for the skills that carry the cookbooks and guidance. `docs/RELEASE_NOTES_v2.0.0.md` maps the 0.x tools to their replacements.

### Requirement: Tool docstrings are concise with the load-bearing guidance first
**Reason**: The 0.x tool surface this requirement governed (`execute_step`, `execute_batch`, `build_test_suite`, `find_keywords`, `get_locator_guidance`, tool profiles, instruction templates and in-process keyword execution) was removed in rf-mcp 2.0.

**Migration**: The robotcode-backed tools replace it: see `robotcode-mcp-tools` for the tools, `robotcode-error-hinting` for failure guidance and `mcp-skills-provider` for the skills that carry the cookbooks and guidance. `docs/RELEASE_NOTES_v2.0.0.md` maps the 0.x tools to their replacements.
