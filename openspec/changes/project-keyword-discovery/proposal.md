# Proposal: project-keyword-discovery

## Why

rf-mcp can **execute** a project's own keywords but cannot **find or describe** them.

Measured against a project whose custom keyword library is pip-installed into its own
`.venv`, plus a `.resource` file of user keywords, driven through a real MCP client on an
in-project installed rf-mcp (evidence:
`experiments/project_keywords_discovery_evidence.md`):

| capability | custom Python library (in `.venv`) | user keywords in `.resource` |
|---|---|---|
| import (session init / `import_library` / `import_resource`) | works | works |
| `execute_step` with arguments, return value, `assign_to` | works | works |
| `build_test_suite` emits correct imports; suite passes under plain `robot` | works | works |
| **`find_keywords`** | **never returned** | **never returned** |
| **`get_keyword_info`** | **"not found in any loaded library"** | same |

```
execute_step  Acme Add 2 40               -> '42'            assigned {'${total}': 42}
execute_step  Acme Login rf-mcp           -> 'hello rf-mcp'  assigned {'${greet}': 'hello rf-mcp'}

find_keywords(query="Acme Add")           -> 336 matches, top: Close Window (SeleniumLibrary, 0.75)
get_keyword_info("Acme Add")              -> {"success": false,
                                              "error": "Keyword 'Acme Add' not found in any loaded library"}
```

No parameter combination surfaces them: `library_name="AcmeLibrary"` returns 169
Browser/SeleniumLibrary keywords, `strict_library=True` returns 0, `strategy="exact"` on
the exact name returns 0, `context="generic"` returns only BuiltIn.

### Why this matters

`find_keywords`' own docstring says *"ALWAYS before calling execute_step with an unfamiliar
keyword"*. For a user's existing Robot Framework project, the keywords an agent most needs
are precisely the ones discovery cannot see. An agent following the documented workflow
concludes the project's domain keywords do not exist and re-implements them from raw
Browser/SeleniumLibrary calls - the opposite of what a project with a curated keyword layer
wants. The keywords are executable the whole time; the agent just has to already know their
names, which is exactly what discovery exists to avoid.

### Root cause

Two different indexes, and the tools read the one that does not have the project's keywords.

- The in-process **keyword cache** DOES receive custom-library keywords. Verified directly:
  after `load_session_libraries(["BuiltIn","AcmeLibrary"], ...)` the cache contains
  `acme add`, `acme greet`, `acmelibrary.acme add`. The session's search order contains the
  library too (`old_search_order: ["BuiltIn","AcmeLibrary"]`).
- The **discovery tools** do not read that cache. `get_keyword_info` resolves through a
  **LibDoc-backed store** and returns "not found in any loaded library" when the keyword is
  absent from it (`execution_coordinator.py:1134`), with an inspection-based fallback below
  it. `find_keywords` likewise ranks over rf-mcp's own catalogue. The decisive tell: a
  session initialised with `libraries=["BuiltIn","AcmeLibrary"]` returned matches
  exclusively from **Browser and SeleniumLibrary** - libraries that were never in that
  session but ARE in rf-mcp's installation.

Resource files compound it: they are not libraries at all, so they have no path into a
library-shaped index by construction, even though RF's namespace resolves their keywords
for execution.

### The fix is well-founded

Robot Framework's own LibDoc already produces exactly the needed metadata for **both**
kinds, including type hints and defaults - verified:

```
LibraryDocumentation("AcmeLibrary")                -> name='AcmeLibrary' type='LIBRARY'  keywords=3
    'Acme Add'   args=['a: int', 'b: int']   doc='Add two integers and return the sum.'
LibraryDocumentation("resources/acme_kw.resource") -> name='acme_kw'    type='RESOURCE' keywords=2
    'Acme Login' args=['user', 'password=secret']  doc='High-level user keyword built on ...'
```

rf-mcp already imports resources through RF's `Importer`
(`rf_native_context_manager.py:1376` `namespace.import_resource`), and `ResourceFile`
exposes `keywords` with `name`, `doc`, `args` and `source`. Nothing new has to be invented.

## What Changes

- **Session-imported libraries enter the discovery index.** When a library is imported into
  a session - at `init` via `libraries`, or later via `import_library` - its keywords become
  findable by `find_keywords` and describable by `get_keyword_info`, using LibDoc metadata
  so arguments, defaults, type hints and documentation match what execution accepts.
- **Resource files enter the discovery index.** `import_resource` registers the resource's
  user keywords for discovery, attributed to the resource rather than to a library, so an
  agent can find a project's high-level keyword layer.
- **Registration is session-scoped.** The keyword cache is process-global; one session's
  project keywords MUST NOT leak into another session's discovery results. Two sessions
  against different projects must each see only their own.
- **`library_name` and `strict_library` work for project keywords**, so an agent can scope
  discovery to a project's own library or resource.
- **Misses become actionable.** When a keyword is not found, the message distinguishes "no
  such keyword anywhere" from "this session has not imported the library or resource that
  defines it", and names the import call that would fix it.

## Non-Goals

- Scanning a project directory for resources or libraries that have NOT been imported.
  Discovery follows what the session declares; auto-discovery of the filesystem is a
  separate question with its own scoping and safety trade-offs.
- Changing execution. Import and `execute_step` already work correctly for both kinds and
  must not regress.
- The semantic ranker's quality. Making project keywords *present* in results is this
  change; how they are ranked against bundled keywords is only in scope to the extent that
  an exact-name query must return an exact match.
- The environment-fidelity defects (D1-D4) covered by `launch-env-fidelity`, and the
  `[all]`/pre-release packaging defects covered by `install-extra-resolvability`.
