# Tasks: project-keyword-discovery

## 1. Establish the failing baseline
- [x] 1.1 Port the evidence fixture into the test suite: a custom Python library and a `.resource` with documented, argument-taking user keywords
- [x] 1.2 Failing test: a session-imported custom library's keyword is absent from `find_keywords`
- [x] 1.3 Failing test: an imported resource's user keyword is absent from `find_keywords`
- [x] 1.4 Failing test: `get_keyword_info` reports "not found in any loaded library" for both
- [x] 1.5 Passing test (must stay passing): `execute_step` runs both, with arguments and `assign_to`

## 2. Register imported sources for discovery
- [x] 2.1 Add a session-scoped registry of imported sources keyed by (session, resolved path or library name)
- [x] 2.2 On `import_library` and on session `init` with a library not in rf-mcp's own catalogue, generate LibDoc and register its keywords
- [x] 2.3 On `import_resource`, generate LibDoc (`type='RESOURCE'`) and register its user keywords, attributed to the resource
- [x] 2.4 Cache by resolved source path + mtime; a re-import is cheap, an edited resource is re-read
- [x] 2.5 A source that fails to parse degrades discovery for that source only - reported, never failing the import or execution

## 3. Read the registry from the discovery tools
- [x] 3.1 `get_keyword_info` consults the session's registered sources before returning "not found"
- [x] 3.2 `find_keywords` includes the session's registered sources in its candidate set
- [x] 3.3 Report the defining source on each result: library name for a library, resource name for a resource
- [x] 3.4 Preserve the existing shadowing/ambiguity handling when a project keyword collides with a bundled one

## 4. Session scoping
- [x] 4.1 Lookup filters registered sources by the calling session
- [x] 4.2 Test: session A imports a resource; session B in the SAME process does not see its keywords
- [x] 4.3 Test: bundled keywords remain discoverable in both sessions
- [x] 4.4 Test: a session that imported nothing behaves exactly as before

## 5. Filters and exact matching
- [x] 5.1 `library_name` selects a session-imported custom library or resource
- [x] 5.2 `strict_library=True` returns only that source's keywords
- [x] 5.3 An exact-name query returns the exact match (today `query="Acme Add"` returns 336 results topped by `Close Window`)
- [x] 5.4 Check whether `library_name` constrains results for BUNDLED libraries either - it currently returns 169 unrelated keywords; if broken, raise it separately rather than widening this change

## 6. Actionable misses
- [x] 6.1 Distinguish "no such keyword anywhere" from "defining source not imported in this session"
- [x] 6.2 The latter names the import call that would make it available
- [x] 6.3 Tests for both message paths

## 7. Metadata fidelity
- [x] 7.1 `get_keyword_info` reports argument names, defaults and declared types from LibDoc
- [x] 7.2 Test: resource keyword with a default (`password=secret`) reports the default
- [x] 7.3 Test: custom library keyword with type hints (`a: int`) reports the types
- [x] 7.4 Test: documentation is carried through

## 8. Verification
- [x] 8.1 Re-run the evidence matrix; every `find_keywords`/`get_keyword_info` row must flip while execution rows stay passing
- [x] 8.2 Confirm `build_test_suite` output is unchanged (it is already correct - it imports the resource and relies on the resource's own `Library`)
- [x] 8.3 Full unit suite green
