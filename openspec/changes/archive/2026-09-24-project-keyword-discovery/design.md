# Design: project-keyword-discovery

## Two indexes, and the tools read the wrong one

The surprising part of this defect is that the keyword cache is NOT the problem. Measured
directly in-process:

```python
d.library_manager.load_session_libraries(["BuiltIn","AcmeLibrary"], d.keyword_discovery)
# keyword_cache now contains: 'acme add', 'acme greet', 'acmelibrary.acme add', ...
```

And the session genuinely carries the library (`old_search_order: ["BuiltIn","AcmeLibrary"]`).
Yet `find_keywords` and `get_keyword_info` return nothing for it.

```
    session namespace                  keyword_cache                LibDoc-backed store
    (execution reads this)             (populated, unread)          (discovery reads this)
         |                                   |                              |
   custom lib   YES                    custom lib   YES              custom lib   NO
   resource kw  YES                    resource kw  NO               resource kw  NO
         |                                                                  |
    execute_step works  <----------- the disagreement -----------> find_keywords/get_keyword_info fail
```

So the change is not "teach rf-mcp about custom libraries" - it already knows - but
**"make the discovery tools read a source that includes what the session imported."**

`get_keyword_info` resolves through the LibDoc store and falls through to
`"not found in any loaded library"` (`execution_coordinator.py:1134`); an inspection-based
fallback exists below it but is not reached for this case. `find_keywords` ranks over
rf-mcp's own catalogue - proven by a session declaring only `["BuiltIn","AcmeLibrary"]`
returning matches exclusively from Browser and SeleniumLibrary.

## Why LibDoc is the right source

Robot Framework's own LibDoc already produces the exact metadata shape for BOTH kinds, and
the discovery store is already LibDoc-shaped, so project keywords can be registered without
inventing a parallel representation:

```
LibraryDocumentation("AcmeLibrary")                 type='LIBRARY'   3 keywords, args with type hints
LibraryDocumentation("resources/acme_kw.resource")  type='RESOURCE'  2 keywords, args with defaults
```

This matters for the second requirement: `args=['a: int', 'b: int']` and
`args=['user', 'password=secret']` come straight from LibDoc, so described metadata cannot
drift from what execution accepts.

Alternatives considered:

| option | verdict |
|---|---|
| Point discovery at the live `keyword_cache` | **Partial.** Fixes custom libraries but not resources, which never enter the cache because they are not libraries. Would also change ranking inputs wholesale - a much larger blast radius than the defect warrants. |
| Read RF's `ResourceFile.keywords` directly | Workable for resources (`name`/`doc`/`args`/`source` are all exposed and rf-mcp already holds the object), but produces a second metadata shape to maintain alongside LibDoc. Prefer LibDoc for both. |
| **Register LibDoc output for whatever the session imports** | **Chosen.** One source, both kinds, already the shape the discovery store expects, and it reuses RF's own parsing rather than rf-mcp re-deriving signatures. |

## Session scoping is a correctness requirement, not polish

The keyword cache is **process-global** - one `DynamicKeywordDiscovery` shared by every
session. Registering a project's keywords globally would mean a user working on two
projects in two sessions sees project A's domain keywords while driving project B, and
`execute_step` would then fail on a keyword discovery had just advertised - re-creating the
very discovery/execution disagreement this change removes, in the opposite direction.

So registration must be keyed by session, and lookup must filter by the calling session.
Bundled keywords stay shared; only imported project sources are partitioned.

## Cost

LibDoc parsing is not free, and `import_resource` / `import_library` sit on an interactive
path. Generate once per (session, source) at import time and cache by resolved source path
plus mtime, so a re-import is cheap and an edited resource is re-read. A resource that
fails to parse must not fail the import - execution would still work, so a parse failure
degrades discovery for that source and is reported, rather than breaking the session.

## Ranking

Making project keywords *present* is the requirement; re-tuning the semantic ranker is not.
One exception is specified deliberately: an exact-name query must return the exact match.
Today `find_keywords(query="Acme Add")` returns 336 results topped by `Close Window`
(0.75) - a project keyword losing to an unrelated bundled one on its own exact name is
indefensible regardless of ranker design.

A related observation, deliberately left out of scope: `library_name="AcmeLibrary"` today
returns 169 Browser/SeleniumLibrary keywords - the filter does not appear to constrain the
result set at all. The spec requires that filter to work for project sources; whether it is
also broken for bundled libraries should be checked during implementation and, if so,
raised separately.

## Evidence

`experiments/project_keywords_discovery_evidence.md` - controlled matrix with a **fresh
server process per row** (the cache is process-global, so a shared process cross-
contaminates), inspecting only `matches[].keyword_name`. An earlier pass that grepped the
raw artifact produced a false positive: `find_keywords` echoes the query in
`action_description`, so a query containing a keyword's name matched itself. Untracked -
`experiments/` is git-excluded via `.git/info/exclude:17`.
