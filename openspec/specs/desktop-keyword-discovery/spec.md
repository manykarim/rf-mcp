# desktop-keyword-discovery Specification

## Purpose
TBD - created by archiving change desktop-mcp-workflow-correctness. Update Purpose after archive.
## Requirements
### Requirement: find_keywords surfaces PlatynUI desktop interaction keywords

The system SHALL return the PlatynUI desktop interaction keywords (e.g.
Pointer Click, Keyboard Type, Query, Get Attribute, and the window-management
keywords) from `find_keywords` for desktop sessions and PlatynUI-scoped
queries, so an agent can discover a workable desktop entry point instead of
zero matches.

#### Scenario: PlatynUI library listing returns keywords (catalog mode)
- **WHEN** `find_keywords(library_name="PlatynUI.BareMetal")` is called in a
  desktop session with no query
- **THEN** the result lists the PlatynUI interaction keywords (non-empty),
  including Pointer Click, Keyboard Type, Query, Get Attribute

#### Scenario: PlatynUI alias listing works
- **WHEN** `find_keywords(library_name="PlatynUI")` is called
- **THEN** it returns the same PlatynUI.BareMetal keywords (alias resolved)

#### Scenario: intent query surfaces a desktop entry point
- **WHEN** `find_keywords` is queried with a single intent term ("click",
  "type", "get window") in a desktop session
- **THEN** at least one PlatynUI desktop keyword is returned

#### Scenario: natural-language catalog query does not silently return zero
- **WHEN** a multi-word natural-language query (e.g. "get window find element ui
  tree") is used with the literal `catalog` strategy
- **THEN** the system either returns matches via a documented semantic fallback
  OR returns zero with guidance that `catalog` is a literal filter and how to
  list by library / use semantic search — it does not strand the agent

### Requirement: Discovery guidance points to PlatynUI for desktop sessions

The system SHALL guide discovery toward PlatynUI desktop keywords (rather than
web/mobile keywords) when the session is a desktop session.

#### Scenario: desktop session discovery guidance
- **WHEN** discovery runs in a desktop session
- **THEN** the guidance/results prioritize PlatynUI desktop keywords over
  web/mobile keywords

