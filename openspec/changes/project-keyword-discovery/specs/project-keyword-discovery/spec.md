# Spec: project-keyword-discovery

## ADDED Requirements

### Requirement: Keywords a session can execute are keywords a session can discover
Any keyword that `execute_step` can run in a session SHALL be discoverable by
`find_keywords` and describable by `get_keyword_info` in that same session. This includes
keywords from a custom Python library imported into the session and user keywords defined
in an imported Robot Framework resource file. Discovery and execution SHALL NOT disagree
about whether a keyword exists.

#### Scenario: a custom library's keywords are findable
- **WHEN** a session imports a custom Python library installed in the project's environment, and a query names one of its keywords
- **THEN** `find_keywords` returns that keyword, attributed to that library

#### Scenario: a resource file's user keywords are findable
- **WHEN** a session imports a project resource file and a query names one of its user keywords
- **THEN** `find_keywords` returns that keyword, attributed to that resource

#### Scenario: an executable keyword is always describable
- **WHEN** `execute_step` can run a keyword in a session
- **THEN** `get_keyword_info` for that keyword in that session succeeds rather than reporting that it was not found

### Requirement: Described metadata matches what execution accepts
`get_keyword_info` for a project keyword SHALL report its argument names, default values,
declared types where the source provides them, and its documentation, consistent with what
`execute_step` accepts for that keyword.

#### Scenario: arguments and defaults are reported for a resource keyword
- **WHEN** `get_keyword_info` describes a resource user keyword declaring an argument with a default value
- **THEN** the reported arguments include that argument and its default

#### Scenario: type hints are reported for a custom library keyword
- **WHEN** `get_keyword_info` describes a custom Python library keyword whose arguments carry type hints
- **THEN** the reported arguments include those types

### Requirement: Project keywords are scoped to the session that imported them
Registration of a session's imported libraries and resources SHALL be scoped to that
session. A keyword registered by one session SHALL NOT appear in another session's
discovery results, so that concurrent sessions against different projects do not see each
other's keywords.

#### Scenario: one session's project keywords do not leak into another
- **WHEN** one session imports a project resource and a second session, which has not imported it, queries for one of its keywords
- **THEN** the second session's results do not include that keyword

#### Scenario: bundled keywords remain available to every session
- **WHEN** a session registers project keywords
- **THEN** the keywords of rf-mcp's own libraries remain discoverable in that session and in others

### Requirement: Discovery can be scoped to a project library or resource
The `library_name` filter SHALL accept a session-imported custom library or resource and
return that source's keywords. With strict scoping requested, results SHALL contain only
keywords from the named source.

#### Scenario: filtering by a custom library returns its keywords
- **WHEN** `find_keywords` is called with a query and `library_name` naming a session-imported custom library
- **THEN** the results contain that library's matching keywords, not only keywords of rf-mcp's bundled libraries

#### Scenario: strict scoping excludes other sources
- **WHEN** `find_keywords` is called with strict scoping and `library_name` naming an imported resource
- **THEN** every returned keyword belongs to that resource

### Requirement: An exact keyword name returns that keyword
A query that exactly names a keyword available in the session SHALL return that keyword
among its results.

#### Scenario: exact name query returns the project keyword
- **WHEN** `find_keywords` is queried with the exact name of a keyword the session has imported
- **THEN** that keyword appears in the results, rather than only similarly-named keywords from other libraries

### Requirement: A discovery miss explains what to import
When a keyword cannot be found, the reported message SHALL distinguish a keyword that
exists nowhere from one whose defining library or resource has not been imported into the
session, and SHALL name the call that would make it available.

#### Scenario: an unimported source is named
- **WHEN** `get_keyword_info` is asked about a keyword whose library or resource the session has not imported
- **THEN** the message says the defining source is not imported in this session and names the import call to make it available

#### Scenario: a genuine miss stays a genuine miss
- **WHEN** a keyword exists in no imported or bundled source
- **THEN** the message reports that the keyword was not found, without claiming an import would fix it
