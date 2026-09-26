"""Project keywords (a project's own libraries and resource files) across rf-mcp tools.

Every case below was a real failure found by driving the tools against a small
project: a custom library imported by path, a second one imported WITH ARGUMENTS and
an ALIAS, and a resource file. Before the fix:

- get_keyword_info could not find qualified names (``AcmeLib.Acme Add``), keywords
  scoped by ``library_name`` to a resource or alias, anything at all without
  ``session_id``, or a resource in ``mode="library"``
- a project keyword sharing its name with a keyword of a library the session never
  imported was reported "not available in this session's libraries"
- find_keywords gave every partial match a flat 0.7 (so "log in to acme" ranked
  `Acme Add` level with `Acme Login As`), ignored a ``library_name`` naming a
  project source, omitted project keywords from the catalog strategy, and excluded
  them entirely from any session with a web-library preference
- set_library_search_order silently dropped project libraries and still reported
  success, and the session order it kept (paths) never reached RF
- every step re-imported a path-imported library bare, so a library imported with
  args or an alias gained a SECOND instance with default configuration
- build_test_suite emitted ``Library  AcmeLib`` (unimportable), dropped import args
  and aliases, turned a resource-qualified keyword into ``Library  <resource>``, and
  stripped the prefix that disambiguated a colliding keyword - so run_test_suite
  failed a suite that had passed live
- check_library_availability called an importable project file "missing"
- execute_flow silently dropped a step's ``args`` (execute_batch accepts both keys)

Run: uv run pytest tests/unit/test_project_keywords_all_tools.py -q
"""
from __future__ import annotations

import asyncio
import json
import textwrap
import uuid
from pathlib import Path

import pytest

from robotmcp import server as S
from robotmcp.compat.fastmcp_compat import get_tool_fn

ACME_LIB = '''
class AcmeLib:
    ROBOT_LIBRARY_SCOPE = "GLOBAL"
    def acme_add(self, a, b):
        """Add two numbers the Acme way. Returns the sum."""
        return int(a) + int(b)
    def acme_status(self):
        """Report status from AcmeLib."""
        return "status-from-lib"
'''
ACME_CFG = '''
class AcmeCfg:
    ROBOT_LIBRARY_SCOPE = "GLOBAL"
    def __init__(self, env="dev", region="eu"):
        self.env, self.region = env, region
    def acme_environment(self):
        """Return the configured environment as env/region."""
        return f"{self.env}/{self.region}"
    def acme_status(self):
        """Report status from AcmeCfg."""
        return "status-from-cfg"
'''
ACME_RES = '''\
*** Keywords ***
Acme Login As
    [Documentation]    Log in to Acme as the given user.
    [Arguments]    ${user}    ${password}=secret
    RETURN    ${user}-token

Acme Double
    [Documentation]    Double a number.
    [Arguments]    ${n}
    ${r}=    Evaluate    int(${n}) * 2
    RETURN    ${r}
'''


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def call(tool, **kwargs):
    async def _go():
        try:
            return await get_tool_fn(getattr(S, tool))(**kwargs)
        except Exception as exc:  # ToolError on execution failure
            return {"success": False, "error": str(exc)}

    return run(_go())


def load(result):
    """find_keywords may externalise its payload to an artifact file."""
    res = result.get("result")
    if isinstance(res, str) and res.startswith("Content saved to "):
        path = res.split("Content saved to ", 1)[1].split(" (", 1)[0]
        return json.loads(Path(path).read_text())
    return res if isinstance(res, dict) else result


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    # MODULE-scoped on purpose: rf-mcp runs every session in ONE process-global RF
    # namespace, so importing a same-named resource from a fresh temp dir per test
    # would give RF N `acme_keywords` resources and make every call ambiguous.
    tmp_path = tmp_path_factory.mktemp("acme")
    (tmp_path / "libs").mkdir()
    (tmp_path / "resources").mkdir()
    lib = tmp_path / "libs" / "AcmeLib.py"
    cfg = tmp_path / "libs" / "AcmeCfg.py"
    res = tmp_path / "resources" / "acme_keywords.resource"
    lib.write_text(textwrap.dedent(ACME_LIB))
    cfg.write_text(textwrap.dedent(ACME_CFG))
    res.write_text(ACME_RES)
    return {"lib": str(lib), "cfg": str(cfg), "res": str(res)}


@pytest.fixture(scope="module")
def session(project):
    sid = f"proj-{uuid.uuid4().hex[:8]}"
    call("manage_session", action="init", session_id=sid, libraries=["BuiltIn"])
    assert call("manage_session", action="import_library", session_id=sid,
                library_name=project["lib"])["success"]
    assert call("manage_session", action="import_library", session_id=sid,
                library_name=project["cfg"], args=["prod", "us"], alias="Cfg")["success"]
    assert call("manage_session", action="import_resource", session_id=sid,
                resource_path=project["res"])["success"]
    return sid


# ============================================================================
# get_keyword_info
# ============================================================================


@pytest.mark.parametrize("name, expected", [
    ("AcmeLib.Acme Add", "Acme Add"),
    ("acme_keywords.Acme Double", "Acme Double"),
    ("Cfg.Acme Environment", "Acme Environment"),
])
def test_qualified_names_resolve(session, name, expected):
    r = call("get_keyword_info", keyword_name=name, session_id=session)
    assert r["success"], r
    assert r["keyword"]["name"] == expected


@pytest.mark.parametrize("library, keyword", [
    ("acme_keywords", "Acme Double"),   # resource by name
    ("Cfg", "Acme Environment"),        # library by alias
])
def test_library_name_scope_reaches_project_sources(session, library, keyword):
    r = call("get_keyword_info", keyword_name=keyword, library_name=library,
             session_id=session)
    assert r["success"], r
    assert r["keyword"]["name"] == keyword


def test_resource_by_path_as_library_name(session, project):
    r = call("get_keyword_info", keyword_name="Acme Double", library_name=project["res"],
             session_id=session)
    assert r["success"], r


def test_without_session_id_answers_with_the_session_it_lives_in(session):
    r = call("get_keyword_info", keyword_name="Acme Add")
    assert r["success"], r
    assert session in {m["session_id"] for m in r["matches"]}
    assert "session_id" in r["note"]


def test_ambiguous_name_lists_every_source(session):
    r = call("get_keyword_info", keyword_name="Acme Status", session_id=session)
    assert r["success"]
    assert {m["library"] for m in r["matches"]} == {"AcmeLib", "AcmeCfg"}


def test_library_mode_documents_a_resource_and_an_alias(session):
    res = call("get_keyword_info", mode="library", library_name="acme_keywords",
               session_id=session)
    assert res["success"], res
    assert res["library"]["keyword_count"] == 2
    cfg = call("get_keyword_info", mode="library", library_name="Cfg", session_id=session)
    assert cfg["success"], cfg
    assert cfg["library"]["import"]["args"] == ["prod", "us"]
    assert cfg["library"]["import"]["alias"] == "Cfg"


def test_project_keyword_is_not_hidden_by_a_same_named_catalogue_keyword(tmp_path):
    """A resource keyword named like a keyword of a library the session did NOT import
    used to be reported "not available in this session's libraries"."""
    from robotmcp.utils.rf_libdoc_integration import get_rf_doc_storage

    storage = get_rf_doc_storage()
    catalogue = storage.get_keywords_documentation_all("Go Back") or []
    if not catalogue:
        pytest.skip("no catalogue library defines 'Go Back' in this environment")
    res = tmp_path / "nav.resource"
    res.write_text("*** Keywords ***\nGo Back\n    [Documentation]    Project nav.\n    No Operation\n")
    sid = f"shadow-{uuid.uuid4().hex[:8]}"
    call("manage_session", action="init", session_id=sid, libraries=["BuiltIn"])
    call("manage_session", action="import_resource", session_id=sid, resource_path=str(res))

    r = call("get_keyword_info", keyword_name="Go Back", session_id=sid)

    assert r["success"], r
    assert any(m["library"] == "nav" for m in r["matches"])


# ============================================================================
# find_keywords
# ============================================================================


def test_graded_ranking_prefers_the_documented_match(session):
    d = load(call("find_keywords", query="log in to acme", session_id=session))
    assert d["matches"][0]["keyword_name"] == "Acme Login As", d["matches"][:3]


def test_scope_to_a_project_source_is_strict(session):
    d = load(call("find_keywords", query="double", library_name="acme_keywords",
                  session_id=session))
    assert {m["library"] for m in d["matches"]} == {"acme_keywords"}


def test_catalog_includes_project_and_keeps_everything_else(session):
    scoped = call("find_keywords", query="acme", strategy="catalog", session_id=session)
    assert scoped["match_count"] >= 5, scoped
    everything = call("find_keywords", query="", strategy="catalog", session_id=session)
    libs = {m.get("library") for m in everything.get("results", [])}
    assert "BuiltIn" in libs, "project entries must be ADDED, not replace the catalogue"


def test_web_preference_session_still_sees_project_keywords(session):
    sess = S.execution_engine.session_manager.get_session(session)
    sess.explicit_library_preference = "SeleniumLibrary"
    d = load(call("find_keywords", query="log in to acme", session_id=session))
    names = [m.get("keyword_name") for m in d.get("matches", [])]
    assert "Acme Login As" in names, names


def test_scorer_stem_matching_is_conservative():
    from robotmcp.utils.rf_libdoc_integration import RFKeywordInfo

    def kw(name, doc=""):
        return RFKeywordInfo(name=name, library="L", doc=doc)

    class _Store:
        def project_keywords(self, *_a, **_k):
            return [kw("Enter Address"), kw("Acme Login As", "Log in to Acme.")]

    import robotmcp.utils.rf_libdoc_integration as m
    orig = m.get_rf_doc_storage
    m.get_rf_doc_storage = lambda: _Store()
    try:
        hits = {h["keyword_name"]: h["confidence"]
                for h in S._project_keyword_matches_for_query("add", "sid")}
        login = S._project_keyword_matches_for_query("log", "sid")
    finally:
        m.get_rf_doc_storage = orig
    assert "Enter Address" not in hits, "'add' must not stem-match 'address'"
    assert login and login[0]["keyword_name"] == "Acme Login As", "'log' ~ 'login'"


# ============================================================================
# execution, search order, flow
# ============================================================================


def test_aliased_library_with_args_has_one_correctly_configured_instance(session):
    for _ in range(3):  # the bare re-import used to happen on EVERY step
        r = call("execute_step", keyword="Acme Environment", session_id=session)
        assert r.get("output") == "prod/us", r
    r = call("execute_step", keyword="AcmeCfg.Acme Environment", session_id=session)
    assert not r.get("success"), "no second, unaliased instance may exist"


def test_search_order_accepts_project_sources_and_reaches_rf(session):
    so = call("set_library_search_order", libraries=["Cfg", "AcmeLib", "BuiltIn"],
              session_id=session)
    assert so["success"] and "Cfg" in so["libraries_applied"], so
    assert call("execute_step", keyword="Acme Status", session_id=session)["output"] == "status-from-cfg"

    call("set_library_search_order", libraries=["AcmeLib", "Cfg", "BuiltIn"], session_id=session)
    assert call("execute_step", keyword="Acme Status", session_id=session)["output"] == "status-from-lib"


def test_search_order_reports_rejects_instead_of_dropping_them(session):
    so = call("set_library_search_order", libraries=["NoSuchLib", "BuiltIn"],
              session_id=session)
    assert so["success"] is False
    assert so["libraries_rejected"] == ["NoSuchLib"]


def test_execute_flow_accepts_args_like_execute_batch(session):
    r = call("execute_flow", structure="for_each", session_id=session, items=["3"],
             item_var="n", then_steps=[{"keyword": "Acme Double", "args": ["${n}"]}])
    assert r["success"], r


# ============================================================================
# availability, state
# ============================================================================


def test_availability_of_project_files(project):
    r = call("check_library_availability", libraries=[project["lib"], project["res"]])
    assert set(r["available_libraries"]) >= {project["lib"], project["res"]}
    assert r["success"]


def test_session_state_lists_project_sources(session):
    r = call("get_session_state", session_id=session, sections=["libraries"])
    ps = {p["name"]: p for p in r["sections"]["libraries"]["project_sources"]}
    assert set(ps) >= {"AcmeLib", "Cfg", "acme_keywords"}
    assert ps["Cfg"]["args"] == ["prod", "us"] and ps["Cfg"]["alias"] == "Cfg"


# ============================================================================
# suite round trip
# ============================================================================


def test_generated_suite_replays_exactly_what_ran(session):
    call("set_library_search_order", libraries=["Cfg", "AcmeLib", "BuiltIn"], session_id=session)
    call("execute_step", keyword="Acme Status", session_id=session)            # -> Cfg
    call("set_library_search_order", libraries=["AcmeLib", "Cfg", "BuiltIn"], session_id=session)
    call("execute_step", keyword="Acme Status", session_id=session)            # -> AcmeLib
    call("execute_step", keyword="Acme Environment", session_id=session)
    call("execute_step", keyword="Acme Add", arguments=["2", "3"], session_id=session)
    call("execute_step", keyword="acme_keywords.Acme Double", arguments=["5"], session_id=session)

    rf = call("build_test_suite", session_id=session, test_name="Acme Test")["rf_text"]
    flat = rf.replace("${/}", "/")

    assert "AcmeLib.py" in flat, "project library imported by path, not bare name"
    assert "AcmeCfg.py    prod    us    AS    Cfg" in flat, "args and alias preserved"
    assert "Library         acme_keywords" not in rf, "resource must be a Resource import"
    assert "acme_keywords.resource" in flat
    # load-bearing prefixes kept, in execution order; unambiguous ones stripped
    body = rf.split("*** Test Cases ***", 1)[1]
    assert body.index("Cfg.Acme Status") < body.index("AcmeLib.Acme Status")
    assert "Acme Environment" in body and "Cfg.Acme Environment" not in body
    assert "    Acme Double    5" in body

    full = call("run_test_suite", session_id=session, mode="full")
    assert full["success"], json.dumps(full, default=str)[:600]
