"""FastMCP 3.x server wiring the 8 `rf_*` tools, Skills over MCP, HintEngine, and OpenTelemetry."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncIterator, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from robotmcp.client_roots import ClientRoots
from robotmcp.args import ArgumentAliases, PlainOptionalSchemas, Shell, StrDict, StrList, Target, tolerant_literal
from robotmcp.config import ServerConfig
from robotmcp.hints import HintContext, HintEngine
from robotmcp.runner import (
    CliRunner,
    DebugAction,
    ReplAction,
    SessionRunner,
    read_project_files,
    sync_inline_files,
)
from robotmcp.runner.cli_runner import (
    ANALYZE_CLEAN_NOTE,
    DiscoverAction,
    DocAction,
    ResultsAction,
    decode_analyze_exit_code,
    project_discover_data,
    run_outcome,
)
from robotmcp.skills import build_server_instructions, mount_skills_provider
from robotmcp.skills.pointers import SkillPointers
from robotmcp.telemetry import record_operation

# Shared parameter types: every tool uses the same names for the same things.
Root = Annotated[str | None, Field(description="Project root (folder with robot.toml); default: --root, the client's workspace folder, or the server's working folder walked up to robot.toml.")]
Profile = Annotated[str | None, Field(description="robot.toml profile to apply (-p).")]
TargetArg = Annotated[Target | None, Field(description="Override the execution mode: 'server' runs robotcode here, 'client' returns a command for your shell.")]
ShellArg = Annotated[Shell | None, Field(description="Shell syntax of client-mode commands.")]
IncludeTags = Annotated[StrList | None, Field(description="Only tests with these tags (-i); a list such as ['smoke'].")]
ExcludeTags = Annotated[StrList | None, Field(description="Skip tests with these tags (-e).")]
ByLongname = Annotated[StrList | None, Field(description="Exact test/suite longnames from rf_discover, e.g. ['Tests.Login.Valid Login'].")]
ByMetadata = Annotated[StrList | None, Field(description="Only tests with this test metadata, 'Name:Value' (-btm).")]
ExcludeByMetadata = Annotated[StrList | None, Field(description="Skip tests with this test metadata, 'Name:Value' (-ebtm).")]
DiscoverToolAction = Literal[DiscoverAction, "read"]
Paths = Annotated[StrList | None, Field(description="Files or folders relative to root; default: the robot.toml `paths`.")]
Files = Annotated[StrDict | None, Field(description="Server mode: {relative path: content} written into the root before the call (remote runners).")]
Variables = Annotated[StrList | None, Field(description="Variables 'NAME:value' (-v).")]
TimeoutSeconds = Annotated[int | None, Field(description="Seconds to wait for this call before returning status 'running'.")]
IdleTimeout = Annotated[int | None, Field(description="Minutes of inactivity after which the session ends itself.")]
SessionId = Annotated[str | None, Field(description="Session id returned by action='start'.")]
StopScope = Annotated[
    tolerant_literal(Literal["id", "project", "all"]),
    Field(description="For action='stop': this session ('id'), all sessions of the project, or all."),
]

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
WRITES_FILES = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)
RUNS_TESTS = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True)
MAY_OVERWRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=True)

_RESULT_FILE_SUFFIXES = (".xml", ".json")


def _reject_result_files(**values: Any) -> None:
    """Raise when a result file is passed anywhere but `output_path` / `baseline` / `merge_inputs`."""
    for name, value in values.items():
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, str) and item.lower().endswith(_RESULT_FILE_SUFFIXES):
                raise ToolError(
                    f"`{name}` got the result file '{item}'. Pass result files only as `output_path` "
                    "(or `baseline` for action='diff', `merge_inputs` for action='merge')."
                )


class FailureStreaks:
    """Count, per test, the rf_run calls in a row it failed in; at 3 (then every 3rd) say to debug instead.

    2026-11-07 Codex/GPT-5.6-Luna (demoshop T1) edited and reran one failing test 14 times without opening
    rf_debug; the "retry once" rule lives in the server instructions, which Codex never shows the model.
    """

    def __init__(self) -> None:
        self.streaks: dict[str, int] = {}

    def update(self, failures: list[str], exit_code: int | None) -> str | None:
        if exit_code == 0 and not failures:
            self.streaks.clear()
            return None
        names = [f.split(": ", 1)[0] for f in failures]
        self.streaks = {n: self.streaks.get(n, 0) + 1 for n in names}
        stuck = [n for n, k in self.streaks.items() if k >= 3 and k % 3 == 0]
        if not stuck:
            return None
        return (f"{', '.join(stuck)} failed in {self.streaks[stuck[0]]} runs in a row. Stop editing and rerunning: "
                "rf_debug action='start' with its longname pauses at the failure, so you can inspect the page or "
                "variables (.vars --user, Get Aria Snapshot) before the next fix.")


def create_server(config: ServerConfig | None = None) -> FastMCP:
    """Create and configure the FastMCP server instance for `rf-mcp`."""
    cfg = config or ServerConfig.from_env_and_args(argv=[])
    hint_engine = HintEngine(hints_file=cfg.resolve_hints_file())
    cli_runner = CliRunner(cfg, hint_engine=hint_engine)
    session_runner = SessionRunner(
        cfg, cli_runner=cli_runner, hint_engine=hint_engine
    )
    failure_streaks = FailureStreaks()

    @asynccontextmanager
    async def server_lifespan(_server: FastMCP) -> AsyncIterator[dict[str, Any]]:
        try:
            yield {"config": cfg}
        finally:
            session_runner.stop_project_sessions()

    mcp = FastMCP(name="rf-mcp", lifespan=server_lifespan)
    mcp.add_middleware(ArgumentAliases())
    client_roots = ClientRoots(cfg)
    mcp.add_middleware(client_roots)
    mcp.add_middleware(SkillPointers(cfg, cli_runner.environments))  # after ClientRoots: needs the call's root
    mcp.add_transform(PlainOptionalSchemas())
    skill_summaries = mount_skills_provider(mcp, cfg, client_roots=client_roots)
    mcp.instructions = build_server_instructions(
        cfg, skill_summaries, hint_load_errors=hint_engine.load_errors
    )

    def _effective_hint_engine(root: str | None) -> HintEngine:
        custom_hints = cfg.resolve_hints_file(root)
        if custom_hints is not None:
            return HintEngine(hints_file=custom_hints)
        return hint_engine

    def _sync(files: dict[str, str] | None, mode: str, root: str | None) -> list[str]:
        if files and mode == "server":
            return sync_inline_files(cfg.resolve_root(root), files)
        return []

    @mcp.tool(annotations=READ_ONLY)
    def rf_discover(
        action: Annotated[
            tolerant_literal(DiscoverToolAction),
            Field(description="What to list: tests, tasks, suites, tags (tag -> tests), files, metadata, info, all, config, profiles, profile_show; read = the text of several project files (paths/globs) in one call."),
        ] = "tests",
        search: Annotated[str | None, Field(description="Only items whose name, doc or tags contain this text.")] = None,
        search_regex: Annotated[str | None, Field(description="Only items matching this regular expression.")] = None,
        include_tags: IncludeTags = None,
        exclude_tags: ExcludeTags = None,
        by_longname: ByLongname = None,
        by_test_metadata: ByMetadata = None,
        exclude_by_test_metadata: ExcludeByMetadata = None,
        include_values: Annotated[bool, Field(description="action='metadata': include the values.")] = True,
        verbose: Annotated[bool, Field(description="Return robotcode's full items instead of longname/source/tags.")] = False,
        paths: Paths = None,
        root: Root = None,
        profile: Profile = None,
        target: TargetArg = None,
        shell: ShellArg = None,
    ) -> dict[str, Any]:
        """Use when you need the tests, tasks, suites, tags, files, metadata, profiles or config of a project - instead of grepping or reading .robot files. Returns each test's longname (use it for by_longname in rf_run/rf_debug), source:line and tags; `search` filters by text, action='tags' maps tags to tests. action='read' with paths=['robot.toml','resources/*.resource','docs/*.md'] returns several files in one call. Mistake to avoid: selecting tests by file path, or one Read per file."""
        mode = cfg.resolved_mode(target)
        if action == "read":
            if mode != "server":
                raise ToolError("action='read' reads files on the rf-mcp host; in client mode read the files with your own tools.")
            if not paths:
                raise ToolError("action='read' needs paths: files or globs relative to the root, e.g. ['robot.toml', 'resources/*.resource'].")
            with record_operation("rf_mcp.discover", operation="discover.read", execution_mode=mode, profile=profile or cfg.profile):
                try:
                    return read_project_files(cfg.resolve_root(root), list(paths), cfg.max_output_chars)
                except ValueError as exc:
                    raise ToolError(str(exc)) from exc
        args = cli_runner.build_discover_args(
            action,
            root=root,
            profile=profile,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            by_longname=by_longname,
            by_test_metadata=by_test_metadata,
            exclude_by_test_metadata=exclude_by_test_metadata,
            include_values=include_values,
            search=search,
            search_regex=search_regex,
            paths=paths,
            mode=mode,
        )
        with record_operation(
            "rf_mcp.discover",
            operation=f"discover.{action}",
            execution_mode=mode,
            profile=profile or cfg.profile,
        ) as tracker:
            res = cli_runner.execute(
                args,
                root=root,
                mode_override=mode,
                shell=shell,
                hint_keyword=f"discover {action}",
            )
            if not verbose:
                res.data = project_discover_data(action, res.data)
            tracker.set_result(exit_code=res.exit_code or 0, hints_count=len(res.hints))
            return res.to_response()

    @mcp.tool(annotations=WRITES_FILES)
    def rf_analyze(
        paths: Annotated[StrList | None, Field(description="Files or folders to analyze; default: the whole project.")] = None,
        severity: Annotated[str | None, Field(description="Only report this severity or worse: error, warning, info, hint.")] = None,
        code: Annotated[str | None, Field(description="Only report this diagnostic code, e.g. KeywordNotFound.")] = None,
        filter_glob: Annotated[str | None, Field(description="Glob of files to analyze, e.g. 'tests/**/*.robot'.")] = None,
        collect_unused: Annotated[bool, Field(description="Also report unused keywords and variables.")] = False,
        files: Files = None,
        root: Root = None,
        profile: Profile = None,
        target: TargetArg = None,
        shell: ShellArg = None,
    ) -> dict[str, Any]:
        """Use when you wrote or edited .robot/.resource files, before any rf_run, and to check a project without running it. Runs robotcode static analysis and returns diagnostics (file, 0-based line/character, severity, code, message) plus has_errors/has_warnings; collect_unused=true finds unused keywords and variables. Mistake to avoid: running a new suite before analysis is clean - each KeywordNotFound then costs a full run."""
        mode = cfg.resolved_mode(target)
        synced = _sync(files, mode, root)
        args = cli_runner.build_analyze_args(
            root=root,
            profile=profile,
            severity=severity,
            code=code,
            filter_glob=filter_glob,
            collect_unused=collect_unused,
            paths=paths,
            mode=mode,
        )
        with record_operation(
            "rf_mcp.analyze",
            operation="analyze",
            execution_mode=mode,
            profile=profile or cfg.profile,
        ) as tracker:
            res = cli_runner.execute(
                args,
                root=root,
                mode_override=mode,
                shell=shell,
                hint_keyword="analyze code",
                hint_kind="static",
            )
            tracker.set_result(exit_code=res.exit_code or 0, hints_count=len(res.hints))
            out = res.to_response()
            if mode == "server" and res.status is None:
                out.update(decode_analyze_exit_code(res.exit_code))
                if not out["has_errors"]:
                    out["note"] = ANALYZE_CLEAN_NOTE
            if synced:
                out["synced_files"] = synced
            return out

    @mcp.tool(annotations=READ_ONLY)
    def rf_doc(
        action: Annotated[
            tolerant_literal(DocAction),
            Field(description="keywords: list matching keywords; keyword: one keyword's signature and doc; lib: library overview."),
        ] = "keywords",
        library: Annotated[str | None, Field(description="Library name (Browser, Collections) or a .resource/.py path.")] = None,
        pattern: Annotated[
            StrList | None,
            Field(description="Keyword name or name fragments; several are matched with OR, e.g. ['upload', 'file', 'attach']."),
        ] = None,
        root: Root = None,
        profile: Profile = None,
        target: TargetArg = None,
        shell: ShellArg = None,
    ) -> dict[str, Any]:
        """Use when you need a keyword's exact name, arguments or documentation - never guess them. action='keywords' lists keywords of `library` whose name matches any `pattern`; matching is by name, not meaning, so pass synonyms (['upload', 'file', 'attach']). action='keyword' shows one keyword's signature, action='lib' a library overview. Project libraries and .resource files resolve via robot.toml. Mistake to avoid: inventing keyword names from memory and finding out at run time."""
        mode = cfg.resolved_mode(target)
        args = cli_runner.build_doc_args(
            action,
            library=library,
            pattern=pattern,
            root=root,
            profile=profile,
            mode=mode,
        )
        with record_operation(
            "rf_mcp.doc",
            operation=f"doc.{action}",
            execution_mode=mode,
            profile=profile or cfg.profile,
        ) as tracker:
            res = cli_runner.execute(
                args,
                root=root,
                mode_override=mode,
                shell=shell,
                hint_keyword=f"doc {action} {library or ''} {' '.join(pattern or [])}".strip(),
            )
            tracker.set_result(exit_code=res.exit_code or 0, hints_count=len(res.hints))
            return res.to_response()

    @mcp.tool(annotations=RUNS_TESTS)
    def rf_run(
        by_longname: ByLongname = None,
        include_tags: IncludeTags = None,
        exclude_tags: ExcludeTags = None,
        by_test_metadata: ByMetadata = None,
        exclude_by_test_metadata: ExcludeByMetadata = None,
        variables: Variables = None,
        rpa: Annotated[bool, Field(description="Run tasks (--rpa).")] = False,
        wrapper: Annotated[str | None, Field(description="Wrapper command (--wrapper), e.g. for a display server.")] = None,
        outputdir: Annotated[str | None, Field(description="Output directory (--outputdir).")] = None,
        output: Annotated[str | None, Field(description="Name of the result file (--output), e.g. 'full.xml' to pin a run.")] = None,
        rerunfailed: Annotated[str | None, Field(description="Rerun only the failed tests of this result file (--rerunfailed).")] = None,
        dryrun: Annotated[bool, Field(description="Check keywords and arguments without executing (--dryrun).")] = False,
        paths: Paths = None,
        files: Files = None,
        root: Root = None,
        profile: Profile = None,
        target: TargetArg = None,
        shell: ShellArg = None,
        timeout_seconds: Annotated[int, Field(description="Stop the run, with all its processes, after this many seconds.")] = 180,
    ) -> dict[str, Any]:
        """Use when you need to execute tests or tasks. Select with by_longname (from rf_discover) or tags, not with a single file in `paths`: the file becomes the top suite and parent __init__.robot setups are skipped. Returns `outcome` (PASS or FAIL with counts), exit_code and `failures` ('test: message'); ask rf_results for details. Pin a full run with output='full.xml', then rerun only failures with rerunfailed. Mistake to avoid: rerunning the whole suite after fixing one test - rerun its longname."""
        mode = cfg.resolved_mode(target)
        synced = _sync(files, mode, root)
        args = cli_runner.build_run_args(
            root=root,
            profile=profile,
            by_longname=by_longname,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            by_test_metadata=by_test_metadata,
            exclude_by_test_metadata=exclude_by_test_metadata,
            variables=variables,
            rpa=rpa,
            wrapper=wrapper,
            outputdir=outputdir,
            output=output,
            rerunfailed=rerunfailed,
            dryrun=dryrun,
            paths=paths,
            mode=mode,
        )
        warning = cli_runner.selection_warning(
            root=root, paths=paths, by_longname=by_longname, mode=mode
        )
        with record_operation(
            "rf_mcp.run",
            operation="run",
            execution_mode=mode,
            profile=profile or cfg.profile,
        ) as tracker:
            res = cli_runner.execute(
                args,
                root=root,
                mode_override=mode,
                shell=shell,
                timeout_seconds=timeout_seconds,
                hint_keyword="robot",
            )
            res.warning = " ".join(w for w in (warning, res.warning) if w) or None
            if res.mode == "server":
                res.instruction = failure_streaks.update(res.failures or [], res.exit_code) or res.instruction
            tracker.set_result(exit_code=res.exit_code or 0, hints_count=len(res.hints))
            out = res.to_response()
            verdict = run_outcome(res, dryrun=dryrun)
            if verdict:
                out = {"outcome": verdict, **out}
            if synced:
                out["synced_files"] = synced
            return out

    @mcp.tool(annotations=READ_ONLY)
    def rf_results(
        action: Annotated[
            tolerant_literal(ResultsAction),
            Field(description="summary, show (test list), log (keyword log), stats, diff (against baseline), merge (rebot --merge)."),
        ] = "summary",
        failed_only: Annotated[bool, Field(description="Only failed tests.")] = False,
        search: Annotated[str | None, Field(description="Only tests whose log contains this text (case-insensitive).")] = None,
        search_regex: Annotated[str | None, Field(description="Only tests whose log matches this regular expression.")] = None,
        include_tags: IncludeTags = None,
        exclude_tags: ExcludeTags = None,
        by_longname: ByLongname = None,
        by_test_metadata: ByMetadata = None,
        exclude_by_test_metadata: ExcludeByMetadata = None,
        level: Annotated[str | None, Field(description="action='log': lowest message level shown (trace, debug, info, warn, error, fail).")] = None,
        max_depth: Annotated[int | None, Field(description="action='log': nesting depth of keywords shown (0 = unlimited).")] = None,
        sort: Annotated[str | None, Field(description="action='show'|'stats': sort field, e.g. 'elapsed', 'name', 'failed'.")] = None,
        by: Annotated[StrList | None, Field(description="action='stats': group by 'tag', 'suite' and/or 'status'.")] = None,
        top: Annotated[int | None, Field(description="action='show'|'stats': at most N entries.")] = None,
        message_chars: Annotated[int | None, Field(description="action='show'|'diff': cut messages to N characters.")] = None,
        extract_dir: Annotated[str | None, Field(description="action='log': copy screenshots and other artefacts into this folder.")] = None,
        output_path: Annotated[str | None, Field(description="Result file relative to root (default: the last run's); for 'diff' the newer file; for 'merge' the file to write.")] = None,
        baseline: Annotated[str | None, Field(description="action='diff': the older result file to compare against.")] = None,
        only: Annotated[StrList | None, Field(description="action='diff': new-failures, new-passes, status-changes, added and/or removed.")] = None,
        merge_inputs: Annotated[StrList | None, Field(description="action='merge': result files, the full run first, then the reruns.")] = None,
        root: Root = None,
        profile: Profile = None,
        target: TargetArg = None,
        shell: ShellArg = None,
    ) -> dict[str, Any]:
        """Use when asked 'why did X fail?' or anything about the last run - call it first; never read or grep output.xml. 'summary'/'show' list results, 'log' shows a test's keyword log (failed_only, search, level, max_depth; extract_dir saves screenshots), 'stats' groups by tag or suite, 'diff' compares with `baseline`, 'merge' combines a pinned full run with reruns. Mistake to avoid: passing a result file anywhere but output_path."""
        _reject_result_files(
            search=search,
            search_regex=search_regex,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            by_longname=by_longname,
            extract_dir=extract_dir,
            sort=sort,
            level=level,
        )
        mode = cfg.resolved_mode(target)
        try:
            args = cli_runner.build_results_args(
                action,
                root=root,
                profile=profile,
                failed_only=failed_only,
                include_tags=include_tags,
                exclude_tags=exclude_tags,
                by_longname=by_longname,
                by_test_metadata=by_test_metadata,
                exclude_by_test_metadata=exclude_by_test_metadata,
                search=search,
                search_regex=search_regex,
                sort=sort,
                level=level,
                max_depth=max_depth,
                message_chars=message_chars,
                by=by,
                top=top,
                extract_dir=extract_dir,
                output_path=output_path,
                baseline=baseline,
                only=only,
                merge_inputs=merge_inputs,
                mode=mode,
            )
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        with record_operation(
            "rf_mcp.results",
            operation=f"results.{action}",
            execution_mode=mode,
            profile=profile or cfg.profile,
        ) as tracker:
            res = cli_runner.execute(
                args,
                root=root,
                mode_override=mode,
                shell=shell,
                hint_keyword=f"results {action}",
                hint_kind="cli",
            )
            if action == "merge" and res.exit_code == 0:
                merged = args[args.index("--output") + 1]
                res.instruction = f"Merged into '{merged}'. Inspect it with rf_results(output_path='{merged}')."
            tracker.set_result(exit_code=res.exit_code or 0, hints_count=len(res.hints))
            return res.to_response()

    @mcp.tool(annotations=MAY_OVERWRITE)
    def rf_repl(
        action: Annotated[
            tolerant_literal(ReplAction),
            Field(description="start, send, wait, status, result, interrupt, restart (reload edited libraries), list, save, stop."),
        ] = "start",
        session_id: SessionId = None,
        statement: Annotated[
            str | None,
            Field(description="action='send': one or more lines of Robot Framework steps, a FOR/IF/TRY block, or a dot command (.kw, .doc, .vars --user, .imports)."),
        ] = None,
        save_path: Annotated[str | None, Field(description="action='save': .robot file to write.")] = None,
        test_name: Annotated[str | None, Field(description="action='save': name of the saved test.")] = None,
        append: Annotated[bool, Field(description="action='save': add the test to an existing file.")] = False,
        keep_failed: Annotated[bool, Field(description="action='save': keep failed steps (commented) for review.")] = False,
        variables: Variables = None,
        variable_files: Annotated[StrList | None, Field(description="action='start': variable files (-V).")] = None,
        source: Annotated[str | None, Field(description="action='start': .robot/.resource file whose imports and keywords the session loads.")] = None,
        timeout_seconds: TimeoutSeconds = None,
        idle_timeout_minutes: IdleTimeout = None,
        stop_scope: StopScope = "id",
        files: Files = None,
        root: Root = None,
        profile: Profile = None,
        target: TargetArg = None,
        shell: ShellArg = None,
    ) -> dict[str, Any]:
        """Use when exploring a library, page or API, or authoring a new test, with no existing test in play. One session keeps imports and variables. A send may hold several lines: they run in order and stop at the first failure; a FOR/IF/TRY block goes in one send. Inspect with 'Get Aria Snapshot', '.kw <text>', '.doc <lib>', '.vars --user', '.imports'. 'save' writes the passing steps as a test. Mistake to avoid: Run Keywords cannot assign variables - send several lines instead."""
        mode = cfg.resolved_mode(target)
        synced = _sync(files, mode, root)
        with record_operation(
            "rf_mcp.repl",
            operation=f"repl.{action}",
            execution_mode=mode,
            profile=profile or cfg.profile,
        ) as tracker:
            try:
                res = session_runner.run_repl(
                    action,
                    session_id=session_id,
                    statement=statement,
                    save_path=save_path,
                    test_name=test_name,
                    append=append,
                    keep_failed=keep_failed,
                    variables=variables,
                    variable_files=variable_files,
                    source=source,
                    timeout_seconds=timeout_seconds,
                    idle_timeout_minutes=idle_timeout_minutes,
                    stop_scope=stop_scope,
                    root=root,
                    profile=profile,
                    mode_override=mode,
                    shell=shell,
                )
            except ValueError as exc:
                raise ToolError(str(exc)) from exc
            tracker.set_result(
                exit_code=res.exit_code or 0,
                hints_count=len(res.hints),
                status=res.status,
            )
            out = res.to_response()
            if synced:
                out["synced_files"] = synced
            return out

    @mcp.tool(annotations=RUNS_TESTS)
    def rf_debug(
        action: Annotated[
            tolerant_literal(DebugAction),
            Field(description="start, send, wait, status, result, interrupt, restart, list, stop."),
        ] = "start",
        session_id: SessionId = None,
        command: Annotated[
            str | None,
            Field(description="action='send': keywords to run in the paused frame and/or debugger commands (.where, .list, .vars --user, .c, .n, .s, .abort), one per line; several lines per send run in order."),
        ] = None,
        by_longname: ByLongname = None,
        include_tags: IncludeTags = None,
        exclude_tags: ExcludeTags = None,
        by_test_metadata: ByMetadata = None,
        breakpoints: Annotated[StrList | None, Field(description="Break locations: 'file.robot:line' or a keyword name.")] = None,
        stop_on_entry: Annotated[bool, Field(description="Pause at the very first step.")] = False,
        no_break_on_exception: Annotated[bool, Field(description="Do not pause at an uncaught failing keyword (the default pause).")] = False,
        break_on_all_exceptions: Annotated[bool, Field(description="Pause at every failing keyword, even handled ones.")] = False,
        break_on_failed_test: Annotated[bool, Field(description="Pause at the end of a failing test.")] = False,
        break_on_failed_suite: Annotated[bool, Field(description="Pause at the end of a failing suite.")] = False,
        keep_outputs: Annotated[bool, Field(description="Write output.xml/log/report (default: none, so the last run's results stay).")] = False,
        variables: Variables = None,
        paths: Paths = None,
        timeout_seconds: TimeoutSeconds = None,
        idle_timeout_minutes: IdleTimeout = None,
        stop_scope: StopScope = "id",
        files: Files = None,
        root: Root = None,
        profile: Profile = None,
        target: TargetArg = None,
        shell: ShellArg = None,
    ) -> dict[str, Any]:
        """Use when an existing test fails or behaves unexpectedly - debug the real test instead of rebuilding it in the REPL. start (with by_longname) pauses at the failure or a breakpoint with the real suite setup and variables, without overwriting output.xml. send runs keywords in the paused frame and debugger commands (.where, .list, .vars --user, .c). Put several lines in one send: they run in order and stop at the first failure. Mistake to avoid: one probe per send, or bare 'where'/'c' - debugger commands start with a dot."""
        mode = cfg.resolved_mode(target)
        synced = _sync(files, mode, root)
        with record_operation(
            "rf_mcp.debug",
            operation=f"debug.{action}",
            execution_mode=mode,
            profile=profile or cfg.profile,
        ) as tracker:
            try:
                res = session_runner.run_debug(
                    action,
                    session_id=session_id,
                    command=command,
                    by_longname=by_longname,
                    include_tags=include_tags,
                    exclude_tags=exclude_tags,
                    by_test_metadata=by_test_metadata,
                    breakpoints=breakpoints,
                    stop_on_entry=stop_on_entry,
                    no_break_on_exception=no_break_on_exception,
                    break_on_all_exceptions=break_on_all_exceptions,
                    break_on_failed_test=break_on_failed_test,
                    break_on_failed_suite=break_on_failed_suite,
                    keep_outputs=keep_outputs,
                    variables=variables,
                    paths=paths,
                    timeout_seconds=timeout_seconds,
                    idle_timeout_minutes=idle_timeout_minutes,
                    stop_scope=stop_scope,
                    root=root,
                    profile=profile,
                    mode_override=mode,
                    shell=shell,
                )
            except ValueError as exc:
                raise ToolError(str(exc)) from exc
            tracker.set_result(
                exit_code=res.exit_code or 0,
                hints_count=len(res.hints),
                status=res.status,
            )
            out = res.to_response()
            if action == "start":
                warning = cli_runner.selection_warning(
                    root=root, paths=paths, by_longname=by_longname, mode=mode
                )
                if warning:
                    out["warning"] = " ".join(w for w in (warning, out.get("warning")) if w)
            if synced:
                out["synced_files"] = synced
            return out

    @mcp.tool(annotations=READ_ONLY)
    def rf_diagnose(
        error_text: Annotated[str, Field(description="The error message or failing output.")],
        keyword: Annotated[str, Field(description="The failing keyword or whole statement, e.g. 'Click    css=.btn'.")] = "",
        arguments: Annotated[StrList | None, Field(description="The keyword's arguments, if not part of `keyword`.")] = None,
        libraries: Annotated[StrList | None, Field(description="Imported libraries, e.g. ['Browser'].")] = None,
        max_hints: Annotated[int, Field(description="Maximum number of hints.")] = 3,
        root: Root = None,
    ) -> dict[str, Any]:
        """Use when a keyword, REPL statement or run failed and the attached hints did not help, or in client mode with the output of a local robotcode run. Returns ranked recovery hints with fix examples for Browser, SeleniumLibrary, RequestsLibrary, AppiumLibrary, BuiltIn and robotcode sessions, plus project rules from .robotframework/hints.toml. Mistake to avoid: retrying the same failing call unchanged - apply the hint and change one thing."""
        engine = _effective_hint_engine(root)
        with record_operation(
            "rf_mcp.diagnose",
            operation="diagnose",
            execution_mode=cfg.resolved_mode(),
            profile=cfg.profile,
        ) as tracker:
            hints = engine.diagnose(
                HintContext(
                    keyword=keyword,
                    arguments=arguments or [],
                    error_text=error_text,
                    libraries=libraries or [],
                ),
                max_hints=max_hints,
            )
            tracker.set_result(exit_code=0, hints_count=len(hints))
            return {
                "keyword": keyword,
                "hints_count": len(hints),
                "hints": [h.model_dump() for h in hints],
            }

    return mcp
