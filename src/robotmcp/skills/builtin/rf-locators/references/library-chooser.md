# Choosing a library

Ported from rf-mcp 0.36.0. Legacy source: main:src/robotmcp/components/library_recommender.py (`LibraryRecommender`: `EXCLUSION_GROUPS`, `LIBRARY_ALIASES`, `_extract_keywords`, `_match_by_context`, `_filter_by_context`, `_apply_preferences`, `_apply_strict_exclusions`); plugins/builtin/definitions.py (`BUILTIN_LIBRARY_DEFINITIONS`); plugins/builtin/{browser,selenium,requests,platynui}_plugin.py (metadata, `get_incompatible_libraries`); utils/library_detection.py (`LIBRARY_RULES_DEFAULT`).

## By context
The first library listed is the primary choice.

| Context | Libraries in legacy order | Notes |
|---|---|---|
| web | **Browser**, SeleniumLibrary, RequestsLibrary | Browser is recommended (Playwright). SeleniumLibrary is the "traditional" option and is dropped when Browser is chosen |
| mobile | **AppiumLibrary** | Browser and SeleniumLibrary are always excluded |
| api | **RequestsLibrary**, REST | UI and mobile libraries only when the scenario mentions them |
| database | **DatabaseLibrary** | |
| desktop | **PlatynUI.BareMetal**, FlaUILibrary, ImageHorizonLibrary, SikuliLibrary | PlatynUI comes first and AppiumLibrary is dropped, so a Linux desktop scenario is not routed to mobile |
| system | SSHLibrary, Process, OperatingSystem | Telnet and FTPLibrary are also registered |
| visual | DocTest.VisualTest, DocTest.PdfTest, Screenshot | |
| enterprise | RoboSAPiens | |
| data | DataDriver, FakerLibrary, XML | |

REST, FlaUILibrary, ImageHorizonLibrary, SikuliLibrary, DocTest.*, FakerLibrary and RoboSAPiens appear in this list but had no built-in registry entry. They were only recommended when a plugin registered them.

## Conflict rules
- **Browser vs SeleniumLibrary: never both.** They form an exclusion group (`web_automation`), and each plugin declares the other incompatible. Import only one per suite.
- Without a stated preference, Browser wins in the web context.
- An explicit preference wins. If the user names a library, honour it and drop the other web library. The alias `PlatynUI` resolves to `PlatynUI.BareMetal`.

## Install notes (from plugin metadata)
| Library | Install |
|---|---|
| Browser | `pip install robotframework-browser`, then `rfbrowser init` (needs Node.js) |
| SeleniumLibrary | `pip install robotframework-seleniumlibrary` |
| RequestsLibrary | `pip install robotframework-requests` |
| AppiumLibrary | `pip install robotframework-appiumlibrary` (needs an Appium server, default `http://127.0.0.1:4723`) |
| DatabaseLibrary, SSHLibrary, FTPLibrary, DataDriver | `pip install robotframework-<name>` (lower-case package names) |
| PlatynUI.BareMetal | Python 3.12 or newer. `pip install --pre platynui-native platynui-cli`, then install the RF library from git branch `new_core`. The library and platynui-native must come from the same commit |
| BuiltIn, Collections, DateTime, Dialogs, OperatingSystem, Process, Screenshot, String, Telnet, XML | built in |
