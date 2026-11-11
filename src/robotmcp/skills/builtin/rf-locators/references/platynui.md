# PlatynUI.BareMetal (desktop) locators

Ported from rf-mcp 0.36.0. Legacy source: main:src/robotmcp/utils/rf_native_type_converter.py (`get_platynui_locator_guidance`); components/execution/desktop_guidance.py (`_LOCATOR_CRIB`); plugins/builtin/platynui_plugin.py (`PLATYNUI_KEYWORDS`, `_ACTIONABILITY_HINT`, `_LINUX_FRAME_HINT`, `_MATCHED_SET_HINT`, usage_examples, PromptBundle); domains/instruction/value_objects.py (`InstructionTemplate.desktop_focused`); domains/intent/aggregates.py (`_builtin_platynui_mappings`).

## Model
The only locator strategy is XPath over the desktop accessibility tree: Windows UIA or Linux AT-SPI2. There are no `css=`, `id=` or `text=` prefixes.

| Namespace | Meaning | Example |
|---|---|---|
| `app:` | application | `/app:*[@Name='gnome-calculator']` |
| `control:` | UI control | `control:Button`, `control:Frame`, `control:Text`, `control:Label` |
| `item:` | list or tree item | `item:ListItem[@Name='File.txt']` |
| `native:` | technology-specific property (advanced) | `native:Text.CharacterCount` |

Attributes are PascalCase: `@Name`, `@AutomationId`, `@ClassName`, `@Bounds`, `@IsVisible`, `@IsEnabled`.

## Rules
| Rule | Why |
|---|---|
| Scope every locator to an app: `/app:*[@Name='myapp']//control:Button[@Name='OK']` | An unscoped `//…` walks the whole desktop. AT-SPI waits about 1s per unresponsive node, which adds up to minutes, and legacy refused such queries |
| `Set Root    /app:*[@Name='myapp']` once, then use short relative queries | faster and cleaner |
| List apps first with `Query    /app:*` (about 1s) | app names must match exactly and are case-sensitive |
| On Linux, top-level windows are `control:Frame`, not `control:Window` | `control:Window` matches only compositor/shell elements and hangs for 30s. On Windows (UIA) they are `control:Window` |
| Prefer `@AutomationId` when it exists: `//control:*[@AutomationId='submitBtn']` | |
| Descriptors resolve lazily with retry (30s timeout, 0.1s interval by default) | no explicit waits needed |

## Launch and focus order
1. Launch the app visibly first: `Start Process    gnome-calculator` (Process library).
2. Wait for its window: `Query    /app:*[@Name='gnome-calculator']//control:Frame    only_first=True`.
3. Do not run any PlatynUI keyword before the app is launched. The desktop tree is snapshotted on first use, so apps started later may be invisible. Legacy refreshed the cache after Start Process. If a name lookup fails right after launch, wait briefly and query again rather than switching to coordinates or OCR.
4. Run `Activate Window    <frame>` before pointer or keyboard input after switching apps. Legacy did this automatically; it can be disabled with `ROBOTMCP_PLATYNUI_NO_FOCUS=1`.

## Keywords (24, new core)
- **Query and context:** Set Root, Query, Get Attribute
- **Pointer:** Pointer Click, Pointer Multi Click, Pointer Press, Pointer Release, Pointer Move To, Get Pointer Position
- **Keyboard:** Keyboard Type, Keyboard Press, Keyboard Release
- **Window:** Focus, Activate Window, Maximize Window, Minimize Window, Restore Window, Close Window, Move Window, Resize Window, Move And Resize Window, Bring To Front
- **Diagnostics:** Take Screenshot, Highlight

Window keywords accept Frame, Window and Dialog nodes alike.

```robotframework
Pointer Click    /app:*[@Name='myapp']//control:Button[@Name='OK']
Keyboard Type    /app:*[@Name='myapp']//control:Text[@Name='Input']    Hello <Ctrl+A>
${b}=    Get Attribute    /app:*[@Name='myapp']//control:Frame    Bounds
Take Screenshot    filename=EMBED
```

- **Keyboard Type** accepts chords: `<Ctrl+A>`, `<Ctrl+C>then more text`, `<Enter>`. Prefer it for shortcuts. A bare `Keyboard Press` only sends key-down and must be paired with `Keyboard Release`, otherwise the key stays held at OS level.
- **Take Screenshot** takes `(descriptor, filename, rect)`. The first positional argument is a node, not a path, so pass `filename=`.
- **Get Attribute and the text mode:** reading "text" means reading the `Name` attribute.

## Duplicates and naming
- On Wayland/AT-SPI one app can expose several roots, and one query can return duplicate controls with different bounds. Do not blindly pick an index. Use `Set Root` on one root, then check `IsVisible`, `IsInView` and `Bounds` via `Get Attribute`.
- Names vary by role. Operator buttons use the symbol (`//control:Button[@Name='+']`), while the same operator on a Label may be a word (`//control:Label[@Name='plus']`). Read the exact `@Name` from the tree instead of guessing.

## Reading results (GTK)
GTK4 does not reliably expose live entry text. Try these in order:
1. **Result or history Labels:** `Get Attribute    <app>//control:Label[@Name='56']    Name`.
2. **Character count as a length proxy:** `Get Attribute    <app>//control:Text    native:Text.CharacterCount`. It can wrongly report 0.
3. **Last resort:** `Take Screenshot` plus OCR via `Run Process`.

## Inspection and troubleshooting
- **Element not found:** run `Query    /app:*`, then descend one level at a time.
  - CLI inspector: `platynui-cli snapshot "/app:*[@Name='myapp']" --max-depth 3`.
- **Empty tree right after launch:** the app may expose no AT-SPI tree. Set `GTK_A11Y=atspi` (the backend name, not `1`) or enable toolkit accessibility, then relaunch.
- **Timeouts:** scope to the app and query intermediate containers first.
- **Evaluate:** `Evaluate` accepts expressions only. Use `Run Process` for multi-statement recovery. A UiNode has no `get_attribute()`; use the `Get Attribute` keyword.
- **Wayland:** legacy forced `XDG_SESSION_TYPE=x11` to avoid a portal consent dialog (`ROBOTMCP_PLATYNUI_KEEP_WAYLAND=1` opts out), and pinned `GDK_BACKEND=x11` / `QT_QPA_PLATFORM=xcb`.
- **ImportError on native symbols:** PlatynUI.BareMetal and platynui-native come from different commits. Rebuild them as a matched set.
