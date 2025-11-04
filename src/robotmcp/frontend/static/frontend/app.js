(() => {
  const basePath = normalizeBasePath(
    (window.ROBOTMCP_FRONTEND && window.ROBOTMCP_FRONTEND.basePath) || "/"
  );

  const state = {
    sessions: [],
    selectedSessionId: null,
    sessionDetails: null,
    sessionVariables: {},
    derivedVariables: {},
    sessionSteps: [],
    sessionStepMap: new Map(),
    stepOrder: [],
    disabledSteps: new Set(),
    editedSteps: new Map(),
    eventLog: [],
    lastEventTimestamp: null,
    refreshTimer: null,
    suiteText: "",
    suiteNotice: null,
    suiteNoticeTimer: null,
    compactMode: false,
  };

  const elements = {
    sessionsList: document.getElementById("sessions-list"),
    refreshSessions: document.getElementById("refresh-sessions"),
    headerRefresh: document.getElementById("header-refresh"),
    headerBuildSuite: document.getElementById("header-build-suite"),
    sessionPanel: document.getElementById("session-detail"),
    sessionTitle: document.getElementById("session-title"),
    sessionSubtitle: document.getElementById("session-subtitle"),
    sessionActions: document.getElementById("session-actions"),
    reloadSession: document.getElementById("reload-session"),
    toggleCompact: document.getElementById("toggle-compact"),
    previewSuiteBtn: document.getElementById("preview-suite"),
    suiteStatus: document.getElementById("suite-status"),
    sessionMeta: document.getElementById("session-meta"),
    sessionVariables: document.getElementById("session-variables"),
    sessionSteps: document.getElementById("session-steps"),
    suitePreview: document.getElementById("suite-preview"),
    suiteCode: document.getElementById("suite-code"),
    copySuite: document.getElementById("copy-suite"),
    eventsLog: document.getElementById("events-log"),
  };

  const BUILTIN_VARIABLE_KEYS = new Set([
    "",
    "-1",
    "/",
    "0",
    "1",
    ":",
    "\\",
    "\\n",
    "\\r",
    "\\t",
    "CURDIR",
    "EXECDIR",
    "OUTPUTDIR",
    "OUTPUT",
    "LOGFILE",
    "TEMPDIR",
    "SPACE",
    "NONE",
    "NULL",
    "TRUE",
    "FALSE",
    "EMPTY",
  ]);

  const PLATFORM_ICON = {
    web: "cpu",
    mobile: "smartphone",
    desktop: "monitor",
    api: "cloud",
  };

  const STATUS_ICON = {
    pass: { icon: "check", stack: "success" },
    fail: { icon: "x-circle", stack: "fail" },
    running: { icon: "clock", stack: "warning" },
  };

  let stepDragContext = null;

  function normalizeBasePath(path) {
    if (!path) {
      return "/";
    }
    let normalized = path.trim();
    if (!normalized.startsWith("/")) {
      normalized = `/${normalized}`;
    }
    if (!normalized.endsWith("/")) {
      normalized = `${normalized}/`;
    }
    return normalized;
  }

  function buildUrl(path) {
    return `${basePath}${path.replace(/^\//, "")}`;
  }

  function getCsrfToken() {
    const name = "csrftoken";
    const cookies = document.cookie ? document.cookie.split(";") : [];
    for (const cookie of cookies) {
      const trimmed = cookie.trim();
      if (trimmed.startsWith(`${name}=`)) {
        return decodeURIComponent(trimmed.substring(name.length + 1));
      }
    }
    return null;
  }

  async function fetchJSON(path, options) {
    const url = path.startsWith("http") ? path : buildUrl(path);
    const headers = {
      "Content-Type": "application/json",
      ...(options && options.headers ? options.headers : {}),
    };

    const init = {
      credentials: "same-origin",
      ...options,
      headers,
    };

    const method = init.method ? init.method.toUpperCase() : "GET";
    if (method !== "GET") {
      const token = getCsrfToken();
      if (token) {
        init.headers["X-CSRFToken"] = token;
      }
    }

    const response = await fetch(url, init);
    if (!response.ok) {
      let message = `Request failed (${response.status})`;
      try {
        const payload = await response.json();
        if (payload && payload.error) {
          message = payload.error;
        }
      } catch (error) {
        /* ignore parse errors */
      }
      throw new Error(message);
    }
    return response.json();
  }

  function filterUserVariables(variables) {
    const filtered = {};
    for (const [key, value] of Object.entries(variables || {})) {
      if (BUILTIN_VARIABLE_KEYS.has(key)) {
        continue;
      }
      if (typeof key === "string") {
        if (key.startsWith("${")) {
          filtered[key] = value;
          continue;
        }
        if (/^[A-Z_]+$/.test(key)) {
          continue;
        }
      }
      filtered[key] = value;
    }
    return filtered;
  }

  function extractAssignedVariableNames(step) {
    if (!step) {
      return [];
    }
    const assigned = step.assigned_variables;
    if (Array.isArray(assigned) && assigned.length) {
      return assigned;
    }
    if (assigned && typeof assigned === "object") {
      return Object.keys(assigned);
    }
    if (step.variables && typeof step.variables === "object") {
      return Object.keys(step.variables);
    }
    return [];
  }

  function getAssignedVariables(step) {
    if (!step || !step.assigned_variables) {
      return [];
    }
    if (Array.isArray(step.assigned_variables)) {
      return step.assigned_variables;
    }
    if (typeof step.assigned_variables === "object") {
      return Object.keys(step.assigned_variables);
    }
    return [];
  }

  function getAssignedVariableValue(step, name) {
    if (!name) {
      return undefined;
    }
    const sources = [step?.variables, state.sessionVariables, state.derivedVariables];
    for (const source of sources) {
      if (source && Object.prototype.hasOwnProperty.call(source, name)) {
        return source[name];
      }
    }
    return undefined;
  }

  function formatValueForDisplay(value) {
    if (value === null || value === undefined) {
      return "None";
    }
    if (typeof value === "object") {
      try {
        return JSON.stringify(value);
      } catch (error) {
        return String(value);
      }
    }
    return String(value);
  }

  function getDisplayStep(stepId) {
    const original = state.sessionStepMap.get(stepId);
    if (!original) {
      return null;
    }
    const edits = state.editedSteps.get(stepId);
    if (!edits) {
      return original;
    }
    return {
      ...original,
      ...edits,
      arguments: edits.arguments !== undefined ? edits.arguments : original.arguments,
      assigned_variables:
        edits.assigned_variables !== undefined
          ? edits.assigned_variables
          : original.assigned_variables,
    };
  }

  function refreshIcons() {
    if (window.feather && typeof window.feather.replace === "function") {
      window.feather.replace({ width: 16, height: 16 });
    }
  }

  function humanizePlatform(platform) {
    if (!platform) {
      return "Unknown";
    }
    return platform.toString().replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  }

  function renderSessions() {
    const container = elements.sessionsList;
    container.innerHTML = "";
    if (!state.sessions.length) {
      const empty = document.createElement("div");
      empty.className = "muted";
      empty.textContent = "No active sessions yet.";
      container.appendChild(empty);
      refreshIcons();
      return;
    }

    const fragment = document.createDocumentFragment();
    const sorted = [...state.sessions].sort((a, b) => (a.last_activity < b.last_activity ? 1 : -1));

    sorted.forEach((session) => {
      const card = document.createElement("article");
      card.className = "session-card";
      if (session.session_id === state.selectedSessionId) {
        card.classList.add("active");
      }
      card.dataset.sessionId = session.session_id;

      const title = document.createElement("h3");
      const icon = PLATFORM_ICON[session.platform_type] || "cpu";
      title.innerHTML = `<i data-feather="${icon}"></i>${session.session_id.slice(0, 8)}`;

      const footer = document.createElement("footer");
      const platformSpan = document.createElement("span");
      platformSpan.innerHTML = `<i data-feather="globe"></i>${humanizePlatform(session.platform_type)}`;

      const stepsSpan = document.createElement("span");
      stepsSpan.innerHTML = `<i data-feather="list"></i>${session.step_count || 0} steps`;

      const timeSpan = document.createElement("span");
      timeSpan.innerHTML = `<i data-feather="clock"></i>${formatRelativeTime(session.last_activity)}`;

      footer.append(platformSpan, stepsSpan, timeSpan);

      card.append(title, footer);
      card.addEventListener("click", () => selectSession(session.session_id));
      fragment.appendChild(card);
    });
    container.appendChild(fragment);
    refreshIcons();
  }

  function formatRelativeTime(isoString) {
    if (!isoString) {
      return "—";
    }
    try {
      const then = new Date(isoString).getTime();
      const now = Date.now();
      const diff = Math.max(0, now - then);
      const minutes = Math.floor(diff / (60 * 1000));
      if (minutes < 1) {
        return "just now";
      }
      if (minutes === 1) {
        return "1 min ago";
      }
      if (minutes < 60) {
        return `${minutes} mins ago`;
      }
      const hours = Math.floor(minutes / 60);
      if (hours === 1) {
        return "1 hr ago";
      }
      if (hours < 24) {
        return `${hours} hrs ago`;
      }
      const days = Math.floor(hours / 24);
      return days === 1 ? "1 day ago" : `${days} days ago`;
    } catch (error) {
      return "—";
    }
  }

  function firstMeaningfulValue(...candidates) {
    for (const candidate of candidates) {
      if (candidate === undefined || candidate === null) {
        continue;
      }
      const value = typeof candidate === "string" ? candidate.trim() : candidate;
      if (typeof value === "string") {
        if (!value) {
          continue;
        }
        const lowered = value.toLowerCase();
        if (lowered === "n/a" || lowered === "none") {
          continue;
        }
        return value;
      }
      return value;
    }
    return undefined;
  }

  function normalizeBrowserName(browser) {
    if (!browser) {
      return "—";
    }
    const normalized = browser.toString().trim();
    if (!normalized || normalized.toLowerCase() === "unknown") {
      return "—";
    }
    return normalized
      .replace(/[_-]+/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase());
  }

  function normalizeUrl(url) {
    if (!url) {
      return "—";
    }
    const trimmed = url.toString().trim();
    if (!trimmed || trimmed === "data:," || trimmed === "null") {
      return "—";
    }
    return trimmed;
  }

  const STEP_LIBRARY_HINTS = {
    Browser: [
      "new browser",
      "new page",
      "go to",
      "go to url",
      "set viewport size",
      "click",
      "fill text",
      "press keys",
      "wait for elements state",
      "take screenshot",
    ],
    SeleniumLibrary: [
      "open browser",
      "click element",
      "input text",
      "press keys",
      "go to",
      "maximize browser window",
      "wait until element is visible",
      "capture page screenshot",
    ],
    RequestsLibrary: [
      "create session",
      "get on session",
      "post on session",
      "put on session",
      "delete on session",
      "get request",
      "post request",
    ],
  };

  function inferLibrariesFromSteps() {
    const libs = new Set();
    (state.sessionSteps || []).forEach((step) => {
      const keyword = (step.keyword || "").toLowerCase();
      Object.entries(STEP_LIBRARY_HINTS).forEach(([lib, hints]) => {
        if (hints.some((hint) => keyword === hint || keyword.includes(hint))) {
          libs.add(lib);
        }
      });
    });
    return libs;
  }

  function inferBrowserFromSteps() {
    for (let i = (state.sessionSteps || []).length - 1; i >= 0; i -= 1) {
      const step = state.sessionSteps[i];
      const keyword = (step.keyword || "").toLowerCase();
      const args = step.arguments || [];
      if (keyword === "open browser" && args.length >= 2) {
        return args[1];
      }
      if (keyword === "new browser" && args.length >= 1) {
        return args[0];
      }
    }
    return undefined;
  }

  function inferUrlFromSteps() {
    for (let i = (state.sessionSteps || []).length - 1; i >= 0; i -= 1) {
      const step = state.sessionSteps[i];
      const args = step.arguments || [];
      for (const arg of args) {
        if (typeof arg === "string" && /^https?:\/\//i.test(arg.trim())) {
          return arg.trim();
        }
      }
    }
    return undefined;
  }

  function renderSessionMeta(detail, sessionState) {
    elements.sessionMeta.innerHTML = "";
    const summary = detail.summary || {};
    const browserState = detail.browser_state || {};
    const domState = sessionState?.dom || {};
    const domBrowserState = domState.browser_state || {};
    const libraryState = sessionState?.libraries || {};

    const browserTypeValue = firstMeaningfulValue(
      browserState.browser_type,
      detail.browser_type,
      domBrowserState.browser_type,
      domBrowserState.browser,
      domState.browser,
      libraryState.browser_type,
      summary.browser_type
    );

    const currentUrlValue = firstMeaningfulValue(
      browserState.current_url,
      detail.current_url,
      domBrowserState.current_url,
      domState.url,
      libraryState.current_url,
      summary.current_url
    );

    const inferredLibraries = inferLibrariesFromSteps();

    const libraryCandidates = [
      ...(Array.isArray(detail.imported_libraries) ? detail.imported_libraries : []),
      ...(Array.isArray(detail.loaded_libraries) ? detail.loaded_libraries : []),
      ...(Array.isArray(detail.search_order) ? detail.search_order : []),
      ...(Array.isArray(libraryState.loaded) ? libraryState.loaded : []),
      ...(Array.isArray(libraryState.available) ? libraryState.available : []),
      ...(Array.isArray(summary.libraries) ? summary.libraries : []),
      ...Array.from(inferredLibraries),
    ]
      .map((lib) => (typeof lib === "string" ? lib.trim() : lib))
      .filter((lib) => typeof lib === "string" && lib.length > 0);

    const libraries = Array.from(new Set(libraryCandidates));

    let activeLibraryValue = firstMeaningfulValue(
      browserState.active_library,
      detail.active_library,
      libraryState.active,
      summary.active_library,
      libraries[0]
    );

    let browserDisplay = normalizeBrowserName(browserTypeValue);
    let currentUrl = normalizeUrl(currentUrlValue);
    let activeLibrary = activeLibraryValue ? activeLibraryValue : "—";

    if (browserDisplay === "—") {
      browserDisplay = normalizeBrowserName(
        firstMeaningfulValue(
          state.sessionVariables?.browser,
          state.sessionVariables?.BROWSER,
          state.sessionVariables?.browser_type,
          state.sessionVariables?.BROWSER_TYPE,
          inferBrowserFromSteps()
        )
      );
    }

    if (currentUrl === "—") {
      currentUrl = normalizeUrl(
        firstMeaningfulValue(
          state.sessionVariables?.current_url,
          state.sessionVariables?.CURRENT_URL,
          inferUrlFromSteps()
        )
      );
    }

    if (activeLibrary === "—") {
      activeLibrary = firstMeaningfulValue(
        state.sessionVariables?.active_library,
        state.sessionVariables?.ACTIVE_LIBRARY,
        libraries[0],
        Array.from(inferredLibraries)[0]
      ) || "—";
      activeLibraryValue = activeLibrary;
    }

    const librariesDisplay = libraries.length ? libraries.join(", ") : "—";

    const items = [
      { label: "Session", icon: "hash", value: detail.session_id },
      { label: "Platform", icon: PLATFORM_ICON[detail.platform_type] || "cpu", value: humanizePlatform(detail.platform_type) },
      { label: "Active Library", icon: "aperture", value: activeLibrary },
      { label: "Browser", icon: "navigation", value: browserDisplay },
      { label: "Current URL", icon: "link", value: currentUrl },
      { label: "Libraries", icon: "package", value: librariesDisplay },
      { label: "Steps", icon: "list", value: detail.step_count || 0 },
      {
        label: "Created",
        icon: "calendar",
        value: detail.created_at ? new Date(detail.created_at).toLocaleString() : "—",
      },
      {
        label: "Last activity",
        icon: "clock",
        value: detail.last_activity ? new Date(detail.last_activity).toLocaleString() : "—",
      },
    ];

    const fragment = document.createDocumentFragment();
    items.forEach(({ label, icon, value }) => {
      const chip = document.createElement("div");
      chip.className = "meta-chip";
      const labelEl = document.createElement("span");
      labelEl.innerHTML = `<i data-feather="${icon}"></i>${label}`;
      const valueEl = document.createElement("strong");
      valueEl.textContent = value;
      chip.append(labelEl, valueEl);
      fragment.appendChild(chip);
    });
    elements.sessionMeta.appendChild(fragment);
    refreshIcons();
  }

  function enrichSessionDetail(detail, sessionState) {
    if (!detail) {
      return detail;
    }

    const next = { ...detail };
    next.browser_state = { ...(detail.browser_state || {}) };
    const summary = { ...(detail.summary || {}) };

    const applySummary = () => {
      if (summary.current_url && !next.browser_state.current_url) {
        next.browser_state.current_url = summary.current_url;
      }
      if (summary.browser_type && !next.browser_state.browser_type) {
        next.browser_state.browser_type = summary.browser_type;
      }
      if (summary.active_library && !next.browser_state.active_library) {
        next.browser_state.active_library = summary.active_library;
      }
      if (Array.isArray(summary.libraries) && summary.libraries.length) {
        next.imported_libraries = Array.from(
          new Set([...(next.imported_libraries || []), ...summary.libraries])
        );
      }
    };

    applySummary();

    if (sessionState) {
      const domState = sessionState.dom || {};
      const domBrowser = domState.browser_state || {};
      const libraryState = sessionState.libraries || {};

      if (domState.browser && !next.browser_state.browser_type) {
        next.browser_state.browser_type = domState.browser;
      }
      if (domState.url && !next.browser_state.current_url) {
        next.browser_state.current_url = domState.url;
      }
      if (domBrowser.browser_type) {
        next.browser_state.browser_type = domBrowser.browser_type;
      }
      if (domBrowser.current_url) {
        next.browser_state.current_url = domBrowser.current_url;
      }
      if (domBrowser.active_library) {
        next.browser_state.active_library = domBrowser.active_library;
      }
      if (libraryState.active && !next.browser_state.active_library) {
        next.browser_state.active_library = libraryState.active;
      }

      const libraries = new Set(next.imported_libraries || []);
      if (Array.isArray(libraryState.loaded)) {
        libraryState.loaded.forEach((lib) => {
          if (lib) libraries.add(lib);
        });
      }
      if (Array.isArray(libraryState.available)) {
        libraryState.available.forEach((lib) => {
          if (lib) libraries.add(lib);
        });
      }
      if (libraries.size) {
        next.imported_libraries = Array.from(libraries);
        summary.libraries = Array.from(libraries);
      }
    }

    next.browser_type = next.browser_state.browser_type || next.browser_type;
    next.current_url = next.browser_state.current_url || next.current_url;
    next.active_library = next.browser_state.active_library || next.active_library;

    summary.active_library = next.browser_state.active_library || summary.active_library;
    summary.browser_type = next.browser_state.browser_type || summary.browser_type;
    summary.current_url = next.browser_state.current_url || summary.current_url;
    if (next.imported_libraries?.length) {
      summary.libraries = Array.from(new Set(next.imported_libraries));
    }

    next.summary = summary;

    return next;
  }

  function renderVariables(variables) {
    const container = elements.sessionVariables;
    container.innerHTML = "";
    const combined = {
      ...filterUserVariables(variables || {}),
      ...filterUserVariables(state.derivedVariables || {}),
    };
    const entries = Object.entries(combined);
    if (!entries.length) {
      const empty = document.createElement("div");
      empty.className = "muted";
      empty.textContent = "No variables captured.";
      container.appendChild(empty);
      return;
    }
    const fragment = document.createDocumentFragment();
    entries.forEach(([name, value]) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.innerHTML = `<i data-feather="sliders"></i>${name}: ${formatValueForDisplay(value)}`;
      fragment.appendChild(chip);
    });
    container.appendChild(fragment);
    refreshIcons();
  }

  function updateSteps(steps) {
    state.sessionSteps = steps;
    state.sessionStepMap = new Map(steps.map((step) => [step.step_id, step]));

    const incomingOrder = steps.map((step) => step.step_id);
    const retainedOrder = state.stepOrder.filter((id) => state.sessionStepMap.has(id));
    const appended = incomingOrder.filter((id) => !retainedOrder.includes(id));
    state.stepOrder = [...retainedOrder, ...appended];

    for (const id of Array.from(state.disabledSteps)) {
      if (!state.sessionStepMap.has(id)) {
        state.disabledSteps.delete(id);
      }
    }

    for (const id of Array.from(state.editedSteps.keys())) {
      if (!state.sessionStepMap.has(id)) {
        state.editedSteps.delete(id);
      }
    }

    const derived = {};
    steps.forEach((step) => {
      const names = extractAssignedVariableNames(step);
      if (!names.length) {
        return;
      }
      const valueSources =
        step.variables && typeof step.variables === "object" && !Array.isArray(step.variables)
          ? step.variables
          : null;
      const resultValue = step.result;
      names.forEach((name, idx) => {
        let value = undefined;
        if (valueSources && Object.prototype.hasOwnProperty.call(valueSources, name)) {
          value = valueSources[name];
        } else if (Array.isArray(resultValue)) {
          value = resultValue[idx] ?? resultValue[0];
        } else if (resultValue !== undefined) {
          value = resultValue;
        }
        if (value !== undefined) {
          derived[name] = value;
        }
      });
    });
    state.derivedVariables = derived;

    renderStepsFromState();
  }

  function createIconStack(status) {
    const normalized = (status || "").toLowerCase();
    const { icon, stack } = STATUS_ICON[normalized] || { icon: "minus", stack: "" };
    const stackEl = document.createElement("div");
    stackEl.className = stack ? `icon-stack ${stack}` : "icon-stack";
    stackEl.innerHTML = `<i data-feather="${icon}"></i>`;
    return stackEl;
  }

  function setupStepDrag(card, stepId) {
    card.dataset.stepId = stepId;
    card.style.touchAction = "none";
    card.addEventListener("pointerdown", (event) => beginStepDrag(event, card, stepId));
  }

  function beginStepDrag(event, card, stepId) {
    if (!elements.sessionSteps) {
      return;
    }
    if (event.button !== 0 && event.pointerType !== "touch" && event.pointerType !== "pen") {
      return;
    }
    if (event.target.closest("button") || event.target.closest(".step-controls")) {
      return;
    }
    try {
      card.setPointerCapture(event.pointerId);
    } catch (error) {
      /* Older browsers may throw; ignore. */
    }
    stepDragContext = {
      status: "pending",
      pointerId: event.pointerId,
      card,
      stepId,
      container: elements.sessionSteps,
      startX: event.clientX,
      startY: event.clientY,
    };
  }

  function activateStepDrag(event) {
    if (!stepDragContext || stepDragContext.status !== "pending") {
      return;
    }
    const { card, container } = stepDragContext;
    const cardRect = card.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();
    const placeholder = document.createElement("div");
    placeholder.className = "step-placeholder";
    placeholder.style.height = `${cardRect.height}px`;
    placeholder.style.width = `${cardRect.width}px`;
    card.parentElement.insertBefore(placeholder, card);

    const offsetY = event.clientY - cardRect.top;
    const offsetX = event.clientX - cardRect.left;

    card.classList.add("dragging");
    card.style.width = `${cardRect.width}px`;
    card.style.height = `${cardRect.height}px`;
    card.style.position = "absolute";
    card.style.top = `${event.clientY - offsetY - containerRect.top + container.scrollTop}px`;
    card.style.left = `${event.clientX - offsetX - containerRect.left + container.scrollLeft}px`;
    card.style.zIndex = "25";
    card.style.pointerEvents = "none";
    card.style.cursor = "grabbing";

    container.classList.add("drag-active");
    container.appendChild(card);

    stepDragContext = {
      ...stepDragContext,
      status: "active",
      placeholder,
      offsetY,
      offsetX,
    };
  }

  function updateStepDragPosition(event) {
    if (!stepDragContext || stepDragContext.status !== "active") {
      return;
    }
    const { card, container, placeholder, offsetY, offsetX } = stepDragContext;
    const containerRect = container.getBoundingClientRect();
    const top = event.clientY - containerRect.top + container.scrollTop - offsetY;
    const left = event.clientX - containerRect.left + container.scrollLeft - offsetX;

    card.style.top = `${top}px`;
    card.style.left = `${left}px`;

    maybeAutoScroll(event, container, containerRect);

    const siblings = Array.from(container.querySelectorAll(".step-card:not(.dragging)"));
    let inserted = false;
    for (const sibling of siblings) {
      const rect = sibling.getBoundingClientRect();
      if (event.clientY < rect.top + rect.height / 2) {
        if (placeholder.nextSibling !== sibling) {
          container.insertBefore(placeholder, sibling);
        }
        inserted = true;
        break;
      }
    }
    if (!inserted) {
      container.appendChild(placeholder);
    }
  }

  function maybeAutoScroll(event, container, containerRect) {
    const edge = 40;
    const delta = 16;
    if (event.clientY < containerRect.top + edge) {
      container.scrollTop = Math.max(container.scrollTop - delta, 0);
    } else if (event.clientY > containerRect.bottom - edge) {
      container.scrollTop += delta;
    }
  }

  function finishStepDrag(event, cancelled = false) {
    if (!stepDragContext || stepDragContext.pointerId !== event.pointerId) {
      return;
    }
    const { card, container, placeholder, status } = stepDragContext;
    try {
      card.releasePointerCapture(event.pointerId);
    } catch (error) {
      /* ignore */
    }

    const shouldRerender = status === "active" && !cancelled;

    if (shouldRerender) {
      if (placeholder) {
        container.insertBefore(card, placeholder);
      }
      syncStepOrderFromDom();
    }

    if (placeholder && placeholder.parentElement) {
      placeholder.parentElement.removeChild(placeholder);
    }

    card.classList.remove("dragging");
    card.style.position = "";
    card.style.top = "";
    card.style.left = "";
    card.style.zIndex = "";
    card.style.width = "";
    card.style.height = "";
    card.style.pointerEvents = "";
    card.style.cursor = "";

    container.classList.remove("drag-active");
    stepDragContext = null;

    if (shouldRerender) {
      renderStepsFromState();
    } else {
      refreshIcons();
    }
  }

  function syncStepOrderFromDom() {
    if (!elements.sessionSteps) {
      return;
    }
    const orderedRaw = Array.from(elements.sessionSteps.querySelectorAll('.step-card'))
      .map((el) => el.dataset.stepId)
      .filter(Boolean);

    const seen = new Set();
    const normalized = [];
    orderedRaw.forEach((id) => {
      if (!id || seen.has(id) || !state.sessionStepMap.has(id)) {
        return;
      }
      seen.add(id);
      normalized.push(id);
    });

    state.sessionSteps.forEach((step) => {
      if (!seen.has(step.step_id)) {
        seen.add(step.step_id);
        normalized.push(step.step_id);
      }
    });

    if (!normalized.length) {
      normalized.push(...state.sessionSteps.map((step) => step.step_id));
    }

    state.stepOrder = normalized;
    updateSuiteStatus();
  }

  function handleStepDragMove(event) {
    if (!stepDragContext || stepDragContext.pointerId !== event.pointerId) {
      return;
    }
    if (stepDragContext.status === "pending") {
      const dx = Math.abs(event.clientX - stepDragContext.startX);
      const dy = Math.abs(event.clientY - stepDragContext.startY);
      if (dx > 4 || dy > 4) {
        activateStepDrag(event);
      }
      return;
    }
    if (stepDragContext.status === "active") {
      event.preventDefault();
      updateStepDragPosition(event);
    }
  }

  function handleStepDragEnd(event) {
    if (!stepDragContext || stepDragContext.pointerId !== event.pointerId) {
      return;
    }
    finishStepDrag(event, false);
  }

  function handleStepDragCancel(event) {
    if (!stepDragContext || stepDragContext.pointerId !== event.pointerId) {
      return;
    }
    finishStepDrag(event, true);
  }

  function renderStepsFromState() {
    const container = elements.sessionSteps;
    container.innerHTML = "";
    const availableStepIds = state.stepOrder.filter((id) => getDisplayStep(id));

    if (!availableStepIds.length) {
      const empty = document.createElement("div");
      empty.className = "muted";
      empty.textContent = "No successful steps recorded.";
      container.appendChild(empty);
      refreshIcons();
      updateSuiteStatus();
      return;
    }

    const fragment = document.createDocumentFragment();
    availableStepIds.forEach((stepId, index) => {
      const step = getDisplayStep(stepId);
      if (!step) {
        return;
      }
      const disabled = state.disabledSteps.has(stepId);

      const card = document.createElement("article");
      card.className = "step-card";
      if (disabled) {
        card.classList.add("inactive");
      }
      setupStepDrag(card, stepId);

      const row = document.createElement("div");
      row.className = "step-row";
      row.appendChild(createIconStack(step.status));

      const label = document.createElement("span");
      label.className = "step-label";
      label.textContent = `${index + 1}. ${step.keyword}`;
      row.appendChild(label);

      if (step.arguments && step.arguments.length) {
        const args = document.createElement("span");
        args.className = "step-args";
        args.textContent = step.arguments.join(" • ");
        args.title = step.arguments.join(", ");
        row.appendChild(args);
      }

      const position = document.createElement("span");
      position.className = "step-id";
      position.innerHTML = `<i data-feather="hash"></i>${step.step_id.slice(0, 6)}`;
      row.appendChild(position);

      card.appendChild(row);

      const assignments = getAssignedVariables(step);
      if (assignments.length) {
        const chips = document.createElement("div");
        chips.className = "assignment-chips";
        assignments.forEach((name) => {
          const chip = document.createElement("span");
          chip.className = "chip";
          const value = getAssignedVariableValue(step, name);
          chip.innerHTML = `<i data-feather="clipboard"></i>${name}: ${formatValueForDisplay(value)}`;
          chips.appendChild(chip);
        });
        card.appendChild(chips);
      }

      const controls = document.createElement("div");
      controls.className = "step-controls";

      const toggleBtn = document.createElement("button");
      toggleBtn.className = "ghost-button step-control-button";
      toggleBtn.title = disabled ? "Enable step" : "Disable step";
      toggleBtn.textContent = disabled ? "Enable" : "Disable";
      toggleBtn.addEventListener("click", () => toggleStep(stepId, disabled));

      const editBtn = document.createElement("button");
      editBtn.className = "ghost-button step-control-button";
      editBtn.title = "Edit keyword & arguments";
      editBtn.textContent = "Edit";
      editBtn.addEventListener("click", () => editStep(stepId));

      const upBtn = document.createElement("button");
      upBtn.className = "ghost-button step-control-button";
      upBtn.title = "Move up";
      upBtn.textContent = "▲";
      upBtn.disabled = index === 0;
      upBtn.addEventListener("click", () => moveStep(stepId, -1));

      const downBtn = document.createElement("button");
      downBtn.className = "ghost-button step-control-button";
      downBtn.title = "Move down";
      downBtn.textContent = "▼";
      downBtn.disabled = index === availableStepIds.length - 1;
      downBtn.addEventListener("click", () => moveStep(stepId, 1));

      controls.append(toggleBtn, editBtn, upBtn, downBtn);
      card.appendChild(controls);

      fragment.appendChild(card);
    });

    container.appendChild(fragment);
    refreshIcons();
    updateSuiteStatus();
  }

  function renderSuitePreview(preview) {
    const codeEl = elements.suiteCode;
    if (!codeEl) {
      return;
    }
    if (!preview || !preview.success) {
      const message =
        preview && preview.error ? preview.error : "Suite preview unavailable.";
      state.suiteText = "";
      codeEl.textContent = message;
      if (elements.copySuite) {
        elements.copySuite.disabled = true;
        elements.copySuite.classList.remove("copied");
      }
      if (window.Prism && typeof window.Prism.highlightElement === "function") {
        window.Prism.highlightElement(codeEl);
      }
      clearSuiteNotice();
      const summary = getSuiteStatusSummary();
      elements.suiteStatus.textContent = preview && preview.error
        ? summary
          ? `Preview unavailable • ${summary}`
          : "Preview unavailable"
        : summary;
      return;
    }
    const text = preview.rf_text || "";
    state.suiteText = text;
    codeEl.textContent = text;
    if (elements.copySuite) {
      elements.copySuite.disabled = !text;
      elements.copySuite.classList.remove("copied");
    }
    if (window.Prism && typeof window.Prism.highlightElement === "function") {
      window.Prism.highlightElement(codeEl);
    }
    clearSuiteNotice();
    refreshIcons();
    updateSuiteStatus();
  }

  function renderSessionsEmptyState() {
    elements.sessionMeta.innerHTML = "";
    elements.sessionVariables.innerHTML = "";
    elements.sessionSteps.innerHTML = "";
    elements.eventsLog.innerHTML = "";
    if (elements.suiteCode) {
      elements.suiteCode.textContent = "";
    }
    if (elements.copySuite) {
      elements.copySuite.disabled = true;
      elements.copySuite.classList.remove("copied");
    }
    state.suiteText = "";
    clearSuiteNotice();
  }

  function updateCompactButton() {
    if (!elements.toggleCompact) {
      return;
    }
    if (state.compactMode) {
      elements.toggleCompact.dataset.mode = "compact";
      elements.toggleCompact.innerHTML = `<i data-feather="maximize-2"></i>Expand`;
    } else {
      elements.toggleCompact.dataset.mode = "expanded";
      elements.toggleCompact.innerHTML = `<i data-feather="minimize-2"></i>Compact`;
    }
    refreshIcons();
  }

  function setCompactMode(enabled) {
    state.compactMode = Boolean(enabled);
    document.body.classList.toggle("compact-mode", state.compactMode);
    updateCompactButton();
  }

  async function copySuiteToClipboard() {
    if (!state.suiteText) {
      setSuiteNotice("Nothing to copy", 2000);
      return;
    }
    const text = state.suiteText;
    const button = elements.copySuite;
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        const textarea = document.createElement("textarea");
        textarea.value = text;
        textarea.setAttribute("readonly", "");
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.focus();
        textarea.select();
        const success = document.execCommand("copy");
        document.body.removeChild(textarea);
        if (!success) {
          throw new Error("execCommand returned false");
        }
      }
      if (button) {
        button.classList.add("copied");
        window.setTimeout(() => {
          if (button) {
            button.classList.remove("copied");
          }
        }, 1500);
      }
      setSuiteNotice("Copied to clipboard");
    } catch (error) {
      console.error("Failed to copy suite preview", error);
      if (button) {
        button.classList.remove("copied");
      }
      setSuiteNotice("Copy failed", 2500);
    }
  }

  function appendEvent(event) {
    if (!event || !event.timestamp) {
      return;
    }
    if (state.eventLog.some((item) => item.timestamp === event.timestamp && item.event_type === event.event_type && item.step_id === event.step_id)) {
      return;
    }
    state.lastEventTimestamp = event.timestamp;
    state.eventLog.unshift(event);
    if (state.eventLog.length > 100) {
      state.eventLog.pop();
    }

    renderEvents();
  }

  function renderEvents() {
    const list = elements.eventsLog;
    list.innerHTML = "";
    if (!state.eventLog.length) {
      const empty = document.createElement("li");
      empty.className = "muted";
      empty.textContent = "No events captured yet.";
      list.appendChild(empty);
      return;
    }
    const fragment = document.createDocumentFragment();
    state.eventLog.forEach((event) => {
      const item = document.createElement("li");
      item.className = "event-item";
      item.appendChild(createIconStack(event.event_type === "step_failed" ? "fail" : "pass"));

      const body = document.createElement("div");
      const title = document.createElement("strong");
      title.innerHTML = `<i data-feather="hash"></i>${formatEventTitle(event)}`;

      const timestamp = document.createElement("small");
      timestamp.innerHTML = `<i data-feather="clock"></i>${new Date(event.timestamp).toLocaleTimeString()}`;

      const detailsText = formatEventDetails(event);
      if (detailsText) {
        const details = document.createElement("p");
        details.textContent = detailsText;
        body.append(title, timestamp, details);
      } else {
        body.append(title, timestamp);
      }
      item.appendChild(body);
      fragment.appendChild(item);
    });
    list.appendChild(fragment);
    refreshIcons();
  }

  function formatEventTitle(event) {
    const type = event.event_type || "event";
    const session = event.session_id ? `Session ${event.session_id.slice(0, 8)}` : null;
    const readableType = type.replace(/_/g, " ");
    return session ? `${session} • ${readableType}` : readableType;
  }

  function formatEventDetails(event) {
    if (!event.payload) {
      return "";
    }
    const { keyword, arguments: args, error } = event.payload;
    if (error) {
      return error;
    }
    if (keyword) {
      const argText = (args || []).join(", ");
      return argText ? `${keyword}(${argText})` : keyword;
    }
    const parts = Object.entries(event.payload)
      .map(([k, v]) => `${k}: ${typeof v === "object" ? JSON.stringify(v) : v}`)
      .join(" • ");
    return parts || "";
  }

  function updateSuiteStatus() {
    if (!elements.suiteStatus) {
      return;
    }
    const summary = getSuiteStatusSummary();
    if (state.suiteNotice) {
      elements.suiteStatus.textContent = summary
        ? `${state.suiteNotice} • ${summary}`
        : state.suiteNotice;
    } else {
      elements.suiteStatus.textContent = summary;
    }
  }

  function clearSuiteNotice() {
    if (state.suiteNoticeTimer) {
      clearTimeout(state.suiteNoticeTimer);
      state.suiteNoticeTimer = null;
    }
    state.suiteNotice = null;
    updateSuiteStatus();
  }

  function setSuiteNotice(message, duration = 2000) {
    if (state.suiteNoticeTimer) {
      clearTimeout(state.suiteNoticeTimer);
      state.suiteNoticeTimer = null;
    }
    state.suiteNotice = message;
    updateSuiteStatus();
    if (message && duration > 0) {
      state.suiteNoticeTimer = window.setTimeout(() => {
        state.suiteNoticeTimer = null;
        state.suiteNotice = null;
        updateSuiteStatus();
      }, duration);
    }
  }

  function getSuiteStatusSummary() {
    const parts = [];
    if (state.disabledSteps.size) {
      parts.push(`${state.disabledSteps.size} disabled`);
    }
    if (state.editedSteps.size) {
      parts.push(`${state.editedSteps.size} edited`);
    }
    const userVars = Object.keys(filterUserVariables(state.sessionVariables || {}));
    if (userVars.length) {
      parts.push(`${userVars.length} variables`);
    }
    return parts.join(" • ");
  }

  async function loadSessions() {
    try {
      const payload = await fetchJSON("api/sessions/");
      state.sessions = payload.sessions || [];
      renderSessions();

      if (!state.sessions.length) {
        state.selectedSessionId = null;
        state.sessionPanel.dataset.empty = "true";
        state.sessionActions.hidden = true;
        renderSessionsEmptyState();
        return;
      }

      const selectedStillExists = state.selectedSessionId && state.sessions.some((s) => s.session_id === state.selectedSessionId);
      if (!state.selectedSessionId || !selectedStillExists) {
        await selectSession(state.sessions[0].session_id);
      } else if (state.selectedSessionId && selectedStillExists) {
        await loadSessionDetails(state.selectedSessionId, { includeSuite: false });
      }
    } catch (error) {
      console.error("Failed to load sessions", error);
    }
  }

  async function selectSession(sessionId) {
    state.selectedSessionId = sessionId;
    state.stepOrder = [];
    state.disabledSteps.clear();
    state.editedSteps.clear();

    if (elements.sessionPanel) {
      elements.sessionPanel.dataset.empty = "false";
    }
    if (elements.sessionActions) {
      elements.sessionActions.hidden = false;
    }

    document.querySelectorAll(".session-card.active").forEach((card) => card.classList.remove("active"));
    const newCard = document.querySelector(`.session-card[data-session-id="${sessionId}"]`);
    if (newCard) {
      newCard.classList.add("active");
    }

    await loadSessionDetails(sessionId, { includeSuite: true });
  }

  async function loadSessionDetails(sessionId, options = {}) {
    const includeSuite = options.includeSuite !== false;
    try {
      const requests = [
        fetchJSON(`api/sessions/${sessionId}/`),
        fetchJSON(`api/sessions/${sessionId}/steps/`),
        fetchJSON(`api/sessions/${sessionId}/variables/`),
        fetchJSON(`api/sessions/${sessionId}/state/?type=all`).catch(() => null),
      ];
      if (includeSuite) {
        requests.push(fetchJSON(`api/sessions/${sessionId}/suite/`));
      }
      const results = await Promise.all(requests);
      const detail = results[0];
      const steps = results[1];
      const variables = results[2];
      const sessionState = results[3];
      const preview = includeSuite ? results[4] : null;

      state.sessionDetails = detail;
      const baseVariables = {
        ...(sessionState?.variables || {}),
        ...(variables.variables || {}),
      };
      state.sessionVariables = baseVariables;

      const enrichedDetail = sessionState
        ? enrichSessionDetail({ ...state.sessionDetails }, sessionState)
        : state.sessionDetails;
      state.sessionDetails = enrichedDetail;

      elements.sessionTitle.textContent = `Session ${sessionId.slice(0, 8)}`;
      elements.sessionSubtitle.textContent = `Last activity: ${enrichedDetail.last_activity || "—"}`;
      renderSessionMeta(enrichedDetail, sessionState);
      renderVariables(state.sessionVariables);
      updateSteps(steps.steps || []);
      renderEvents();
      if (includeSuite) {
        renderSuitePreview(preview);
      }
      refreshIcons();
    } catch (error) {
      console.error("Failed to load session details", error);
    }
  }

  async function previewSuiteWithOverrides() {
    if (!state.selectedSessionId) {
      return;
    }
    try {
      elements.suiteStatus.textContent = "Rendering preview…";
      const payload = buildSuiteOverridesPayload();
      const preview = await fetchJSON(
        `api/sessions/${state.selectedSessionId}/suite/`,
        payload
          ? {
              method: "POST",
              body: JSON.stringify(payload),
            }
          : undefined
      );
      renderSuitePreview(preview);
    } catch (error) {
      elements.suiteStatus.textContent = error.message;
    }
  }

  function buildSuiteOverridesPayload() {
    const defaultOrder = state.sessionSteps.map((step) => step.step_id);
    const orderChanged = JSON.stringify(defaultOrder) !== JSON.stringify(state.stepOrder);
    const disabled = Array.from(state.disabledSteps);
    const edits = Array.from(state.editedSteps.entries()).map(([stepId, data]) => ({
      step_id: stepId,
      keyword: data.keyword,
      arguments: data.arguments,
    }));

    if (!orderChanged && disabled.length === 0 && edits.length === 0) {
      return null;
    }

    const payload = {};
    if (orderChanged) {
      payload.order = state.stepOrder;
    }
    if (disabled.length) {
      payload.excluded = disabled;
    }
    if (edits.length) {
      payload.edits = edits;
    }
    return payload;
  }

  function toggleStep(stepId, isDisabled) {
    if (isDisabled) {
      state.disabledSteps.delete(stepId);
    } else {
      state.disabledSteps.add(stepId);
    }
    renderStepsFromState();
  }

  function moveStep(stepId, delta) {
    const index = state.stepOrder.indexOf(stepId);
    const target = index + delta;
    if (index < 0 || target < 0 || target >= state.stepOrder.length) {
      return;
    }
    const order = [...state.stepOrder];
    const [item] = order.splice(index, 1);
    order.splice(target, 0, item);
    state.stepOrder = order;
    renderStepsFromState();
  }

  function editStep(stepId) {
    const current = getDisplayStep(stepId);
    if (!current) {
      return;
    }
    const newKeyword = window.prompt("Keyword", current.keyword || "");
    if (newKeyword === null) {
      return;
    }
    const newArgsInput = window.prompt(
      "Arguments (comma separated)",
      (current.arguments || []).join(", ")
    );
    if (newArgsInput === null) {
      return;
    }
    const updatedArgs = newArgsInput
      .split(",")
      .map((item) => item.trim())
      .filter((item) => item.length > 0);

    const original = state.sessionStepMap.get(stepId);
    if (!original) {
      return;
    }

    const keywordChanged = newKeyword !== original.keyword;
    const argsChanged = JSON.stringify(updatedArgs) !== JSON.stringify(original.arguments || []);

    if (!keywordChanged && !argsChanged) {
      state.editedSteps.delete(stepId);
    } else {
      state.editedSteps.set(stepId, {
        keyword: newKeyword,
        arguments: updatedArgs,
        assigned_variables: current.assigned_variables,
      });
    }

    renderStepsFromState();
  }

  function scheduleSessionRefresh(sessionId) {
    if (state.selectedSessionId !== sessionId) {
      return;
    }
    if (state.refreshTimer) {
      return;
    }
    state.refreshTimer = setTimeout(async () => {
      state.refreshTimer = null;
      await loadSessionDetails(sessionId, { includeSuite: false });
    }, 400);
  }

  function startEventStream() {
    try {
      const source = new EventSource(buildUrl("api/events/"));
      source.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          appendEvent(payload);
          if (payload.event_type === "session_created" || payload.event_type === "session_removed") {
            loadSessions();
          } else if (payload.session_id) {
            scheduleSessionRefresh(payload.session_id);
          }
        } catch (error) {
          console.error("Failed to parse event payload", error);
        }
      };
    } catch (error) {
      console.error("Event stream unavailable", error);
    }
  }

  async function loadRecentEvents() {
    try {
      const params = "?limit=100";
      const data = await fetchJSON(`api/events/recent/${params}`);
      const events = (data.events || []).sort(
        (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
      );
      state.eventLog = events;
      renderEvents();
    } catch (error) {
      console.error("Failed to load events", error);
    }
  }

  function setupEventListeners() {
    if (elements.refreshSessions) {
      elements.refreshSessions.addEventListener("click", () => loadSessions());
    }
    if (elements.headerRefresh) {
      elements.headerRefresh.addEventListener("click", () => loadSessions());
    }
    if (elements.headerBuildSuite) {
      elements.headerBuildSuite.addEventListener("click", () => previewSuiteWithOverrides());
    }
    if (elements.reloadSession) {
      elements.reloadSession.addEventListener("click", () => {
        if (state.selectedSessionId) {
          loadSessionDetails(state.selectedSessionId, { includeSuite: true });
        }
      });
    }
    if (elements.toggleCompact) {
      elements.toggleCompact.addEventListener("click", () =>
        setCompactMode(!state.compactMode)
      );
    }
    if (elements.previewSuiteBtn) {
      elements.previewSuiteBtn.addEventListener("click", () => previewSuiteWithOverrides());
    }
    if (elements.copySuite) {
      elements.copySuite.addEventListener("click", () => copySuiteToClipboard());
    }
  }

  function initialize() {
    setCompactMode(false);
    setupEventListeners();
    loadSessions();
    loadRecentEvents();
    startEventStream();
    refreshIcons();
  }

  document.addEventListener("pointermove", handleStepDragMove, { passive: false });
  document.addEventListener("pointerup", handleStepDragEnd);
  document.addEventListener("pointercancel", handleStepDragCancel);

  initialize();
})();
