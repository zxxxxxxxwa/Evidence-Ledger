"use strict";

const LANGUAGE_KEY = "evidenceLedger.language";
const WORKSPACE_KEY = "evidenceLedger.workspace";
const VALID_LANGUAGES = new Set(["en", "zh"]);
const TRACE_MODAL_QUERY = globalThis.matchMedia("(max-width: 1080px)");
const TRACE_BACKGROUND_SELECTORS = [
  ".topbar",
  ".development-ribbon",
  ".step-rail",
  ".workspace",
  ".status-bar"
];

const state = {
  dictionary: { en: {}, zh: {} },
  language: "en",
  workspaceId: "",
  profile: null,
  materials: [],
  assessment: null,
  reviews: [],
  framework: { dimensions: [], indicators: [] },
  resources: { resources: [] },
  system: {},
  activeView: "context",
  lastTraceTrigger: null,
  activeTraceIndicator: null,
  reviewDrafts: {}
};

const qs = (selector, root = document) => root.querySelector(selector);
const qsa = (selector, root = document) => Array.from(root.querySelectorAll(selector));

function newWorkspaceId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, character => {
    const random = Math.random() * 16 | 0;
    const value = character === "x" ? random : (random & 0x3) | 0x8;
    return value.toString(16);
  });
}

function ensureWorkspace() {
  let value = localStorage.getItem(WORKSPACE_KEY);
  const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
  if (!value || !uuidPattern.test(value)) {
    value = newWorkspaceId();
    localStorage.setItem(WORKSPACE_KEY, value);
  }
  state.workspaceId = value;
}

function t(key, values = {}) {
  const current = state.dictionary[state.language] || {};
  const fallback = state.dictionary.en || {};
  let text = current[key] ?? fallback[key] ?? key;
  for (const [name, value] of Object.entries(values)) {
    text = text.replaceAll(`{${name}}`, String(value));
  }
  return text;
}

function localText(value) {
  if (value && typeof value === "object") {
    return String(value[state.language] || value.en || value.zh || "");
  }
  return String(value || "");
}

function localizedKey(prefix, value, fallback = "") {
  const key = `${prefix}.${value}`;
  const translated = t(key);
  return translated === key ? (fallback || String(value || "")) : translated;
}

function flagLabel(code) {
  return localizedKey("flag", code, code);
}

function difficultyLabel(value) {
  return localizedKey("difficulty", value, value);
}

function resourceTitle(resource) {
  if (!resource) return "";
  return state.language === "zh" ? (resource.title_zh || resource.title) : (resource.title || resource.title_zh);
}

function resourceTitleById(resourceId) {
  const resource = (state.resources.resources || []).find(item => item.id === resourceId);
  return resourceTitle(resource) || resourceId;
}

function resourceLicense(resource) {
  return localizedKey("license", resource.id, resource.license);
}

function generationModeLabel(value) {
  const keys = {
    llm_rule_checked: "path.mode.llmRuleChecked",
    catalog_rule_fallback: "path.mode.catalogRuleFallback"
  };
  return t(keys[value] || "path.mode.unknown");
}

function frameworkMappingLabel(value) {
  const mappingKeys = {
    "Higher-ed assessment literacy": "framework.assessmentLiteracy",
    "Higher-ed inclusive teaching": "framework.inclusiveTeaching",
    "Higher-ed research capability": "framework.researchCapability",
    "Higher-ed teaching design": "framework.teachingDesign",
    "Professional engagement": "framework.professionalEngagement",
    "Research integrity": "framework.researchIntegrity",
    "UNESCO AI CFT: AI for professional development": "framework.aiProfessionalDevelopment",
    "UNESCO AI CFT: AI foundations and applications": "framework.aiFoundations",
    "UNESCO AI CFT: AI pedagogy": "framework.aiPedagogy",
    "UNESCO AI CFT: Ethics of AI": "framework.aiEthics",
    "UNESCO AI CFT: Human-centred mindset": "framework.humanCentred"
  };
  return mappingKeys[value] ? t(mappingKeys[value]) : value;
}

function locatorLabel(value) {
  const text = String(value || "");
  const characterMatch = /^chars\s+(\d+)-(\d+)$/i.exec(text);
  if (characterMatch) {
    return t("trace.locatorChars", { start: characterMatch[1], end: characterMatch[2] });
  }
  const slideMatch = /^slide\s+(\d+)$/i.exec(text);
  if (slideMatch) return t("trace.locatorSlide", { number: slideMatch[1] });
  return text;
}

function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function applyTranslations() {
  document.documentElement.lang = state.language === "zh" ? "zh-CN" : "en";
  document.title = t("app.title");
  qsa("[data-i18n]").forEach(node => {
    node.textContent = t(node.dataset.i18n);
  });
  qsa("[data-i18n-placeholder]").forEach(node => {
    node.setAttribute("placeholder", t(node.dataset.i18nPlaceholder));
  });
  qsa("[data-i18n-aria]").forEach(node => {
    node.setAttribute("aria-label", t(node.dataset.i18nAria));
  });
  qsa("[data-language]").forEach(button => {
    const active = button.dataset.language === state.language;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  });
}

function setStatus(message, isError = false) {
  const target = qs("#global-status");
  target.textContent = message;
  target.style.color = isError ? "var(--conflict)" : "var(--ink-soft)";
}

function reportLocalizedValidity(form) {
  qsa("input, select, textarea", form).forEach(control => {
    control.setCustomValidity("");
    const validity = control.validity;
    if (validity.valueMissing) {
      control.setCustomValidity(t("validation.required"));
    } else if (validity.rangeUnderflow || validity.rangeOverflow) {
      control.setCustomValidity(t("validation.range", {
        min: control.min || "—",
        max: control.max || "—"
      }));
    } else if (validity.typeMismatch || validity.badInput || validity.patternMismatch) {
      control.setCustomValidity(t("validation.invalid"));
    }
  });
  const valid = form.checkValidity();
  if (!valid) form.reportValidity();
  return valid;
}

function errorText(error) {
  const code = error?.code || "generic";
  return t(`error.${code}`) === `error.${code}` ? t("error.generic") : t(`error.${code}`);
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("X-Workspace-ID", state.workspaceId);
  if (options.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  let response;
  try {
    response = await fetch(path, { ...options, headers });
  } catch (networkError) {
    const error = new Error(t("error.network"));
    error.code = "network";
    throw error;
  }
  let payload = {};
  try {
    payload = await response.json();
  } catch (_) {
    payload = { error: { code: "generic" } };
  }
  if (!response.ok) {
    const error = new Error(errorText(payload.error));
    error.code = payload.error?.code || "generic";
    error.details = payload.error?.details;
    throw error;
  }
  return payload;
}

function showView(view) {
  state.activeView = view;
  qsa("[data-view-panel]").forEach(panel => {
    const active = panel.dataset.viewPanel === view;
    panel.hidden = !active;
    panel.classList.toggle("is-active", active);
  });
  qsa("[data-view]").forEach(button => {
    const active = button.dataset.view === view;
    button.classList.toggle("is-active", active);
    if (active) button.setAttribute("aria-current", "step");
    else button.removeAttribute("aria-current");
  });
  qs("#main-content").focus({ preventScroll: true });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function populateProfile() {
  const profile = state.profile;
  if (!profile) return;
  const form = qs("#context-form");
  for (const name of ["display_name", "discipline", "career_stage", "role", "weekly_minutes", "constraints"]) {
    if (form.elements[name]) form.elements[name].value = profile[name] ?? "";
  }
  form.elements.goals.value = (profile.goals || []).join("\n");
  form.elements.developmental_use_acknowledged.checked = Boolean(profile.developmental_use_acknowledged);
}

function renderSystem() {
  const rail = qs("#rail-system");
  rail.replaceChildren();
  const llm = element("div", "system-line");
  llm.append(
    element("span", `system-dot ${state.system.llm_configured ? "ready" : "missing"}`),
    element("span", "", t(state.system.llm_configured ? "system.llmReady" : "system.llmMissing"))
  );
  const local = element("div", "system-line");
  local.append(element("span", "system-dot ready"), element("span", "", t("system.localOnly")));
  rail.append(llm, local);

  const calibration = state.system.calibration || { expert_calibrated: false };
  const notice = qs("#calibration-notice");
  notice.classList.toggle("is-calibrated", Boolean(calibration.expert_calibrated));
  const titleKey = calibration.expert_calibrated ? "calibration.loadedTitle" : "calibration.provisionalTitle";
  const bodyKey = calibration.expert_calibrated ? "calibration.loadedBody" : "calibration.provisionalBody";
  notice.replaceChildren(element("strong", "", t(titleKey)), element("span", "", t(bodyKey)));
  const limitBytes = Number(state.system.max_file_bytes) || 10 * 1024 * 1024;
  const limitMb = Math.round((limitBytes / (1024 * 1024)) * 10) / 10;
  const fileHelp = qs("#file-help");
  if (fileHelp) fileHelp.textContent = t("materials.fileHelpDynamic", { size: limitMb });
}

function materialTypeLabel(value) {
  const map = {
    paper: "material.paper",
    research_project: "material.project",
    slides: "material.slides",
    lesson_plan: "material.lesson",
    teaching_achievement: "material.achievement",
    other: "material.other"
  };
  return t(map[value] || "material.other");
}

function contributionLabel(value) {
  const map = {
    sole: "contribution.sole",
    lead: "contribution.lead",
    substantial: "contribution.substantial",
    limited: "contribution.limited",
    unclear: "contribution.unclear"
  };
  return t(map[value] || "contribution.unclear");
}

function renderMaterials() {
  qs("#material-count").textContent = String(state.materials.length);
  const list = qs("#materials-list");
  list.replaceChildren();
  if (!state.materials.length) {
    list.append(element("p", "empty-state", t("materials.empty")));
    return;
  }
  state.materials.forEach(material => {
    const row = element("article", "register-row");
    const title = element("div", "register-title");
    title.append(element("strong", "", material.title), element("small", "", material.filename));
    const type = element("div", "register-meta", materialTypeLabel(material.material_type));
    type.append(document.createElement("br"), document.createTextNode(contributionLabel(material.contribution_level)));
    const meta = element("div", "register-meta");
    meta.append(
      document.createTextNode(t("materials.characters", { count: material.parse_meta?.characters || 0 })),
      document.createElement("br"),
      document.createTextNode(t("materials.chunks", { count: material.chunk_count || 0 }))
    );
    if ((material.safety_flags || []).length) {
      const flags = element("div", "flag-list");
      material.safety_flags.forEach(flag => {
        const chip = element("span", "flag-chip", flagLabel(flag.code));
        chip.title = t(
          String(flag.code || "").startsWith("POSSIBLE_PII_")
            ? "flag.piiHelp"
            : (flag.severity === "high" ? "flag.highHelp" : "flag.warningHelp")
        );
        flags.append(chip);
      });
      meta.append(document.createElement("br"), flags);
    }
    if (material.parse_meta?.truncated) {
      const truncation = element("span", "flag-chip", t("materials.truncated"));
      truncation.title = t("materials.truncated");
      meta.append(document.createElement("br"), truncation);
    }
    const remove = element("button", "quiet-button", t("common.delete"));
    remove.type = "button";
    remove.addEventListener("click", () => removeMaterial(material.id));
    row.append(title, type, meta, remove);
    list.append(row);
  });
}

function dimensionTitle(dimensionId) {
  const dimension = (state.framework.dimensions || []).find(item => item.id === dimensionId);
  if (!dimension) return dimensionId;
  return state.language === "zh" ? dimension.title_zh : dimension.title_en;
}

function indicatorTitle(indicatorId) {
  const fromResult = state.assessment?.result?.indicators?.find(item => item.indicator_id === indicatorId);
  if (fromResult) return state.language === "zh" ? fromResult.title_zh : fromResult.title_en;
  const indicator = (state.framework.indicators || []).find(item => item.id === indicatorId);
  if (!indicator) return indicatorId;
  return state.language === "zh" ? indicator.title_zh : indicator.title_en;
}

function stateSymbol(stateName) {
  return { estimable: "E", insufficient: "I", conflict: "C", not_applicable: "—" }[stateName] || "?";
}

function renderAssessment() {
  const ledger = qs("#evidence-ledger");
  const summary = qs("#state-summary");
  const status = qs("#assessment-status");
  ledger.replaceChildren();
  summary.replaceChildren();
  status.className = "assessment-status";
  status.textContent = "";

  const assessment = state.assessment;
  if (!assessment || !assessment.result) {
    if (assessment?.status === "failed" && assessment.error_code) {
      status.classList.add("is-error");
      status.textContent = errorText({ code: assessment.error_code });
    }
    ledger.append(element("p", "empty-state", t("evidence.noAssessment")));
    return;
  }
  if (assessment.status === "stale") {
    status.classList.add("is-error");
    status.textContent = t("evidence.stale");
  }

  const result = assessment.result;
  ["estimable", "insufficient", "conflict", "not_applicable"].forEach(name => {
    const item = element("div", `state-count is-${name}`);
    item.append(
      element("span", "state-symbol", stateSymbol(name)),
      element("strong", "", result.state_counts?.[name] || 0),
      element("span", "", t(`state.${name}`))
    );
    summary.append(item);
  });

  let lastDimension = null;
  result.indicators.forEach(decision => {
    if (decision.dimension !== lastDimension) {
      const divider = element("div", "dimension-divider", dimensionTitle(decision.dimension));
      ledger.append(divider);
      lastDimension = decision.dimension;
    }
    const row = element("button", "ledger-row");
    row.type = "button";
    row.dataset.indicatorId = decision.indicator_id;
    if (state.activeTraceIndicator === decision.indicator_id) state.lastTraceTrigger = row;
    const node = element("span", `state-node is-${decision.state}`);
    node.setAttribute("aria-hidden", "true");
    const name = element("span", "indicator-name");
    name.append(
      element("strong", "", indicatorTitle(decision.indicator_id)),
      element("small", "", decision.indicator_id)
    );
    const statusCell = element("span", "state-label");
    statusCell.append(
      element("strong", "", t(`state.${decision.state}`)),
      element("small", "", t("evidence.familyCount", { count: decision.family_count }))
    );
    const level = element("span", "level-cell");
    if (decision.state === "estimable" && decision.level) {
      const interval = decision.level_interval || [decision.level, decision.level];
      level.classList.add("conditional-level");
      level.textContent = interval[0] === interval[1]
        ? t("evidence.levelSingle", { level: decision.level })
        : t("evidence.levelValue", { level: decision.level, low: interval[0], high: interval[1] });
    } else {
      level.textContent = t("evidence.noScore");
    }
    const action = element("span", "action-cell", t(`route.${decision.route?.route || "no_action"}`));
    row.append(node, name, statusCell, level, action);
    row.addEventListener("click", event => openTrace(decision, event.currentTarget));
    ledger.append(row);
  });
}

function appendDefinition(list, term, description) {
  list.append(element("dt", "", term), element("dd", "", description));
}

function renderPaths() {
  const list = qs("#path-list");
  list.replaceChildren();
  const result = state.assessment?.result;
  if (!result || state.assessment?.status === "stale" || !(result.paths || []).length) {
    const message = state.assessment?.status === "stale"
      ? t("evidence.stale")
      : (result ? t("path.noActive") : t("path.empty"));
    list.append(element("p", "empty-state", message));
    return;
  }
  result.paths.forEach(path => {
    const group = element("article", "path-group");
    const meta = element("div", "path-meta");
    meta.append(
      element("span", "path-mode", t(`route.${path.route}`)),
      element("h2", "", indicatorTitle(path.indicator_id)),
      element("p", "", localText(path.summary)),
      element("p", "", t("path.total", { count: path.total_minutes || 0 })),
      element("p", "", generationModeLabel(path.generation_mode))
    );
    const steps = element("div", "path-steps");
    const citationMap = Object.fromEntries((path.citations || []).map(item => [item.resource_id, item]));
    (path.steps || []).forEach((step, index) => {
      const citation = citationMap[step.resource_id] || {};
      const stepNode = element("section", "path-step");
      stepNode.append(
        element("span", "step-number", t("path.step", { count: index + 1 })),
        element("h3", "", resourceTitle(citation))
      );
      const details = element("dl");
      appendDefinition(details, t("path.objective"), localText(step.objective));
      appendDefinition(details, t("path.reason"), localText(step.reason));
      appendDefinition(details, t("path.practice"), localText(step.practice_task));
      appendDefinition(details, t("path.completion"), localText(step.completion_evidence));
      stepNode.append(details);
      if (citation.url) {
        const link = element(
          "a",
          "source-link",
          t("path.openResource", {
            title: resourceTitle(citation),
            publisher: citation.publisher,
            count: step.duration_minutes
          })
        );
        link.href = citation.url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        stepNode.append(link, element("p", "resource-meta", t("path.sourceCheck", { date: citation.checked_at || "—" })));
      }
      steps.append(stepNode);
    });
    group.append(meta, steps);
    list.append(group);
  });
}

function renderReviews() {
  const list = qs("#review-list");
  qsa(".review-row[data-review-id]", list).forEach(row => {
    const select = qs("select", row);
    const note = qs("input", row);
    if (select || note) {
      state.reviewDrafts[row.dataset.reviewId] = {
        outcome: select?.value || "",
        note: note?.value || ""
      };
    }
  });
  list.replaceChildren();
  if (!state.reviews.length) {
    list.append(element("p", "empty-state", t("review.empty")));
    return;
  }
  state.reviews.forEach(review => {
    const row = element("article", "review-row");
    row.dataset.reviewId = review.id;
    const header = element("header");
    const heading = element("div");
    heading.append(
      element("h3", "", indicatorTitle(review.indicator_id)),
      element("span", "resource-meta", t("review.localOnly"))
    );
    header.append(heading, element("span", `status-pill ${review.status}`, t(`review.${review.status}`)));
    row.append(header, element("p", "", localText(review.reason?.rationale)));
    if (review.status === "open") {
      const help = element("p", "review-help", t("review.resolveHelp"));
      const fields = element("div", "review-fields");
      const selectLabel = element("label");
      selectLabel.append(element("span", "", t("review.outcome")));
      const select = document.createElement("select");
      select.append(new Option(t("review.chooseOutcome"), ""));
      select.append(new Option(t("review.clarified"), "clarified"));
      select.append(new Option(t("review.insufficient"), "insufficient"));
      select.append(new Option(t("review.confirmedConflict"), "confirmed_conflict"));
      select.append(new Option(t("review.withdrawn"), "withdrawn"));
      select.value = state.reviewDrafts[review.id]?.outcome || "";
      select.addEventListener("change", () => {
        state.reviewDrafts[review.id] = {
          outcome: select.value,
          note: state.reviewDrafts[review.id]?.note || ""
        };
      });
      selectLabel.append(select);
      const noteLabel = element("label");
      noteLabel.append(element("span", "", t("review.reviewerNote")));
      const note = document.createElement("input");
      note.maxLength = 1000;
      note.value = state.reviewDrafts[review.id]?.note || "";
      note.addEventListener("input", () => {
        state.reviewDrafts[review.id] = {
          outcome: state.reviewDrafts[review.id]?.outcome || select.value,
          note: note.value
        };
      });
      noteLabel.append(note);
      const button = element("button", "secondary-button", t("review.resolve"));
      button.type = "button";
      button.addEventListener("click", () => resolveReview(review.id, select.value, note.value, button));
      fields.append(selectLabel, noteLabel, button);
      row.append(help, fields);
    }
    list.append(row);
  });
}

function renderResources() {
  const query = qs("#resource-search")?.value?.trim().toLocaleLowerCase(state.language === "zh" ? "zh-CN" : "en") || "";
  const list = qs("#resource-list");
  list.replaceChildren();
  const resources = (state.resources.resources || []).filter(resource => {
    if (!query) return true;
    const haystack = [resource.title, resource.title_zh, resource.summary_en, resource.summary_zh, ...(resource.topics || [])]
      .join(" ").toLocaleLowerCase(state.language === "zh" ? "zh-CN" : "en");
    return haystack.includes(query);
  });
  if (!resources.length) {
    list.append(element("p", "empty-state", t("resources.empty")));
    return;
  }
  resources.forEach(resource => {
    const row = element("article", "resource-row");
    const title = element("header");
    const heading = element("div");
    heading.append(
      element("h3", "", resourceTitle(resource)),
      element("span", "resource-meta", t("resources.publisher", { publisher: resource.publisher }))
    );
    let accessControl;
    if (resource.accessible) {
      accessControl = element("a", "source-link", t("common.openSource"));
      accessControl.href = resource.url;
      accessControl.target = "_blank";
      accessControl.rel = "noopener noreferrer";
    } else {
      accessControl = element("span", "status-pill", t("resources.unavailable"));
    }
    title.append(heading, accessControl);
    const summary = element("p", "resource-summary", state.language === "zh" ? resource.summary_zh : resource.summary_en);
    const meta = element("div", "resource-meta");
    meta.append(
      element("span", "", `${difficultyLabel(resource.difficulty)} · ${t("common.minutes", { count: resource.duration_minutes })}`),
      element("span", "", t("resources.checked", { date: resource.checked_at })),
      element("span", "", `${t("resources.access")}: ${resourceLicense(resource)}`)
    );
    if ((resource.prerequisites || []).length) {
      const prerequisites = resource.prerequisites.map(resourceTitleById).join(state.language === "zh" ? "、" : ", ");
      meta.append(element("span", "", t("resources.prerequisites", { items: prerequisites })));
    }
    row.append(title, summary, meta);
    list.append(row);
  });
}

function setTraceBackgroundInert(value) {
  TRACE_BACKGROUND_SELECTORS.forEach(selector => {
    const node = qs(selector);
    if (node) node.inert = value;
  });
}

function traceIsOpen() {
  return qs("#app-shell").classList.contains("drawer-open");
}

function updateTraceMode() {
  const drawer = qs("#trace-drawer");
  if (!traceIsOpen()) {
    setTraceBackgroundInert(false);
    drawer.inert = true;
    drawer.setAttribute("role", "complementary");
    drawer.removeAttribute("aria-modal");
    return;
  }
  drawer.inert = false;
  if (TRACE_MODAL_QUERY.matches) {
    drawer.setAttribute("role", "dialog");
    drawer.setAttribute("aria-modal", "true");
    setTraceBackgroundInert(true);
    if (!drawer.contains(document.activeElement)) qs("#close-trace").focus();
  } else {
    drawer.setAttribute("role", "complementary");
    drawer.removeAttribute("aria-modal");
    setTraceBackgroundInert(false);
  }
}

function renderTraceContent(decision) {
  qs("#trace-heading").textContent = indicatorTitle(decision.indicator_id);
  const content = qs("#trace-content");
  content.replaceChildren();

  const summary = element("section", "trace-section");
  summary.append(element("h3", "", t("trace.summary")));
  const stateLine = element("p");
  stateLine.append(
    element("span", `status-pill ${decision.state}`, `${stateSymbol(decision.state)} ${t(`state.${decision.state}`)}`),
    document.createTextNode(` ${localText(decision.rationale)}`)
  );
  summary.append(stateLine);
  content.append(summary);

  if ((decision.framework_mappings || []).length) {
    const section = element("section", "trace-section");
    section.append(element("h3", "", t("trace.framework")));
    const list = element("ul");
    decision.framework_mappings.forEach(item => list.append(element("li", "", frameworkMappingLabel(item))));
    section.append(list);
    content.append(section);
  }

  if ((decision.missing_evidence || []).length) {
    const section = element("section", "trace-section");
    section.append(element("h3", "", t("trace.missing")));
    const list = element("ul");
    decision.missing_evidence.forEach(item => list.append(element("li", "", localText(item))));
    section.append(list);
    content.append(section);
  }

  if ((decision.conflict_pairs || []).length) {
    const section = element("section", "trace-section");
    section.append(element("h3", "", t("trace.conflict")));
    const list = element("ul");
    decision.conflict_pairs.forEach(pair => {
      const label = pair.map(id => t("trace.family", { id })).join(" ↔ ");
      list.append(element("li", "", label));
    });
    section.append(list);
    content.append(section);
  }

  const excerpts = element("section", "trace-section");
  excerpts.append(element("h3", "", t("trace.excerpts")));
  const selectedFamilyIds = new Set(decision.family_ids || []);
  const families = (state.assessment?.result?.evidence_families || [])
    .filter(family => selectedFamilyIds.has(family.id));
  const materialNodes = Object.fromEntries(
    (state.assessment?.result?.evidence_graph?.nodes || [])
      .filter(node => node.type === "material")
      .map(node => [node.id, node])
  );
  let excerptCount = 0;
  families.forEach(family => {
    (family.members || []).filter(member => member.indicator_id === decision.indicator_id).forEach(member => {
      excerptCount += 1;
      const quote = document.createElement("blockquote");
      quote.className = "trace-quote";
      quote.append(document.createTextNode(`“${member.quote}”`));
      const footerParts = [
        t(`trace.polarity.${member.polarity}`),
        materialNodes[member.material_id]?.label || member.material_id,
        locatorLabel(member.locator),
        t("trace.family", { id: family.id })
      ];
      if (member.security_restricted) footerParts.push(t("trace.securityRestricted"));
      const footer = element("footer", "", footerParts.filter(Boolean).join(" · "));
      quote.append(footer);
      excerpts.append(quote);
    });
  });
  if (!excerptCount) excerpts.append(element("p", "empty-state", t("trace.noExcerpt")));
  content.append(excerpts);
}

function openTrace(decision, trigger) {
  state.lastTraceTrigger = trigger;
  state.activeTraceIndicator = decision.indicator_id;
  renderTraceContent(decision);

  qs("#app-shell").classList.add("drawer-open");
  const drawer = qs("#trace-drawer");
  drawer.inert = false;
  drawer.setAttribute("aria-hidden", "false");
  updateTraceMode();
  qs("#close-trace").focus();
}

function closeTrace() {
  const returnTarget = state.lastTraceTrigger?.isConnected ? state.lastTraceTrigger : qs("#main-content");
  setTraceBackgroundInert(false);
  qs("#app-shell").classList.remove("drawer-open");
  returnTarget?.focus();
  const drawer = qs("#trace-drawer");
  drawer.inert = true;
  drawer.setAttribute("aria-hidden", "true");
  drawer.setAttribute("role", "complementary");
  drawer.removeAttribute("aria-modal");
  state.activeTraceIndicator = null;
  state.lastTraceTrigger = null;
}

function refreshOpenTrace() {
  if (!traceIsOpen() || !state.activeTraceIndicator) return;
  const decision = state.assessment?.result?.indicators?.find(
    item => item.indicator_id === state.activeTraceIndicator
  );
  if (!decision) {
    closeTrace();
    return;
  }
  renderTraceContent(decision);
}

function renderAll() {
  applyTranslations();
  populateProfile();
  renderSystem();
  renderMaterials();
  renderAssessment();
  renderPaths();
  renderReviews();
  renderResources();
}

function renderForLanguageChange() {
  applyTranslations();
  renderSystem();
  renderMaterials();
  renderAssessment();
  renderPaths();
  renderReviews();
  renderResources();
  refreshOpenTrace();
}

async function loadWorkspace() {
  setStatus(t("status.loading"));
  const payload = await api("/api/bootstrap");
  state.profile = payload.profile;
  state.materials = payload.materials || [];
  state.assessment = payload.assessment;
  state.reviews = payload.reviews || [];
  state.framework = payload.framework || state.framework;
  state.resources = payload.resources || state.resources;
  state.system = payload.system || {};
  renderAll();
  setStatus("");
}

async function saveProfile(event) {
  event.preventDefault();
  const form = event.currentTarget;
  if (!reportLocalizedValidity(form)) return;
  const button = qs("button[type=submit]", form);
  button.disabled = true;
  qs("#context-status").textContent = "";
  const data = new FormData(form);
  const payload = {
    display_name: data.get("display_name"),
    discipline: data.get("discipline"),
    career_stage: data.get("career_stage"),
    role: data.get("role"),
    goals: String(data.get("goals") || "").split("\n").map(item => item.trim()).filter(Boolean),
    weekly_minutes: Number(data.get("weekly_minutes")),
    constraints: data.get("constraints"),
    developmental_use_acknowledged: data.get("developmental_use_acknowledged") === "on"
  };
  try {
    const response = await api("/api/profile", { method: "POST", body: JSON.stringify(payload) });
    state.profile = response.profile;
    if (state.assessment?.status === "completed") state.assessment.status = "stale";
    qs("#context-status").textContent = t("context.saved");
    setStatus(t("status.saved"));
    renderAssessment();
    renderPaths();
  } catch (error) {
    qs("#context-status").textContent = errorText(error);
    setStatus(errorText(error), true);
  } finally {
    button.disabled = false;
  }
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
}

async function addMaterial(event) {
  event.preventDefault();
  const form = event.currentTarget;
  if (!reportLocalizedValidity(form)) return;
  const file = form.elements.file.files[0];
  if (!file) return;
  if (state.system.max_file_bytes && file.size > Number(state.system.max_file_bytes)) {
    const message = t("error.FILE_TOO_LARGE");
    qs("#material-status").textContent = message;
    setStatus(message, true);
    return;
  }
  const button = qs("button[type=submit]", form);
  const status = qs("#material-status");
  button.disabled = true;
  status.textContent = t("materials.adding");
  try {
    const buffer = await file.arrayBuffer();
    const payload = {
      filename: file.name,
      title: form.elements.title.value,
      material_type: form.elements.material_type.value,
      source_date: form.elements.source_date.value,
      contribution_level: form.elements.contribution_level.value,
      contribution_note: form.elements.contribution_note.value,
      source_ref: form.elements.source_ref.value,
      privacy_confirmed: form.elements.privacy_confirmed.checked,
      content_base64: arrayBufferToBase64(buffer)
    };
    const response = await api("/api/materials", { method: "POST", body: JSON.stringify(payload) });
    state.materials.unshift(response.material);
    if (state.assessment?.status === "completed") state.assessment.status = "stale";
    form.reset();
    qs("#selected-file-name").textContent = "";
    status.textContent = t("materials.added");
    renderMaterials();
    renderAssessment();
    renderPaths();
    setStatus(t("status.saved"));
  } catch (error) {
    status.textContent = errorText(error);
    setStatus(errorText(error), true);
  } finally {
    button.disabled = false;
  }
}

async function removeMaterial(materialId) {
  if (!globalThis.confirm(t("materials.removeConfirm"))) return;
  try {
    await api(`/api/materials/${materialId}`, { method: "DELETE" });
    state.materials = state.materials.filter(item => item.id !== materialId);
    state.assessment = null;
    state.reviews = [];
    state.reviewDrafts = {};
    if (traceIsOpen()) closeTrace();
    renderMaterials();
    renderAssessment();
    renderPaths();
    renderReviews();
    setStatus(t("status.deleted"));
  } catch (error) {
    setStatus(errorText(error), true);
  }
}

async function runAssessment() {
  const button = qs("#run-assessment-button");
  const status = qs("#assessment-status");
  button.disabled = true;
  status.className = "assessment-status is-running";
  status.textContent = t("evidence.running");
  try {
    const response = await api("/api/assessments", {
      method: "POST",
      body: JSON.stringify({ language: state.language })
    });
    state.assessment = response.assessment;
    const bootstrap = await api("/api/bootstrap");
    state.reviews = bootstrap.reviews || [];
    status.className = "assessment-status";
    status.textContent = t("evidence.ready");
    renderAssessment();
    renderPaths();
    renderReviews();
    setStatus(t("evidence.ready"));
  } catch (error) {
    status.className = "assessment-status is-error";
    status.textContent = errorText(error);
    setStatus(errorText(error), true);
  } finally {
    button.disabled = false;
  }
}

async function resolveReview(reviewId, outcome, note, button) {
  if (!outcome) {
    setStatus(t("error.REVIEW_OUTCOME_INVALID"), true);
    return;
  }
  button.disabled = true;
  try {
    const response = await api(`/api/reviews/${reviewId}/resolve`, {
      method: "POST",
      body: JSON.stringify({ outcome, note })
    });
    state.reviews = response.reviews || [];
    delete state.reviewDrafts[reviewId];
    if (state.assessment) state.assessment.status = "stale";
    renderReviews();
    renderAssessment();
    renderPaths();
    setStatus(t("status.saved"));
  } catch (error) {
    setStatus(errorText(error), true);
    button.disabled = false;
  }
}

async function deleteWorkspace() {
  const confirmation = qs("#delete-confirmation").value;
  try {
    await api("/api/workspace", { method: "DELETE", body: JSON.stringify({ confirmation }) });
    localStorage.removeItem(WORKSPACE_KEY);
    ensureWorkspace();
    state.profile = null;
    state.materials = [];
    state.assessment = null;
    state.reviews = [];
    state.reviewDrafts = {};
    qs("#data-dialog").close();
    qs("#context-form").reset();
    await loadWorkspace();
    setStatus(t("status.workspaceDeleted"));
    showView("context");
  } catch (error) {
    setStatus(errorText(error), true);
  }
}

function bindEvents() {
  qsa("#context-form input, #context-form select, #context-form textarea, #material-form input, #material-form select, #material-form textarea")
    .forEach(control => {
      const clearValidation = () => control.setCustomValidity("");
      control.addEventListener("input", clearValidation);
      control.addEventListener("change", clearValidation);
    });
  qsa("[data-view]").forEach(button => button.addEventListener("click", () => showView(button.dataset.view)));
  qsa("[data-language]").forEach(button => button.addEventListener("click", () => {
    const next = button.dataset.language;
    if (!VALID_LANGUAGES.has(next)) return;
    state.language = next;
    localStorage.setItem(LANGUAGE_KEY, next);
    renderForLanguageChange();
  }));
  qs("#context-form").addEventListener("submit", saveProfile);
  qs("#material-form").addEventListener("submit", addMaterial);
  qs("#material-form input[type=file]").addEventListener("change", event => {
    const file = event.currentTarget.files[0];
    qs("#selected-file-name").textContent = file?.name || "";
    if (file && !qs("#material-form [name=title]").value) qs("#material-form [name=title]").value = file.name.replace(/\.[^.]+$/, "");
  });
  qs("#run-assessment-button").addEventListener("click", runAssessment);
  qs("#close-trace").addEventListener("click", closeTrace);
  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && qs("#app-shell").classList.contains("drawer-open")) closeTrace();
    if (event.key === "Tab" && traceIsOpen() && TRACE_MODAL_QUERY.matches) {
      const focusable = qsa('button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])', qs("#trace-drawer"))
        .filter(node => node.offsetParent !== null);
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (!qs("#trace-drawer").contains(document.activeElement)) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
      } else if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
  });
  if (TRACE_MODAL_QUERY.addEventListener) TRACE_MODAL_QUERY.addEventListener("change", updateTraceMode);
  qs("#resource-search").addEventListener("input", renderResources);
  qs("#data-controls-button").addEventListener("click", () => qs("#data-dialog").showModal());
  qs("#clear-language").addEventListener("click", () => {
    localStorage.removeItem(LANGUAGE_KEY);
    state.language = "en";
    renderForLanguageChange();
    setStatus(t("status.languageReset"));
  });
  qs("#delete-workspace").addEventListener("click", deleteWorkspace);
}

async function initialize() {
  ensureWorkspace();
  const savedLanguage = localStorage.getItem(LANGUAGE_KEY);
  state.language = VALID_LANGUAGES.has(savedLanguage) ? savedLanguage : "en";
  try {
    const response = await fetch("/static/i18n.json", { cache: "no-store" });
    state.dictionary = await response.json();
    applyTranslations();
    bindEvents();
    await loadWorkspace();
  } catch (error) {
    setStatus(errorText(error), true);
  }
}

document.addEventListener("DOMContentLoaded", initialize);

