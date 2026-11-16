/**
 * Content script: floating WSB Thread Radar panel on reddit.com.
 *
 * Two modes share one panel:
 *   - semantic search through the lab-2 backend;
 *   - natural-language assistant through the LLM gateway.
 *
 * All network requests go through the background service worker, which has
 * host permissions for the local HTTP services.
 */
(() => {
  "use strict";

  if (window.top !== window.self) return;
  const HOST_ID = "wsb-thread-radar-host";
  if (document.getElementById(HOST_ID)) return;

  const PORT_NAME = "wsb-thread-radar";
  const DEFAULT_TOP_K = 10;

  const TEMPLATE = `
    <div class="panel">
      <div class="head">
        <div class="logo" aria-hidden="true">TR</div>
        <div class="title">WSB Thread Radar</div>
        <span class="status" data-state="idle" title="Статус сервисов"></span>
        <button class="icon-btn watch-current" type="button" hidden
                title="Добавить текущий тред в мониторинг">＋</button>
        <button class="icon-btn collapse" type="button" title="Свернуть">▾</button>
      </div>
      <div class="body">
        <div class="tabs" role="tablist">
          <button class="tab active" type="button" data-tab="search" role="tab">Поиск</button>
          <button class="tab" type="button" data-tab="assistant" role="tab">Ассистент</button>
        </div>

        <section class="pane search-pane active">
          <form class="search-form" autocomplete="off">
            <input class="query" type="search" name="q"
                   placeholder="акции APPL, GameStop squeeze…"
                   autocomplete="off" spellcheck="false"
                   aria-label="Семантический поиск по r/WallStreetBets">
            <select class="topk" aria-label="Количество результатов">
              <option value="5">5</option>
              <option value="10" selected>10</option>
              <option value="20">20</option>
            </select>
            <button class="primary" type="submit">Найти</button>
          </form>
          <div class="meta" hidden></div>
          <div class="search-results" role="region" aria-live="polite"></div>
        </section>

        <section class="pane assistant-pane">
          <div class="assistant-log" role="log" aria-live="polite"></div>
          <div class="assistant-context" hidden></div>
          <form class="assistant-form">
            <textarea class="assistant-input" rows="2"
                      placeholder="Например: найди посты про GME 2021 и сохрани лучший в мониторинг"></textarea>
            <button class="primary" type="submit">Отправить</button>
          </form>
          <div class="assistant-foot">
            <span class="assistant-status">готов</span>
            <button class="link-btn reset-chat" type="button">новый диалог</button>
          </div>
        </section>

        <div class="foot">
          <span class="endpoint" title="Адрес backend">127.0.0.1:8000</span>
          <button class="link-btn settings" type="button">настройки</button>
        </div>
      </div>
    </div>
  `;

  const FALLBACK_CSS = `
    :host { all: initial; position: fixed; right: 20px; bottom: 20px;
      z-index: 2147483000; display:block; font: 13px/1.4 sans-serif; }
    .panel { width:min(480px,calc(100vw - 32px)); color:#e8eaed; background:#16181d;
      border:1px solid #2f333c; border-radius:14px; overflow:hidden; }
    .head{display:flex;align-items:center;gap:8px;padding:9px 12px;background:#1c1f26}
    .logo{background:#ff4500;color:#fff;padding:4px 6px;border-radius:8px;font-weight:800;font-size:9px}
    .title{flex:1;font-weight:650}.body{display:block}.pane{display:none}.pane.active{display:block}
    .tabs{display:flex;gap:4px;padding:8px 12px 0}.tab{padding:6px 10px;border:0;border-radius:7px;background:#222833;color:#aab2bf}
    .tab.active{background:#ff5a12;color:#fff}.search-form,.assistant-form{display:flex;gap:6px;padding:10px 12px}
    input,textarea,select{color:#e8eaed;background:#0f1115;border:1px solid #343945;border-radius:8px;padding:8px;font:inherit}
    input,textarea{flex:1;min-width:0}.primary{padding:0 14px;color:#fff;background:#ff5a12;border:0;border-radius:8px;font-weight:650}
    .result{display:block;padding:10px;color:inherit;text-decoration:none;border-radius:10px}.result:hover{background:#1f232b}
    .result-title{font-weight:600}.result-meta,.snippet{margin-left:27px;color:#9aa3af;font-size:11px}
    .chip{padding:1px 6px;background:#242a33;border:1px solid #333b47;border-radius:999px}
    .assistant-log{max-height:46vh;overflow:auto;padding:8px 12px}.msg{padding:8px 10px;margin:6px 0;border-radius:9px;white-space:pre-wrap}
    .msg.user{background:#223044;margin-left:28px}.msg.assistant{background:#1d2028;margin-right:18px}
    .msg.error{background:#3a1d1d}.foot{display:flex;gap:6px;padding:7px 12px;background:#14161b;color:#7d8694;font-size:11px}
    .endpoint{flex:1;overflow:hidden;text-overflow:ellipsis}.link-btn{border:0;background:none;color:#ff9a62;cursor:pointer}
  `;

  async function loadStyles(shadow) {
    try {
      const css = await (await fetch(chrome.runtime.getURL("panel.css"))).text();
      if (!css.trim()) throw new Error("empty");
      const style = document.createElement("style");
      style.textContent = css;
      shadow.appendChild(style);
    } catch (_err) {
      const style = document.createElement("style");
      style.textContent = FALLBACK_CSS;
      shadow.appendChild(style);
    }
  }

  let port = null;
  let messageSeq = 0;
  const pending = new Map();
  let reconnectTimer = null;

  function openPort() {
    if (port) return port;
    try {
      port = chrome.runtime.connect({ name: PORT_NAME });
    } catch (_err) {
      port = null;
      return null;
    }
    port.onMessage.addListener((reply) => {
      const waiter = pending.get(reply && reply.seq);
      if (!waiter) return;
      pending.delete(reply.seq);
      waiter.resolve(reply);
    });
    port.onDisconnect.addListener(() => {
      port = null;
      for (const [id, waiter] of pending) {
        pending.delete(id);
        waiter.reject(new Error("background worker disconnected"));
      }
      if (!reconnectTimer) {
        reconnectTimer = setTimeout(() => {
          reconnectTimer = null;
          openPort();
        }, 400);
      }
    });
    return port;
  }

  function send(message, timeoutMs = 130000) {
    return new Promise((resolve) => {
      const worker = openPort();
      if (!worker) {
        chrome.runtime.sendMessage(message, (response) => {
          void chrome.runtime.lastError;
          resolve(response || { ok: false, error: "Фоновый процесс недоступен." });
        });
        return;
      }
      const id = ++messageSeq;
      const timer = setTimeout(() => {
        if (!pending.has(id)) return;
        pending.delete(id);
        resolve({ ok: false, error: "Фоновый процесс не ответил вовремя." });
      }, timeoutMs);
      pending.set(id, {
        resolve: (reply) => {
          clearTimeout(timer);
          resolve(reply || { ok: false, error: "Пустой ответ." });
        },
        reject: (err) => {
          clearTimeout(timer);
          resolve({ ok: false, error: `Связь потеряна: ${err && err.message ? err.message : err}` });
        },
      });
      try {
        worker.postMessage(Object.assign({ seq: id }, message));
      } catch (err) {
        pending.delete(id);
        clearTimeout(timer);
        resolve({ ok: false, error: `Не удалось отправить запрос: ${err}` });
      }
    });
  }

  function storageGet(defaults) {
    return new Promise((resolve) => {
      try {
        chrome.storage.local.get(defaults, (values) => {
          void chrome.runtime.lastError;
          resolve(values || defaults);
        });
      } catch (_err) {
        resolve(defaults);
      }
    });
  }

  function storageSet(values) {
    try {
      chrome.storage.local.set(values, () => void chrome.runtime.lastError);
    } catch (_err) {
      /* ignore */
    }
  }

  function formatNumber(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "";
    try {
      return new Intl.NumberFormat().format(number);
    } catch (_err) {
      return String(number);
    }
  }

  function formatElapsed(ms) {
    if (!Number.isFinite(ms)) return "";
    return ms >= 1000 ? `${(ms / 1000).toFixed(2)} с` : `${Math.round(ms)} мс`;
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function chip(text, extraClass) {
    return el("span", extraClass ? `chip ${extraClass}` : "chip", text);
  }

  function isPostId(value) {
    const text = String(value || "").trim();
    return /^[a-z0-9]{4,12}$/i.test(text);
  }

  function safeUrl(item) {
    if (!item || typeof item !== "object") return null;
    if (item.url) {
      try {
        const url = new URL(String(item.url), "https://www.reddit.com");
        if (url.protocol === "http:" || url.protocol === "https:") return url.href;
      } catch (_err) {
        /* ignore */
      }
    }
    // Only real Reddit post ids may be turned into a thread URL.  Entity ids
    // such as watch_… / task_… are internal and must never be used here.
    const candidates = [item.post_id];
    if (item.kind === "thread") candidates.push(item.target);
    const postId = candidates.find(isPostId);
    if (!postId) return null;
    return (
      "https://www.reddit.com/r/wallstreetbets/comments/" +
      `${encodeURIComponent(postId)}/`
    );
  }

  function currentThreadContext() {
    const match = location.pathname.match(/\/r\/[^/]+\/comments\/([a-z0-9]+)/i);
    if (!match) return null;
    const h1 = document.querySelector("h1");
    return {
      post_id: match[1],
      title: (h1 && h1.textContent ? h1.textContent : document.title || "").trim(),
      url: location.href,
    };
  }

  async function init() {
    const host = document.createElement("div");
    host.id = HOST_ID;
    const shadow = host.attachShadow({ mode: "open" });
    await loadStyles(shadow);
    const template = document.createElement("template");
    template.innerHTML = TEMPLATE;
    shadow.append(template.content);
    document.documentElement.appendChild(host);

    const panel = shadow.querySelector(".panel");
    const head = shadow.querySelector(".head");
    const collapseButton = shadow.querySelector(".collapse");
    const watchCurrent = shadow.querySelector(".watch-current");
    const status = shadow.querySelector(".status");
    const endpoint = shadow.querySelector(".endpoint");
    const settingsButton = shadow.querySelector(".settings");
    const tabs = Array.from(shadow.querySelectorAll(".tab"));
    const panes = {
      search: shadow.querySelector(".search-pane"),
      assistant: shadow.querySelector(".assistant-pane"),
    };
    const searchForm = shadow.querySelector(".search-form");
    const queryInput = shadow.querySelector(".query");
    const topk = shadow.querySelector(".topk");
    const meta = shadow.querySelector(".meta");
    const searchResults = shadow.querySelector(".search-results");
    const assistantLog = shadow.querySelector(".assistant-log");
    const assistantContext = shadow.querySelector(".assistant-context");
    const assistantForm = shadow.querySelector(".assistant-form");
    const assistantInput = shadow.querySelector(".assistant-input");
    const assistantStatus = shadow.querySelector(".assistant-status");
    const resetChat = shadow.querySelector(".reset-chat");

    let requestSeq = 0;
    let lastQuery = "";
    let conversationId = "";
    let busy = false;
    const thread = currentThreadContext();

    const stored = await storageGet({ conversationId: "", topK: DEFAULT_TOP_K, collapsed: false });
    conversationId = String(stored.conversationId || "");
    if (Array.from(topk.options).some((option) => option.value === String(stored.topK))) {
      topk.value = String(stored.topK);
    }
    if (stored.collapsed) setCollapsed(true);

    if (thread) {
      watchCurrent.hidden = false;
      assistantContext.hidden = false;
      assistantContext.replaceChildren(
        el("b", "", "Контекст: "),
        document.createTextNode(`тред ${thread.post_id}${thread.title ? ` — ${thread.title}` : ""}`)
      );
    }

    function setStatus(state, title) {
      status.dataset.state = state;
      status.title = title || "";
    }

    function setEndpoint(base) {
      let label = base || "127.0.0.1:8000";
      try {
        label = new URL(label).host;
      } catch (_err) {
        /* keep */
      }
      endpoint.textContent = `${label} · LLM 8010`;
    }

    function setTab(name) {
      tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === name));
      Object.entries(panes).forEach(([key, pane]) => pane.classList.toggle("active", key === name));
      if (name === "assistant") assistantInput.focus();
      else queryInput.focus();
    }

    function setCollapsed(collapsed) {
      panel.classList.toggle("collapsed", collapsed);
      collapseButton.textContent = collapsed ? "▴" : "▾";
      storageSet({ collapsed });
    }

    function setAssistantStatus(text, state) {
      assistantStatus.textContent = text;
      assistantStatus.dataset.state = state || "";
    }

    function scrollAssistant() {
      assistantLog.scrollTop = assistantLog.scrollHeight;
    }

    function appendMessage(role, text) {
      const node = el("div", `msg ${role}`, text);
      assistantLog.append(node);
      scrollAssistant();
      return node;
    }

    function showAssistantError(message) {
      appendMessage("assistant error", `Ошибка: ${message}`);
      setAssistantStatus("ошибка", "error");
    }

    function renderSearchMeta(payload, elapsed) {
      meta.hidden = true;
      meta.replaceChildren();
      const normalized = String(payload.normalized_query || "");
      const original = String(payload.query || "");
      if (normalized && normalized.toLowerCase() !== original.toLowerCase()) {
        const line = el("div");
        line.append("нормализовано: ", el("b", "", normalized));
        meta.append(line);
      }
      meta.append(el("div", "", `${(payload.results || []).length} результатов · ${formatElapsed(elapsed)}`));
      meta.hidden = false;
    }

    function renderSearchResults(payload, elapsed) {
      searchResults.replaceChildren();
      renderSearchMeta(payload, elapsed);
      const items = Array.isArray(payload.results) ? payload.results : [];
      if (!items.length) {
        searchResults.append(el("div", "state", "Ничего не найдено. Попробуйте переформулировать запрос."));
        return;
      }
      items.forEach((item, index) => {
        const card = el("div", "result");
        const top = el("div", "result-top");
        top.append(el("span", "rank", `#${item.rank || index + 1}`),
                   el("span", "result-title", item.title || "(без заголовка)"));
        card.append(top);

        const metaRow = el("div", "result-meta");
        if (item.year) metaRow.append(chip(String(item.year)));
        if (item.label) metaRow.append(chip(String(item.label)));
        if (Number.isFinite(Number(item.score))) metaRow.append(chip(`sim ${Number(item.score).toFixed(3)}`, "score"));
        (item.tickers || []).slice(0, 4).forEach((ticker) => metaRow.append(chip(`$${ticker}`, "ticker")));
        if (item.post_score !== undefined) metaRow.append(chip(`▲ ${formatNumber(item.post_score)}`));
        if (item.num_comments !== undefined) metaRow.append(chip(`💬 ${formatNumber(item.num_comments)}`));
        card.append(metaRow);

        if (item.snippet) card.append(el("div", "snippet", item.snippet));

        const actions = el("div", "result-actions");
        const open = el("a", "mini-btn", "Открыть тред");
        const openUrl = safeUrl(item);
        if (openUrl) {
          open.href = openUrl;
          open.target = "_blank";
          open.rel = "noopener noreferrer";
          actions.append(open);
        }
        const save = el("button", "mini-btn", "＋ В мониторинг");
        save.type = "button";
        save.addEventListener("click", () => {
          setTab("assistant");
          runAssistant("Добавь этот тред в мониторинг", {
            post_id: item.post_id,
            title: item.title,
            url: item.url,
          });
        });
        actions.append(save);
        card.append(actions);
        searchResults.append(card);
      });
    }

    async function runSearch(rawQuery) {
      const query = String(rawQuery || queryInput.value || "").trim();
      if (!query) {
        queryInput.focus();
        return;
      }
      lastQuery = query;
      const seq = ++requestSeq;
      const started = performance.now();
      meta.hidden = true;
      searchResults.replaceChildren(el("div", "loading", "Ищем по смыслу…"));
      setStatus("loading", "Поиск…");
      const response = await send({ type: "wsb:search", query, topK: Number(topk.value) || DEFAULT_TOP_K });
      if (seq !== requestSeq) return;
      if (!response || !response.ok) {
        setStatus("error", "Backend недоступен");
        searchResults.replaceChildren(el("div", "state error", (response && response.error) || "Ошибка запроса."));
        return;
      }
      setStatus("ok", "Backend доступен");
      setEndpoint(response.base);
      renderSearchResults(response.data || {}, performance.now() - started);
    }

    function renderPending(pending) {
      const preview = pending.preview || {};
      const box = el("div", "pending");
      box.append(el("div", "pending-title", "Требуется подтверждение"));
      box.append(el("div", "pending-summary", preview.summary || "Подготовлена операция."));
      const changes = el("pre", "pending-changes", JSON.stringify(preview.changes || {}, null, 2));
      box.append(changes);
      const actions = el("div", "pending-actions");
      const confirm = el("button", "primary small", "Подтвердить");
      const reject = el("button", "ghost small", "Отмена");
      confirm.type = "button";
      reject.type = "button";
      confirm.addEventListener("click", async () => {
        confirm.disabled = true;
        reject.disabled = true;
        await runConfirm(pending.action_id, "confirm");
      });
      reject.addEventListener("click", async () => {
        confirm.disabled = true;
        reject.disabled = true;
        await runConfirm(pending.action_id, "reject");
      });
      actions.append(confirm, reject);
      box.append(actions);
      return box;
    }

    function resultCards(items) {
      const box = el("div", "actual-results");
      items.forEach((item) => {
        const card = el("div", "actual-card");
        const title = item.title || item.name || item.key || item.ticker || "(без названия)";
        const url = safeUrl(item);
        if (url) {
          const link = el("a", "actual-title", title);
          link.href = url;
          link.target = "_blank";
          link.rel = "noopener noreferrer";
          card.append(link);
        } else {
          card.append(el("div", "actual-title", title));
        }
        const metaRow = el("div", "actual-meta");
        if (item.id) metaRow.append(chip(item.id));
        if (item.kind) metaRow.append(chip(item.kind));
        if (item.target) metaRow.append(chip(item.target));
        if (item.post_id) metaRow.append(chip(item.post_id));
        if (item.year) metaRow.append(chip(String(item.year)));
        if (item.label) metaRow.append(chip(String(item.label)));
        if (item.status) metaRow.append(chip(String(item.status), "status-chip"));
        if (item.count !== undefined) metaRow.append(chip(`count ${item.count}`));
        (item.tickers || []).forEach((ticker) => metaRow.append(chip(`$${ticker}`, "ticker")));
        card.append(metaRow);
        if (item.snippet) card.append(el("div", "actual-snippet", item.snippet));
        box.append(card);
      });
      return box;
    }

    function renderActual(result) {
      if (!result || typeof result !== "object") return null;
      const data = result.data || result;
      const box = el("div", "actual");
      const head = el("div", "actual-head");
      head.append(el("span", "actual-tool", `фактический результат: ${result.tool || result.status || "tool"}`));
      box.append(head);

      const items = data.results || data.matches || data.new_matches;
      if (Array.isArray(items) && items.length) {
        box.append(resultCards(items));
        return box;
      }
      if (Array.isArray(data.rows) && data.rows.length) {
        const table = el("table", "data-table");
        const keys = Object.keys(data.rows[0]);
        const header = el("tr");
        keys.forEach((key) => header.append(el("th", "", key)));
        table.append(header);
        data.rows.forEach((row) => {
          const tr = el("tr");
          keys.forEach((key) => tr.append(el("td", "", String(row[key] ?? ""))));
          table.append(tr);
        });
        box.append(table);
        if (data.total !== undefined) box.append(el("div", "actual-note", `всего: ${data.total}`));
        return box;
      }
      if (data.record && typeof data.record === "object") {
        box.append(resultCards([data.record]));
        return box;
      }
      if (data.target === "service" || data.status === "ok" || data.total !== undefined) {
        box.append(el("pre", "actual-json", JSON.stringify(data, null, 2).slice(0, 2500)));
        return box;
      }
      if (result.status === "pending_confirmation") return null;
      box.append(el("pre", "actual-json", JSON.stringify(data, null, 2).slice(0, 2500)));
      return box;
    }

    function renderTrace(trace) {
      if (!Array.isArray(trace) || !trace.length) return null;
      const details = el("details", "trace");
      details.append(el("summary", "", `trace (${trace.length})`));
      const pre = el("pre", "", JSON.stringify(trace.map((item) => ({
        tool: item.tool,
        arguments: item.arguments,
        status: item.status,
      })), null, 2));
      details.append(pre);
      return details;
    }

    function renderAssistantResponse(payload) {
      if (!payload) {
        showAssistantError("пустой ответ LLM-сервиса");
        return;
      }
      if (payload.conversation_id) {
        conversationId = payload.conversation_id;
        storageSet({ conversationId });
      }
      if (payload.notice) appendMessage("assistant notice", payload.notice);
      const reply = payload.reply || (payload.ok ? "Готово." : "Ошибка.");
      const bubble = appendMessage("assistant", reply);
      if (!payload.ok) bubble.classList.add("error");
      if (payload.pending_action) bubble.append(renderPending(payload.pending_action));
      const actual = renderActual(payload.actual_result);
      if (actual) bubble.append(actual);
      const trace = renderTrace(payload.trace);
      if (trace) bubble.append(trace);
      if (payload.error) {
        bubble.append(el("pre", "error-json", JSON.stringify(payload.error, null, 2)));
      }
      setAssistantStatus(payload.ok ? "готов" : "ошибка", payload.ok ? "" : "error");
      scrollAssistant();
    }

    async function runAssistant(message, context) {
      const text = String(message || "").trim();
      if (!text || busy) return;
      busy = true;
      setAssistantStatus("думаю…", "loading");
      appendMessage("user", text);
      const loader = appendMessage("assistant", "…");
      const payload = await send({
        type: "wsb:assistant-chat",
        message: text,
        conversationId,
        context: context || (currentThreadContext() || {}),
      });
      loader.remove();
      busy = false;
      if (!payload || !payload.ok) {
        showAssistantError((payload && payload.error) || "LLM-сервис недоступен.");
        return;
      }
      renderAssistantResponse(payload.data || payload);
    }

    async function runConfirm(actionId, decision) {
      setAssistantStatus("подтверждаю…", "loading");
      const payload = await send({
        type: "wsb:assistant-confirm",
        actionId,
        decision,
        conversationId,
      });
      if (!payload || !payload.ok) {
        showAssistantError((payload && payload.error) || "Не удалось подтвердить операцию.");
        return;
      }
      renderAssistantResponse(payload.data || payload);
    }

    async function refreshHealth() {
      const [backend, assistant] = await Promise.all([
        send({ type: "wsb:health" }, 7000),
        send({ type: "wsb:assistant-health" }, 7000),
      ]);
      const ok = backend && backend.ok && assistant && assistant.ok;
      setStatus(ok ? "ok" : "error",
        `backend: ${backend && backend.ok ? "ok" : "нет"}; assistant: ${assistant && assistant.ok ? "ok" : "нет"}`);
      if (backend && backend.ok) setEndpoint(backend.base);
    }

    function resetConversation() {
      conversationId = "";
      storageSet({ conversationId: "" });
      assistantLog.replaceChildren();
      setAssistantStatus("новый диалог", "");
    }

    tabs.forEach((tab) => {
      tab.addEventListener("click", () => setTab(tab.dataset.tab));
    });
    searchForm.addEventListener("submit", (event) => {
      event.preventDefault();
      event.stopPropagation();
      runSearch(queryInput.value);
    });
    topk.addEventListener("change", () => storageSet({ topK: Number(topk.value) || DEFAULT_TOP_K }));
    assistantForm.addEventListener("submit", (event) => {
      event.preventDefault();
      event.stopPropagation();
      const text = assistantInput.value;
      assistantInput.value = "";
      runAssistant(text);
    });
    assistantInput.addEventListener("keydown", (event) => {
      event.stopPropagation();
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        assistantForm.requestSubmit();
      }
    });
    assistantInput.addEventListener("keyup", (event) => event.stopPropagation());
    queryInput.addEventListener("keydown", (event) => {
      event.stopPropagation();
      if (event.key === "Escape") setCollapsed(true);
    });
    queryInput.addEventListener("keyup", (event) => event.stopPropagation());
    resetChat.addEventListener("click", resetConversation);
    settingsButton.addEventListener("click", () => send({ type: "wsb:open-options" }));
    collapseButton.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      setCollapsed(!panel.classList.contains("collapsed"));
    });
    head.addEventListener("click", (event) => {
      if (event.target.closest("button")) return;
      if (panel.classList.contains("collapsed")) {
        setCollapsed(false);
        queryInput.focus();
      }
    });
    watchCurrent.addEventListener("click", () => {
      const ctx = currentThreadContext();
      if (!ctx) return;
      setTab("assistant");
      runAssistant("Добавь текущий тред в мониторинг", ctx);
    });
    document.addEventListener("keydown", (event) => {
      if (event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey) {
        if (event.code === "KeyS") {
          event.preventDefault();
          setCollapsed(false);
          setTab("search");
          queryInput.select();
        }
        if (event.code === "KeyA") {
          event.preventDefault();
          setCollapsed(false);
          setTab("assistant");
          assistantInput.focus();
        }
      }
    }, true);
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible") refreshHealth();
    });

    setEndpoint("127.0.0.1:8000");
    refreshHealth();
  }

  init().catch((err) => console.error("[WSB Thread Radar] init failed", err));
})();
