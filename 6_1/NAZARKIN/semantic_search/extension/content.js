/**
 * Content script: injects a floating semantic-search panel into reddit.com.
 *
 * The panel lives in a shadow root so Reddit's CSS (and our CSS) cannot leak
 * between the page and the widget. Network calls go through the background
 * service worker, which has host permissions for the local API.
 */
(() => {
  "use strict";

  if (window.top !== window.self) return;

  const HOST_ID = "wsb-semantic-search-host";
  if (document.getElementById(HOST_ID)) return;

  const DEFAULT_TOP_K = 10;
  const DEFAULT_SERVER = "http://127.0.0.1:8000";
  const PORT_NAME = "wsb-search";

  const TEMPLATE = `
    <div class="panel">
      <div class="head">
        <div class="logo" aria-hidden="true">WSB</div>
        <div class="title">Semantic search</div>
        <span class="status" data-state="idle" title="Статус локального API"></span>
        <button class="icon-btn collapse" type="button" aria-label="Свернуть панель"
                title="Свернуть / развернуть">▾</button>
      </div>
      <div class="body">
        <form class="form" autocomplete="off">
          <input class="query" type="search" name="q"
                 placeholder="акции APPL, GameStop squeeze…"
                 autocomplete="off" spellcheck="false"
                 aria-label="Семантический поиск по r/WallStreetBets">
          <select class="topk" aria-label="Количество результатов" title="Количество результатов">
            <option value="5">5</option>
            <option value="10" selected>10</option>
            <option value="20">20</option>
            <option value="50">50</option>
          </select>
          <button class="primary" type="submit">Найти</button>
        </form>
        <div class="meta" hidden></div>
        <div class="results" role="region" aria-live="polite" aria-label="Результаты поиска"></div>
        <div class="foot">
          <span>local API</span>
          <span class="endpoint" title="Адрес локального сервера">127.0.0.1:8000</span>
          <button class="link-btn settings" type="button"
                  title="Настройки расширения">настройки</button>
        </div>
      </div>
    </div>
  `;

  const FALLBACK_CSS = `
    :host { all: initial; position: fixed; right: 20px; bottom: 20px;
            z-index: 2147483000; display: block; font-family: sans-serif; font-size: 13px; }
    .panel { width: min(440px, calc(100vw - 32px)); color: #e8eaed; background: #16181d;
             border: 1px solid #2f333c; border-radius: 14px; overflow: hidden; }
    .head { display: flex; align-items: center; gap: 8px; padding: 9px 12px;
            background: #1c1f26; border-bottom: 1px solid #2f333c; }
    .logo { padding: 4px 6px; border-radius: 8px; background: #ff4500;
            color: #fff; font-size: 9px; font-weight: 800; }
    .title { flex: 1; font-weight: 650; }
    .status { width: 8px; height: 8px; border-radius: 50%; background: #6b7280; }
    .status[data-state="ok"] { background: #22c55e; }
    .status[data-state="loading"] { background: #f59e0b; }
    .status[data-state="error"] { background: #ef4444; }
    .form { display: flex; gap: 6px; padding: 10px 12px 8px; }
    .query { flex: 1; min-width: 0; padding: 9px 11px; color: #e8eaed; background: #0f1115;
             border: 1px solid #343945; border-radius: 9px; font: inherit; }
    .primary { padding: 0 14px; color: #fff; background: #ff5a12; border: 0;
               border-radius: 9px; font: inherit; font-weight: 650; cursor: pointer; }
    .meta { padding: 0 13px 8px; color: #8b94a1; font-size: 11px; }
    .results { max-height: 50vh; overflow-y: auto; padding: 0 8px 8px; }
    .result { display: block; padding: 10px; color: inherit; text-decoration: none; border-radius: 10px; }
    .result:hover { background: #1f232b; }
    .result-meta, .snippet { margin-left: 27px; }
    .result-meta { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 6px;
                   color: #9aa3af; font-size: 11px; }
    .snippet { margin-top: 6px; color: #b6bec9; font-size: 12px; }
    .chip { padding: 1px 6px; background: #242a33; border: 1px solid #333b47; border-radius: 999px; }
    .foot { display: flex; gap: 6px; padding: 7px 12px; color: #7d8694;
            background: #14161b; border-top: 1px solid #2a2f38; font-size: 11px; }
    .endpoint { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .link-btn, .icon-btn { border: 0; background: none; color: #ff9a62; font: inherit; cursor: pointer; }
    .panel.collapsed { width: auto; min-width: 210px; }
    .panel.collapsed .body { display: none; }
  `;

  async function loadStyles(shadow) {
    try {
      const url = chrome.runtime.getURL("panel.css");
      const css = await (await fetch(url)).text();
      if (!css.trim()) throw new Error("empty stylesheet");
      if ("adoptedStyleSheets" in shadow && typeof CSSStyleSheet !== "undefined") {
        const sheet = new CSSStyleSheet();
        sheet.replaceSync(css);
        shadow.adoptedStyleSheets = [sheet];
        return;
      }
      const style = document.createElement("style");
      style.textContent = css;
      shadow.appendChild(style);
    } catch (err) {
      console.warn("[WSB search] cannot load panel.css, using fallback styles", err);
      const style = document.createElement("style");
      style.textContent = FALLBACK_CSS;
      shadow.appendChild(style);
    }
  }

  /*
   * A long-lived port to the background service worker is the primary transport.
   * It keeps the worker alive during a search and avoids the "message port closed
   * before a response was received" behaviour that one-shot sendMessage can show on
   * some Chromium builds. If connect() is unavailable, sendMessage is used instead.
   */
  let port = null;
  let messageSeq = 0;
  const pending = new Map(); // seq -> { resolve, reject }
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

  function sendViaMessage(message) {
    return new Promise((resolve) => {
      try {
        chrome.runtime.sendMessage(message, (response) => {
          const lastError = chrome.runtime.lastError;
          if (lastError) {
            resolve({
              ok: false,
              error:
                "Расширение потеряло связь с фоновым процессом " +
                `(${lastError.message}). Перезагрузите страницу.`,
            });
            return;
          }
          resolve(response);
        });
      } catch (err) {
        resolve({
          ok: false,
          error: `Расширение недоступно (${err}). Перезагрузите страницу.`,
        });
      }
    });
  }

  function send(message, timeoutMs = 50000) {
    return new Promise((resolve) => {
      const worker = openPort();
      if (!worker) {
        sendViaMessage(message).then(resolve);
        return;
      }

      const id = ++messageSeq;
      const timer = setTimeout(() => {
        if (!pending.has(id)) return;
        pending.delete(id);
        resolve({
          ok: false,
          error: "Фоновый процесс не ответил вовремя. Попробуйте ещё раз.",
        });
      }, timeoutMs);

      pending.set(id, {
        resolve: (reply) => {
          clearTimeout(timer);
          resolve(reply || { ok: false, error: "Пустой ответ от фонового процесса." });
        },
        reject: (err) => {
          clearTimeout(timer);
          resolve({
            ok: false,
            error:
              "Расширение потеряло связь с фоновым процессом " +
              `(${err && err.message ? err.message : err}). Попробуйте ещё раз.`,
          });
        },
      });

      try {
        worker.postMessage(Object.assign({ seq: id }, message));
      } catch (err) {
        pending.delete(id);
        clearTimeout(timer);
        resolve({
          ok: false,
          error: `Не удалось отправить сообщение (${err}). Перезагрузите страницу.`,
        });
      }
    });
  }

  function storageGet(defaults) {
    return new Promise((resolve) => {
      try {
        chrome.storage.local.get(defaults, (values) => {
          void chrome.runtime.lastError;
          resolve(values && Object.keys(values).length ? values : defaults);
        });
      } catch (_err) {
        resolve(defaults);
      }
    });
  }

  function storageSet(values) {
    try {
      chrome.storage.local.set(values, () => {
        void chrome.runtime.lastError;
      });
    } catch (_err) {
      /* storage is optional */
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

  function chip(text, extraClass) {
    const node = document.createElement("span");
    node.className = extraClass ? `chip ${extraClass}` : "chip";
    node.textContent = text;
    return node;
  }

  function safePostUrl(item, fallbackPostId) {
    const fallback =
      `https://www.reddit.com/r/wallstreetbets/comments/` +
      `${encodeURIComponent(fallbackPostId || "")}/`;
    if (!item || !item.url) return fallback;
    try {
      const url = new URL(String(item.url), "https://www.reddit.com");
      if (url.protocol === "http:" || url.protocol === "https:") return url.href;
    } catch (_err) {
      /* ignore malformed URL */
    }
    return fallback;
  }

  async function init() {
    const host = document.createElement("div");
    host.id = HOST_ID;
    const shadow = host.attachShadow({ mode: "open" });

    const template = document.createElement("template");
    template.innerHTML = TEMPLATE;
    await loadStyles(shadow);
    shadow.append(template.content);

    const panel = shadow.querySelector(".panel");
    const head = shadow.querySelector(".head");
    const collapseButton = shadow.querySelector(".collapse");
    const form = shadow.querySelector(".form");
    const input = shadow.querySelector(".query");
    const topk = shadow.querySelector(".topk");
    const primary = shadow.querySelector(".primary");
    const meta = shadow.querySelector(".meta");
    const results = shadow.querySelector(".results");
    const status = shadow.querySelector(".status");
    const endpoint = shadow.querySelector(".endpoint");
    const settingsButton = shadow.querySelector(".settings");

    document.documentElement.appendChild(host);

    let requestSeq = 0;
    let lastQuery = "";

    function setStatus(state, title) {
      status.dataset.state = state;
      status.title = title || "";
    }

    function setEndpoint(base, title) {
      let label = base || DEFAULT_SERVER;
      try {
        const url = new URL(label);
        label = url.host;
      } catch (_err) {
        /* keep as is */
      }
      endpoint.textContent = label;
      endpoint.title = title || base || DEFAULT_SERVER;
    }

    function clearMeta() {
      meta.hidden = true;
      meta.replaceChildren();
    }

    function renderLoading() {
      clearMeta();
      results.replaceChildren();
      const box = document.createElement("div");
      box.className = "loading";
      const spinner = document.createElement("span");
      spinner.className = "spinner";
      box.append(spinner, "Ищем по смыслу…");
      results.append(box);
    }

    function renderError(message) {
      clearMeta();
      results.replaceChildren();
      const box = document.createElement("div");
      box.className = "state";

      const title = document.createElement("div");
      title.className = "state-title";
      title.textContent = "Не удалось получить результаты";

      const text = document.createElement("div");
      text.className = "state-text";
      text.textContent = message || "Неизвестная ошибка.";

      const hint = document.createElement("div");
      hint.className = "state-hint";
      const code = document.createElement("code");
      code.textContent = "python server.py --host 127.0.0.1 --port 8000";
      hint.append("Проверьте, что локальный сервер запущен: ", code);

      const retry = document.createElement("button");
      retry.type = "button";
      retry.className = "link-btn";
      retry.textContent = "Повторить";
      retry.addEventListener("click", () => {
        if (lastQuery) runSearch(lastQuery);
      });

      box.append(title, text, hint, retry);
      results.append(box);
    }

    function renderResults(payload, elapsed) {
      clearMeta();
      results.replaceChildren();

      const normalized = String(payload.normalized_query || "");
      const original = String(payload.query || "");
      const items = Array.isArray(payload.results) ? payload.results : [];
      const requested = payload.top_k || topk.value;

      if (normalized && normalized.toLowerCase() !== original.toLowerCase()) {
        const line = document.createElement("div");
        line.append("нормализовано: ");
        const value = document.createElement("b");
        value.textContent = normalized;
        line.append(value);
        meta.append(line);
      }

      const count = document.createElement("div");
      count.textContent = `${items.length} из ${requested} · ${formatElapsed(elapsed)}`;
      meta.append(count);
      meta.hidden = false;

      if (!items.length) {
        const box = document.createElement("div");
        box.className = "state";
        const title = document.createElement("div");
        title.className = "state-title";
        title.textContent = "Ничего не найдено";
        const text = document.createElement("div");
        text.className = "state-text";
        text.textContent = "Попробуйте переформулировать запрос или увеличить top_k.";
        box.append(title, text);
        results.append(box);
        return;
      }

      const fragment = document.createDocumentFragment();

      items.forEach((item, index) => {
        const postId = String(item.post_id || "");
        const link = document.createElement("a");
        link.className = "result";
        link.href = safePostUrl(item, postId);
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.title = item.title || "Открыть пост на Reddit";

        const top = document.createElement("div");
        top.className = "result-top";

        const rank = document.createElement("span");
        rank.className = "rank";
        rank.textContent = `#${item.rank || index + 1}`;

        const title = document.createElement("span");
        title.className = "result-title";
        title.textContent = item.title || "(без заголовка)";

        top.append(rank, title);
        link.append(top);

        const metaRow = document.createElement("div");
        metaRow.className = "result-meta";
        if (item.year) metaRow.append(chip(String(item.year)));
        if (item.flair) metaRow.append(chip(String(item.flair)));

        const similarity = Number(item.score);
        if (Number.isFinite(similarity)) {
          const scoreChip = chip(`sim ${similarity.toFixed(3)}`, "score");
          scoreChip.title = "Косинусное сходство запроса и чанка";
          metaRow.append(scoreChip);
        }

        if (Array.isArray(item.tickers)) {
          item.tickers.slice(0, 4).forEach((ticker) => {
            metaRow.append(chip(`$${ticker}`, "ticker"));
          });
        }

        if (item.post_score !== undefined && item.post_score !== null) {
          metaRow.append(chip(`▲ ${formatNumber(item.post_score)}`));
        }
        if (item.num_comments !== undefined && item.num_comments !== null) {
          metaRow.append(chip(`💬 ${formatNumber(item.num_comments)}`));
        }
        if (metaRow.childNodes.length) link.append(metaRow);

        if (item.snippet) {
          const snippet = document.createElement("div");
          snippet.className = "snippet";
          snippet.textContent = item.snippet;
          link.append(snippet);
        }

        fragment.append(link);
      });

      results.append(fragment);
      results.scrollTop = 0;
    }

    async function runSearch(rawQuery) {
      const query = String(rawQuery || input.value || "").trim();
      if (!query) {
        input.focus();
        return;
      }

      lastQuery = query;
      const seq = ++requestSeq;
      const requestedTopK = Number(topk.value) || DEFAULT_TOP_K;
      const started = performance.now();

      setStatus("loading", "Поиск…");
      renderLoading();

      const response = await send({
        type: "wsb:search",
        query,
        topK: requestedTopK,
      });

      if (seq !== requestSeq) return;

      const elapsed = performance.now() - started;

      if (!response || !response.ok) {
        setStatus("error", "Ошибка запроса");
        setEndpoint(response && response.base, response && response.url);
        renderError((response && response.error) || "Пустой ответ от расширения.");
        return;
      }

      setStatus("ok", "Сервер доступен");
      setEndpoint(response.base, response.url);
      renderResults(response.data || {}, elapsed);
    }

    async function refreshHealth() {
      const response = await send({ type: "wsb:health" }, 7000);
      if (response && response.ok) {
        setStatus("ok", "Сервер доступен");
        setEndpoint(response.base, response.url);
      } else {
        setStatus("error", "Сервер недоступен");
        setEndpoint(response && response.base, response && response.url);
      }
    }

    function setCollapsed(collapsed) {
      panel.classList.toggle("collapsed", collapsed);
      collapseButton.textContent = collapsed ? "▴" : "▾";
      collapseButton.setAttribute(
        "aria-label",
        collapsed ? "Развернуть панель" : "Свернуть панель"
      );
    }

    function expandAndFocus() {
      setCollapsed(false);
      input.focus();
      input.select();
    }

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      event.stopPropagation();
      runSearch(input.value);
    });

    input.addEventListener("keydown", (event) => {
      // Do not let Reddit's global shortcuts (/, Enter, etc.) see our keys.
      event.stopPropagation();
      if (event.key === "Escape") {
        event.preventDefault();
        setCollapsed(true);
        input.blur();
      }
    });
    ["keyup", "keypress"].forEach((type) => {
      input.addEventListener(type, (event) => event.stopPropagation());
    });

    topk.addEventListener("change", () => {
      storageSet({ topK: Number(topk.value) || DEFAULT_TOP_K });
    });

    collapseButton.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      setCollapsed(!panel.classList.contains("collapsed"));
      if (!panel.classList.contains("collapsed")) input.focus();
    });

    head.addEventListener("click", (event) => {
      if (event.target.closest("button")) return;
      if (panel.classList.contains("collapsed")) expandAndFocus();
    });

    settingsButton.addEventListener("click", () => {
      send({ type: "wsb:open-options" });
    });

    document.addEventListener(
      "keydown",
      (event) => {
        if (
          event.altKey &&
          !event.ctrlKey &&
          !event.metaKey &&
          !event.shiftKey &&
          (event.code === "KeyS" || event.key.toLowerCase() === "s")
        ) {
          event.preventDefault();
          expandAndFocus();
        }
      },
      true
    );

    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible") refreshHealth();
    });

    try {
      chrome.storage.onChanged.addListener((changes, area) => {
        if (area === "local" && changes.serverUrl) refreshHealth();
      });
    } catch (_err) {
      /* storage is optional */
    }

    try {
      const stored = await storageGet({ topK: DEFAULT_TOP_K });
      const value = String(stored.topK);
      if (Array.from(topk.options).some((option) => option.value === value)) {
        topk.value = value;
      }
    } catch (_err) {
      /* ignore */
    }

    refreshHealth();
  }

  init().catch((err) => {
    console.error("[WSB search] failed to initialise panel", err);
  });
})();
