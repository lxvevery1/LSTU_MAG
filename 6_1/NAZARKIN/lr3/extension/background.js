/**
 * Background service worker.
 *
 * Content scripts on https://reddit.com cannot call http://127.0.0.1 directly
 * because the page is HTTPS and the local API does not send CORS headers.  The
 * service worker has host permissions for localhost, so it performs the request
 * and returns JSON to the content script.
 */
"use strict";

const DEFAULT_BACKEND = "http://127.0.0.1:8000";
const DEFAULT_ASSISTANT = "http://127.0.0.1:8010";
const HEALTH_TIMEOUT_MS = 5000;
const SEARCH_TIMEOUT_MS = 60000;
const CHAT_TIMEOUT_MS = 120000;

function normalizeServer(value, fallback) {
  const trimmed = String(value || "").trim().replace(/\/+$/, "");
  if (!trimmed) return fallback;
  if (/^https?:\/\//i.test(trimmed)) return trimmed;
  return `http://${trimmed}`;
}

async function getSettings() {
  try {
    const stored = await chrome.storage.local.get({
      backendUrl: DEFAULT_BACKEND,
      assistantUrl: DEFAULT_ASSISTANT,
      serverUrl: "",
    });
    return {
      backendUrl: normalizeServer(stored.backendUrl || stored.serverUrl, DEFAULT_BACKEND),
      assistantUrl: normalizeServer(stored.assistantUrl, DEFAULT_ASSISTANT),
    };
  } catch (_err) {
    return { backendUrl: DEFAULT_BACKEND, assistantUrl: DEFAULT_ASSISTANT };
  }
}

function failure(base, url, error) {
  return { ok: false, base, url, error };
}

async function requestJson(path, {
  timeoutMs = SEARCH_TIMEOUT_MS,
  serverUrl = "",
  method = "GET",
  body = null,
} = {}) {
  const settings = await getSettings();
  const base = serverUrl ? normalizeServer(serverUrl, settings.backendUrl) : settings.backendUrl;
  const url = `${base}${path}`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      method,
      headers: {
        Accept: "application/json",
        ...(body ? { "Content-Type": "application/json" } : {}),
      },
      body: body ? JSON.stringify(body) : undefined,
      cache: "no-store",
      signal: controller.signal,
    });
    const text = await response.text();
    let payload = null;
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch (_err) {
        payload = null;
      }
    }
    if (!response.ok) {
      const detail =
        (payload && payload.error && payload.error.message) ||
        (payload && payload.error) ||
        text.slice(0, 400) ||
        response.statusText ||
        "unknown error";
      return failure(base, url, `HTTP ${response.status}: ${detail}`);
    }
    if (payload === null) {
      return failure(base, url, "Сервер вернул ответ, который не является JSON.");
    }
    return { ok: true, base, url, data: payload };
  } catch (err) {
    if (err && err.name === "AbortError") {
      return failure(base, url, `Таймаут запроса к ${url}.`);
    }
    return failure(
      base,
      url,
      `Не удалось подключиться к ${url}. ` +
        "Проверьте, что запущены backend и LLM-сервис: " +
        "python backend/app.py и python llm_service/app.py. " +
        `(${err && err.message ? err.message : err})`
    );
  } finally {
    clearTimeout(timer);
  }
}

async function assistantRequest(path, body, timeoutMs = CHAT_TIMEOUT_MS) {
  const settings = await getSettings();
  const result = await requestJson(path, {
    serverUrl: settings.assistantUrl,
    method: "POST",
    body,
    timeoutMs,
  });
  return result;
}

function search(query, topK) {
  const params = new URLSearchParams({
    q: query,
    top_k: String(topK),
    normalize: "1",
  });
  return requestJson(`/search?${params.toString()}`);
}

function health(serverUrl) {
  return requestJson("/health", { timeoutMs: HEALTH_TIMEOUT_MS, serverUrl });
}

function assistantHealth() {
  return getSettings().then((settings) =>
    requestJson("/health", { timeoutMs: HEALTH_TIMEOUT_MS, serverUrl: settings.assistantUrl })
  );
}

async function dispatch(message) {
  if (!message || typeof message.type !== "string") return null;

  if (message.type === "wsb:search") {
    const query = String(message.query || "").trim();
    const rawTopK = Number.parseInt(message.topK, 10);
    const topK = Math.min(Math.max(Number.isFinite(rawTopK) ? rawTopK : 10, 1), 100);
    if (!query) return { ok: false, error: "Пустой поисковый запрос." };
    return search(query, topK);
  }

  if (message.type === "wsb:health") return health(message.serverUrl);
  if (message.type === "wsb:assistant-health") return assistantHealth();

  if (message.type === "wsb:assistant-chat") {
    return assistantRequest("/assistant/chat", {
      message: String(message.message || ""),
      conversation_id: message.conversationId || undefined,
      context: message.context || {},
    });
  }

  if (message.type === "wsb:assistant-confirm") {
    return assistantRequest("/assistant/confirm", {
      action_id: String(message.actionId || ""),
      decision: String(message.decision || ""),
      conversation_id: message.conversationId || undefined,
    });
  }

  if (message.type === "wsb:history") {
    const cid = encodeURIComponent(String(message.conversationId || ""));
    return requestJson(`/assistant/history?conversation_id=${cid}`, {
      serverUrl: (await getSettings()).assistantUrl,
      timeoutMs: HEALTH_TIMEOUT_MS,
    });
  }

  if (message.type === "wsb:open-options") {
    chrome.runtime.openOptionsPage();
    return { ok: true };
  }

  return null;
}

const PORT_NAME = "wsb-thread-radar";

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== PORT_NAME) return;
  port.onMessage.addListener(async (message) => {
    let reply;
    try {
      reply = await dispatch(message);
    } catch (err) {
      reply = { ok: false, error: err && err.message ? err.message : String(err) };
    }
    if (!reply) return;
    try {
      port.postMessage(Object.assign({ seq: message && message.seq }, reply));
    } catch (_err) {
      /* port closed */
    }
  });
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || typeof message.type !== "string") return false;
  dispatch(message)
    .then((reply) => sendResponse(reply || { ok: false, error: "Неизвестный тип сообщения." }))
    .catch((err) => sendResponse({ ok: false, error: String(err) }));
  return true;
});

chrome.action.onClicked.addListener(() => {
  chrome.runtime.openOptionsPage();
});
