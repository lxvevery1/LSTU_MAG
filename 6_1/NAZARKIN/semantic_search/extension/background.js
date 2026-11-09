/**
 * Background service worker.
 *
 * Content scripts on https://reddit.com cannot call http://127.0.0.1 directly:
 * the page is HTTPS and the local API does not send CORS headers. The service
 * worker has host permissions for localhost, so it can perform the request and
 * pass the JSON back to the content script.
 */
"use strict";

const DEFAULT_SERVER = "http://127.0.0.1:8000";
const HEALTH_TIMEOUT_MS = 5000;
const SEARCH_TIMEOUT_MS = 45000;

function normalizeServer(value) {
  const trimmed = String(value || "").trim().replace(/\/+$/, "");
  if (!trimmed) return DEFAULT_SERVER;
  if (/^https?:\/\//i.test(trimmed)) return trimmed;
  return `http://${trimmed}`;
}

async function getServerUrl() {
  try {
    const stored = await chrome.storage.local.get({ serverUrl: DEFAULT_SERVER });
    return normalizeServer(stored.serverUrl);
  } catch (_err) {
    return DEFAULT_SERVER;
  }
}

function failure(base, url, error) {
  return { ok: false, base, url, error };
}

/**
 * @param {string} path path with query string, e.g. "/search?q=tesla"
 * @param {{timeoutMs?: number, serverUrl?: string}} options
 */
async function requestJson(path, { timeoutMs = SEARCH_TIMEOUT_MS, serverUrl = "" } = {}) {
  const base = serverUrl ? normalizeServer(serverUrl) : await getServerUrl();
  const url = `${base}${path}`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      method: "GET",
      headers: { Accept: "application/json" },
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
        (payload && payload.error) ||
        text.slice(0, 300) ||
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
      return failure(base, url, `Таймаут запроса к ${url}. Сервер не ответил за ${Math.round(timeoutMs / 1000)} с.`);
    }
    return failure(
      base,
      url,
      `Не удалось подключиться к ${url}. ` +
        "Проверьте, что запущен локальный сервер: " +
        "python server.py --host 127.0.0.1 --port 8000. " +
        `(${err && err.message ? err.message : err})`
    );
  } finally {
    clearTimeout(timer);
  }
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

const PORT_NAME = "wsb-search";

function stringifyError(err) {
  return err && err.message ? err.message : String(err);
}

/*
 * One request router shared by both transports.
 *
 * A long-lived port (chrome.runtime.connect) is the primary transport: it keeps the
 * MV3 worker alive for the duration of a search and is more reliable than one-shot
 * sendMessage on some Chromium builds. sendMessage stays wired up as a fallback.
 */
async function dispatch(message) {
  if (!message || typeof message.type !== "string") return null;

  if (message.type === "wsb:search") {
    const query = String(message.query || "").trim();
    const rawTopK = Number.parseInt(message.topK, 10);
    const topK = Math.min(Math.max(Number.isFinite(rawTopK) ? rawTopK : 10, 1), 100);
    if (!query) return { ok: false, error: "Пустой поисковый запрос." };
    return search(query, topK);
  }

  if (message.type === "wsb:health") {
    return health(message.serverUrl);
  }

  if (message.type === "wsb:open-options") {
    chrome.runtime.openOptionsPage();
    return { ok: true };
  }

  return null;
}

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== PORT_NAME) return;

  port.onMessage.addListener(async (message) => {
    let reply;
    try {
      reply = await dispatch(message);
    } catch (err) {
      reply = { ok: false, error: stringifyError(err) };
    }
    if (!reply) return;
    try {
      // Echo the sequence number so the content script can match the reply.
      port.postMessage(Object.assign({ seq: message && message.seq }, reply));
    } catch (_err) {
      /* port closed while the request was in flight */
    }
  });
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || typeof message.type !== "string") return false;
  dispatch(message)
    .then((reply) => sendResponse(reply || { ok: false, error: "Неизвестный тип сообщения." }))
    .catch((err) => sendResponse({ ok: false, error: stringifyError(err) }));
  return true; // response is asynchronous
});

chrome.action.onClicked.addListener(() => {
  chrome.runtime.openOptionsPage();
});
