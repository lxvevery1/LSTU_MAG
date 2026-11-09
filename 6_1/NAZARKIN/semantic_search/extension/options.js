"use strict";

const DEFAULT_SERVER = "http://127.0.0.1:8000";

const serverInput = document.getElementById("server");
const saveButton = document.getElementById("save");
const testButton = document.getElementById("test");
const resetButton = document.getElementById("reset");
const statusBox = document.getElementById("status");

function setStatus(state, text) {
  statusBox.dataset.state = state;
  statusBox.textContent = text;
}

function normalizeServer(value) {
  let trimmed = String(value || "").trim();
  if (!trimmed) return DEFAULT_SERVER;
  if (!/^https?:\/\//i.test(trimmed)) trimmed = `http://${trimmed}`;
  return trimmed.replace(/\/+$/, "");
}

function isLocalServer(url) {
  try {
    const host = new URL(url).hostname.toLowerCase();
    return host === "localhost" || host === "127.0.0.1" || host === "[::1]" || host === "::1";
  } catch (_err) {
    return false;
  }
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
  return new Promise((resolve) => {
    try {
      chrome.storage.local.set(values, () => {
        const err = chrome.runtime.lastError;
        resolve(err ? new Error(err.message) : null);
      });
    } catch (err) {
      resolve(err);
    }
  });
}

/*
 * Health checks go through the same long-lived port as the content script, so the
 * options page exercises exactly the transport the search panel uses.
 */
const PORT_NAME = "wsb-search";
let port = null;
let messageSeq = 0;
const pending = new Map(); // seq -> { resolve, reject }

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
  });

  return port;
}

function callBackground(message, timeoutMs = 7000) {
  return new Promise((resolve, reject) => {
    const worker = openPort();
    if (!worker) {
      reject(new Error("cannot reach the extension background worker"));
      return;
    }

    const id = ++messageSeq;
    const timer = setTimeout(() => {
      if (!pending.has(id)) return;
      pending.delete(id);
      reject(new Error("the background worker did not answer in time"));
    }, timeoutMs);

    pending.set(id, {
      resolve: (reply) => {
        clearTimeout(timer);
        resolve(reply);
      },
      reject: (err) => {
        clearTimeout(timer);
        reject(err);
      },
    });

    try {
      worker.postMessage(Object.assign({ seq: id }, message));
    } catch (err) {
      pending.delete(id);
      clearTimeout(timer);
      reject(err);
    }
  });
}

async function testConnection() {
  const url = normalizeServer(serverInput.value);
  setStatus("pending", `Проверяю ${url}/health …`);

  try {
    const reply = await callBackground({ type: "wsb:health", serverUrl: url });
    if (!reply || !reply.ok) {
      throw new Error((reply && reply.error) || "нет ответа от расширения");
    }
    const data = reply.data || {};
    const parts = [];
    if (data.model) parts.push(`модель: ${data.model}`);
    if (data.vectors !== undefined) parts.push(`векторов: ${data.vectors}`);
    if (data.version) parts.push(`версия API: ${data.version}`);
    setStatus(
      "ok",
      `Сервер доступен: ${reply.base || url}${parts.length ? `\n${parts.join(", ")}` : ""}`
    );
    return true;
  } catch (err) {
    setStatus(
      "error",
      `Нет соединения с ${url}: ${err && err.message ? err.message : err}\n` +
        "Убедитесь, что запущен server.py и порт совпадает."
    );
    return false;
  }
}

async function saveServer() {
  const url = normalizeServer(serverInput.value);
  if (!isLocalServer(url)) {
    setStatus(
      "error",
      "Адрес должен указывать на localhost или 127.0.0.1: " +
        "расширение имеет host-permissions только для локального сервера."
    );
    return false;
  }

  serverInput.value = url;
  const error = await storageSet({ serverUrl: url });
  if (error) {
    setStatus("error", `Не удалось сохранить настройки: ${error.message || error}`);
    return false;
  }
  setStatus("ok", `Сохранено: ${url}`);
  return true;
}

async function load() {
  const stored = await storageGet({ serverUrl: DEFAULT_SERVER });
  serverInput.value = stored.serverUrl || DEFAULT_SERVER;
  await testConnection();
}

saveButton.addEventListener("click", async () => {
  const saved = await saveServer();
  if (saved) await testConnection();
});

testButton.addEventListener("click", () => {
  testConnection();
});

resetButton.addEventListener("click", async () => {
  serverInput.value = DEFAULT_SERVER;
  const error = await storageSet({ serverUrl: DEFAULT_SERVER });
  if (error) {
    setStatus("error", `Не удалось сбросить настройки: ${error.message || error}`);
    return;
  }
  await testConnection();
});

serverInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    saveButton.click();
  }
});

load().catch((err) => {
  setStatus("error", `Не удалось загрузить настройки: ${err}`);
});
