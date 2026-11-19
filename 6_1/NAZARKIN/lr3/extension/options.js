"use strict";

const DEFAULTS = {
  backendUrl: "http://127.0.0.1:8000",
  assistantUrl: "http://127.0.0.1:8010",
};

const backendInput = document.getElementById("backendUrl");
const assistantInput = document.getElementById("assistantUrl");
const status = document.getElementById("status");
const checkStatus = document.getElementById("checkStatus");

function normalize(value, fallback) {
  const trimmed = String(value || "").trim().replace(/\/+$/, "");
  if (!trimmed) return fallback;
  return /^https?:\/\//i.test(trimmed) ? trimmed : `http://${trimmed}`;
}

function load() {
  chrome.storage.local.get(DEFAULTS, (values) => {
    backendInput.value = values.backendUrl || DEFAULTS.backendUrl;
    assistantInput.value = values.assistantUrl || DEFAULTS.assistantUrl;
  });
}

document.getElementById("save").addEventListener("click", () => {
  const backendUrl = normalize(backendInput.value, DEFAULTS.backendUrl);
  const assistantUrl = normalize(assistantInput.value, DEFAULTS.assistantUrl);
  backendInput.value = backendUrl;
  assistantInput.value = assistantUrl;
  chrome.storage.local.set({ backendUrl, assistantUrl, serverUrl: backendUrl }, () => {
    status.textContent = "Сохранено";
    setTimeout(() => { status.textContent = ""; }, 2000);
  });
});

document.getElementById("check").addEventListener("click", async () => {
  checkStatus.textContent = " проверяю…";
  const backend = normalize(backendInput.value, DEFAULTS.backendUrl);
  const assistant = normalize(assistantInput.value, DEFAULTS.assistantUrl);
  const results = await Promise.all([
    fetch(`${backend}/health`).then((r) => r.ok).catch(() => false),
    fetch(`${assistant}/health`).then((r) => r.ok).catch(() => false),
  ]);
  checkStatus.textContent = ` backend: ${results[0] ? "ok" : "нет"}, assistant: ${results[1] ? "ok" : "нет"}`;
});

load();
