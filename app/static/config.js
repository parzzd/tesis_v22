window.SICHER_CONFIG = {
  "API_BASE_URL": "",
  "WS_BASE_URL": ""
};
window.sicherApiUrl = function(path) {
  const base = (window.SICHER_CONFIG && window.SICHER_CONFIG.API_BASE_URL) || "";
  return base ? base.replace(/\/$/, "") + path : path;
};
window.sicherWsUrl = function(path) {
  const config = window.SICHER_CONFIG || {};
  const base = config.WS_BASE_URL || config.API_BASE_URL || "";
  if (!base) {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    return proto + "://" + window.location.host + path;
  }
  const url = new URL(base);
  const proto = url.protocol === "https:" ? "wss:" : "ws:";
  return proto + "//" + url.host + path;
};
