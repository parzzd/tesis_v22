window.SICHER_CONFIG = {
  "API_BASE_URL": "",
  "WS_BASE_URL": "",

  // ===== MediaMTX (publicar la webcam del navegador por WebRTC/WHIP) =====
  // Endpoint WebRTC de MediaMTX al que el NAVEGADOR publica (puerto 8889).
  // Si se deja vacio se deriva del host actual usando el puerto 8889.
  "MEDIAMTX_WHIP_BASE": "",
  // Usuario/clave de publicacion (deben coincidir con authInternalUsers en mediamtx.yml).
  "MEDIAMTX_PUBLISH_USER": "tesis",
  "MEDIAMTX_PUBLISH_PASS": "tesis",
  // Plantilla de la URL RTSP que LEE el backend (corre junto a MediaMTX en EC2).
  // {path} se reemplaza por la ruta generada para la camara.
  "MEDIAMTX_RTSP_TEMPLATE": "rtsp://tesis:tesis@127.0.0.1:8554/{path}"
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
window.sicherWebcamConfig = function() {
  const cfg = window.SICHER_CONFIG || {};
  let whipBase = (cfg.MEDIAMTX_WHIP_BASE || "").replace(/\/$/, "");
  if (!whipBase) {
    const host = window.location.hostname;
    if (host === "localhost" || host === "127.0.0.1") {
      // Local: el navegador publica directo a MediaMTX en el puerto 8889.
      whipBase = window.location.protocol + "//" + host + ":8889";
    } else {
      // Produccion (EC2): se publica al mismo origen HTTPS bajo /whip,
      // que nginx reenvia a MediaMTX. Evita el bloqueo por mixed-content.
      whipBase = window.location.origin + "/whip";
    }
  }
  return {
    whipBase,
    user: cfg.MEDIAMTX_PUBLISH_USER || "",
    pass: cfg.MEDIAMTX_PUBLISH_PASS || "",
    rtspTemplate: cfg.MEDIAMTX_RTSP_TEMPLATE || "rtsp://127.0.0.1:8554/{path}"
  };
};
