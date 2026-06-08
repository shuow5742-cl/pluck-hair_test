const DEFAULT_API_BASE = "http://localhost:8000/api";

function getApiBase() {
  const base = import.meta.env.VITE_API_BASE ?? DEFAULT_API_BASE;
  try {
    return new URL(base, window.location.origin).toString().replace(/\/$/, "");
  } catch {
    return DEFAULT_API_BASE;
  }
}

export function buildApiUrl(path: string) {
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }

  const base = getApiBase();
  const normalizedPath = path.startsWith("/") ? path.slice(1) : path;

  try {
    const url = new URL(base);
    url.pathname = `${url.pathname.replace(/\/$/, "")}/${normalizedPath}`;
    url.search = "";
    url.hash = "";
    return url.toString();
  } catch {
    return `${base}/${normalizedPath}`;
  }
}

export function buildWsUrl() {
  const explicit = import.meta.env.VITE_WS_URL;
  if (explicit) return explicit;

  const apiBase = getApiBase();
  const baseUrl = new URL(apiBase);
  const normalizedPath = baseUrl.pathname.replace(/\/$/, "");
  const apiPath = normalizedPath.endsWith("/api")
    ? normalizedPath
    : `${normalizedPath}/api`;

  baseUrl.protocol = baseUrl.protocol === "https:" ? "wss:" : "ws:";
  baseUrl.pathname = `${apiPath}/ws/events`;
  baseUrl.search = "";
  baseUrl.hash = "";

  return baseUrl.toString();
}
