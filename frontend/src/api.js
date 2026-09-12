import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api";
const TOKEN_KEY = "medtriage_user_token";

// ── Token management ──────────────────────────────────────────
async function getOrCreateToken() {
  let token = localStorage.getItem(TOKEN_KEY);
  if (!token) {
    try {
      const res = await fetch(`${BASE_URL}/token`, { method: "POST" });
      const data = await res.json();
      token = data.token;
      localStorage.setItem(TOKEN_KEY, token);
    } catch {
      token = crypto.randomUUID();
      localStorage.setItem(TOKEN_KEY, token);
    }
  }
  return token;
}

// ── Axios instance ────────────────────────────────────────────
const api = axios.create({
  baseURL: BASE_URL,
  headers: { "Content-Type": "application/json" },
});

api.interceptors.request.use(async (config) => {
  const token = await getOrCreateToken();
  config.headers["X-User-Token"] = token;
  return config;
});

// ── Exports ───────────────────────────────────────────────────
export const sendMessage   = (message, session_id) =>
  api.post("/chat", { message, session_id });

export const fetchHistory  = (session_id) =>
  api.get(`/history/${session_id}`);

export const fetchSessions = (limit = 20, offset = 0) =>
  api.get(`/sessions?limit=${limit}&offset=${offset}`);

export const getUserToken  = () => localStorage.getItem(TOKEN_KEY);
export const clearToken    = () => localStorage.removeItem(TOKEN_KEY);