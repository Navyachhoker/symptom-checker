import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000/api",
  headers: { "Content-Type": "application/json" },
});

export const sendMessage  = (message, session_id) =>
  api.post("/chat", { message, session_id });

export const fetchHistory  = (session_id) =>
  api.get(`/history/${session_id}`);

export const fetchSessions = (limit = 20, offset = 0) =>
  api.get(`/sessions?limit=${limit}&offset=${offset}`);