/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          bg:      "#0a0f1a",
          surface: "#111827",
          card:    "#1a2233",
          border:  "#1e2d45",
          text:    "#e2e8f0",
          muted:   "#64748b",
          subtle:  "#94a3b8",
          accent:  "#3b82f6",
          accentHover: "#2563eb",
        },
        urgency: {
          low:       { DEFAULT: "#22c55e", bg: "#052e16", border: "#166534" },
          moderate:  { DEFAULT: "#f59e0b", bg: "#1c1105", border: "#92400e" },
          high:      { DEFAULT: "#ef4444", bg: "#1c0505", border: "#991b1b" },
          emergency: { DEFAULT: "#f97316", bg: "#1c0a05", border: "#9a3412" },
        }
      },
    },
  },
  plugins: [],
}