/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        s: {
          bg:      "#0d1117",
          surface: "#161b22",
          card:    "#1c2128",
          border:  "#30363d",
          text:    "#e2e8f0",
          muted:   "#8b949e",
          subtle:  "#c9d1d9",
          accent:  "#58a6ff",
          accentD: "#1f6feb",
        }
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      fontSize: {
        "2xs": "10px",
        xs: "11px",
        sm: "12px",
        base: "13px",
        md: "14px",
        lg: "15px",
      }
    },
  },
  plugins: [],
}