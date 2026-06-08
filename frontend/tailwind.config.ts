import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: "#f5f7fb",
        panel: "#ffffff",
        border: "#e5e7eb",
        accent: "#16a34a",
        danger: "#dc2626",
        text: "#0f172a",
        muted: "#6b7280",
      },
      borderRadius: {
        lg: "14px",
        md: "10px",
        sm: "8px",
      },
      boxShadow: {
        panel: "0 12px 40px rgba(15, 23, 42, 0.12)",
      },
      fontFamily: {
        sans: [
          "\"Noto Sans SC\"",
          "\"PingFang SC\"",
          "\"Microsoft YaHei\"",
          "Arial",
          "sans-serif",
        ],
      },
    },
  },
  plugins: [],
};
export default config;
