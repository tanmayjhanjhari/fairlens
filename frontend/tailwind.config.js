/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        primary:       "#0F172A",
        surface:       "#1E293B",
        card:          "#263349",
        accent:        "#14B8A6",
        accent2:       "#818CF8",
        danger:        "#EF4444",
        warning:       "#F59E0B",
        success:       "#22C55E",
        textPrimary:   "#F1F5F9",
        textSecondary: "#94A3B8",
      },
      fontFamily: {
        sans: ["Space Grotesk", "ui-sans-serif", "system-ui", "-apple-system", "sans-serif"],
      },
    },
  },
  plugins: [],
};
