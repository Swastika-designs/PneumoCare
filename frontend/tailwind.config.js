/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: { sans: ["Inter", "ui-sans-serif", "system-ui", "Segoe UI", "sans-serif"] },
      colors: {
        brand: { 50: "#effaf8", 100: "#d5f2ec", 200: "#ade4da", 500: "#14a38b", 600: "#0f8572", 700: "#0f6b5d", 900: "#0b3d36" },
      },
    },
  },
  plugins: [],
};
