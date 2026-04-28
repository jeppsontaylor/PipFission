/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Mid-night palette tuned for a trading dashboard.
        ink: {
          950: "#08090d",
          900: "#0d0f15",
          800: "#13161e",
          700: "#1c2030",
          600: "#262b3d",
          500: "#3b4159",
        },
        accent: {
          500: "#3b82f6",
          600: "#2563eb",
        },
        bull: "#22c55e",
        bear: "#ef4444",
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Monaco", "monospace"],
      },
    },
  },
  plugins: [],
};
