/** @type {import('tailwindcss').Config} */
const { fontFamily } = require("tailwindcss/defaultTheme");

module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./pages/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        mont: ["var(--font-mont)", ...fontFamily.sans],
      },
      colors: {
        dark: "#0a0e17",
        light: "#f5f5f5",
        primary: "#2ca89a",
        primaryDark: "#58E6D9",
        coral: "#e8734a",
        navy: "#1a3a5c",
        gold: "#d4a843",
        surface: "#111827",
        surfaceLight: "#1f2937",
      },
      animation: {
        "spin-slow": "spin 8s linear infinite",
        "pulse-slow": "pulse 4s ease-in-out infinite",
      },
      backgroundImage: {
        circularDark:
          "repeating-radial-gradient(rgba(44,168,154,0.15) 2px,#0a0e17 8px,#0a0e17 100px)",
        circularDarkLg:
          "repeating-radial-gradient(rgba(44,168,154,0.15) 2px,#0a0e17 8px,#0a0e17 80px)",
        circularDarkMd:
          "repeating-radial-gradient(rgba(44,168,154,0.15) 2px,#0a0e17 8px,#0a0e17 60px)",
        circularDarkSm:
          "repeating-radial-gradient(rgba(44,168,154,0.15) 2px,#0a0e17 8px,#0a0e17 40px)",
      },
      boxShadow: {
        "3xl": "0 15px 15px 1px rgba(44,168,154, 0.3)",
        "glow": "0 0 30px rgba(44,168,154, 0.2)",
      },
    },
    screens: {
      "2xl": { max: "1535px" },
      xl: { max: "1279px" },
      lg: { max: "1023px" },
      md: { max: "767px" },
      sm: { max: "639px" },
      xs: { max: "479px" },
    },
  },
  plugins: [
    function ({ addVariant }) {
      addVariant("child", "& > *");
      addVariant("child-hover", "& > *:hover");
    },
  ],
};
