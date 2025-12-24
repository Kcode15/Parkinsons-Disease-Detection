/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        'medical-blue': '#2563EB',
        'medical-pink': '#FCE7F3',
        'medical-gray': '#F8FAFC',
      },
    },
  },
  plugins: [],
}
