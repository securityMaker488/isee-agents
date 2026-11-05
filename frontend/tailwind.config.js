/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        isee: {
          bg: '#09090B',
          surface: '#111114',
          card: '#18181C',
          border: '#27272A',
          text: '#FAFAFA',
          muted: '#A1A1AA',
          red: '#DC2626',
          redHover: '#EF4444',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
};
