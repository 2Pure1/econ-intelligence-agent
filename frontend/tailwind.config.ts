import type { Config } from "tailwindcss";

const config: Config = {
    content: [
        "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
        "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
        "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    ],
    theme: {
        extend: {
            colors: {
                surface: {
                    DEFAULT: "#0a0a0c",
                    1: "#121216",
                    2: "#181820",
                },
                border: "#202028",
                text: {
                    primary: "#f8fafc",
                    muted: "#94a3b8",
                    dim: "#475569",
                },
            },
            fontFamily: {
                mono: ["var(--font-geist-mono)", "ui-monospace", "SFMono-Regular", "Menlo", "Monaco", "Consolas", '"Liberation Mono"', '"Courier New"', "monospace"],
                display: ["Inter", "system-ui", "sans-serif"],
            },
            animation: {
                fadeIn: "fadeIn 0.3s ease-out forwards",
            },
            keyframes: {
                fadeIn: {
                    "0%": { opacity: "0", transform: "translateY(10px)" },
                    "100%": { opacity: "1", transform: "translateY(0)" },
                },
            },
        },
    },
    plugins: [],
};
export default config;
