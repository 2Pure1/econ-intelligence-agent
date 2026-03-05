import type { Metadata } from "next";
import { Inter, Geist_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const geistMono = Geist_Mono({ subsets: ["latin"], variable: "--font-geist-mono" });

export const metadata: Metadata = {
    title: "Econ Intel | AI Macro Analyst",
    description: "AI assistant for macroeconomic analysis — RAG + ML forecasts + live data",
};

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en" className="dark">
            <body className={`${inter.variable} ${geistMono.variable} antialiased selection:bg-cyan-500/30`}>
                {children}
            </body>
        </html>
    );
}
