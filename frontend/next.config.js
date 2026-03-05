/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: true,
    images: {
        unoptimized: true,
    },
    // Allow connections from anywhere in dev
    async headers() {
        return [
            {
                source: "/api/:path*",
                headers: [
                    { key: "Access-Control-Allow-Origin", value: "*" },
                    { key: "Access-Control-Allow-Methods", value: "GET,POST,PUT,DELETE,OPTIONS" },
                    { key: "Access-Control-Allow-Headers", value: "Content-Type, X-Session-ID" },
                ],
            },
        ];
    },
};

module.exports = nextConfig;
