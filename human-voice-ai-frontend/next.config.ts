import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'standalone',
  reactStrictMode: true,
  swcMinify: true,
  images: {
    unoptimized: true, // Disable Image Optimization API
  },
  // Ensure proper routing in production
  trailingSlash: false,
  // Handle static exports
  distDir: '.next',
  // Enable server components
  experimental: {
    serverActions: {
      bodySizeLimit: '2mb',
    },
  },
};

export default nextConfig;
