import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "Human Voice AI",
  description: "An interactive voice AI application with emotion recognition",
  authors: [{ name: 'Jaden Fix', url: 'https://linkedin.com/in/jadenfix' }],
  keywords: ['voice ai', 'emotion recognition', 'next.js', 'react', 'tailwind css'],
  creator: 'Jaden Fix',
  publisher: 'Jaden Fix',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full">
      <body className={`${inter.variable} font-sans antialiased min-h-screen bg-background text-foreground`}>
        <main className="min-h-screen">
          {children}
        </main>
      </body>
    </html>
  );
}
