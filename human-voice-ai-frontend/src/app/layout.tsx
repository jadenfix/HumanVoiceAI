import type { Metadata } from "next";
import { Inter } from "next/font/google";
import Link from "next/link";
import { usePathname } from "next/navigation";
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
  const pathname = typeof window !== "undefined" ? window.location.pathname : "/";

  const navItems = [
    { href: "/", label: "Home" },
    { href: "/learn", label: "Math & Architecture" },
    { href: "/about", label: "About" },
  ];

  return (
    <html lang="en" className="h-full">
      <body className={`${inter.variable} font-sans antialiased min-h-screen bg-background text-foreground`}>
        {/* Fixed Navigation Bar */}
        <header className="fixed top-0 left-0 right-0 z-50 backdrop-blur-sm bg-black/40 shadow-lg border-b border-white/10">
          <div className="max-w-6xl mx-auto flex items-center justify-between px-6 py-3">
            <Link href="/" className="text-lg font-semibold bg-gradient-to-r from-blue-300 via-purple-300 to-pink-300 bg-clip-text text-transparent">
              Voice&nbsp;AI
            </Link>
            <nav className="space-x-1">
              {navItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`relative px-4 py-2 text-sm font-semibold uppercase tracking-wide rounded-md transition-colors duration-300 hover:bg-white/10 ${pathname === item.href ? "border-b-2 border-purple-300 text-white" : "text-gray-300"}`}
                >
                  {item.label}
                </Link>
              ))}
            </nav>
          </div>
        </header>
        {/* Offset for fixed header */}
        <main className="pt-16 min-h-screen">
          {children}
        </main>
      </body>
    </html>
  );
}
