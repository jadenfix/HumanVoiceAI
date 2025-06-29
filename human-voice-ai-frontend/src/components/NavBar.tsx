'use client';

import { useState } from 'react';
import Link from 'next/link';
import { FaBars, FaTimes } from 'react-icons/fa';

const navItems = [
  { href: '/', label: 'Home' },
  { href: '/math', label: 'Math' },
  { href: '/architecture', label: 'Architecture' },
  { href: '/about', label: 'About' },
];

export default function NavBar() {
  const [open, setOpen] = useState(false);

  return (
    <header className="fixed top-0 left-0 right-0 z-50 backdrop-blur-sm bg-black/40 shadow-lg border-b border-white/10 md:px-0">
      <div className="max-w-6xl mx-auto flex items-center justify-between px-6 py-3">
        <Link href="/" className="text-lg font-semibold bg-gradient-to-r from-blue-300 via-purple-300 to-pink-300 bg-clip-text text-transparent">
          Voice&nbsp;AI
        </Link>
        {/* Desktop nav */}
        <nav className="hidden md:flex space-x-1">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="relative px-4 py-2 text-sm font-semibold uppercase tracking-wide rounded-md transition-colors duration-300 hover:bg-white/10 text-gray-300"
            >
              {item.label}
            </Link>
          ))}
        </nav>
        {/* Mobile hamburger */}
        <button
          className="md:hidden text-gray-200 text-2xl focus:outline-none"
          onClick={() => setOpen(!open)}
          aria-label="Toggle navigation menu"
        >
          {open ? <FaTimes /> : <FaBars />}
        </button>
      </div>
      {/* Mobile menu */}
      {open && (
        <nav className="md:hidden bg-black/80 backdrop-blur-sm border-t border-white/10">
          <ul className="flex flex-col">
            {navItems.map((item) => (
              <li key={item.href} onClick={() => setOpen(false)}>
                <Link
                  href={item.href}
                  className="block px-6 py-4 text-sm font-semibold uppercase tracking-wide hover:bg-white/10 text-gray-200"
                >
                  {item.label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
      )}
    </header>
  );
} 