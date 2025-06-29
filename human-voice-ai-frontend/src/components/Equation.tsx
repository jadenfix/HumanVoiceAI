'use client';

import { InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';

export default function Equation({ value }: { value: string }) {
  return (
    <div className="my-4 bg-black/30 backdrop-blur-sm rounded-xl shadow-inner overflow-x-auto px-4 py-2">
      <span className="inline-block min-w-max whitespace-nowrap text-lg md:text-xl lg:text-2xl text-gray-200 font-medium">
        <InlineMath math={value} />
      </span>
    </div>
  );
} 