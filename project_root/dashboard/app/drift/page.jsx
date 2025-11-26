"use client";

import Link from 'next/link';
import { ArrowLeft, ExternalLink } from 'lucide-react';

export default function DriftReportPage() {
  return (
    <div className="flex flex-col h-screen bg-gray-50">
      
      {/* --- Header ย่อย --- */}
      <header className="flex-none bg-white px-6 py-3 shadow-sm border-b border-gray-200 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link 
            href="/dashboard" 
            className="flex items-center gap-2 text-sm font-medium text-slate-600 hover:text-blue-600 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Dashboard
          </Link>
          <div className="h-6 w-px bg-gray-300 mx-2"></div>
          <h1 className="text-lg font-bold text-slate-800">Model Drift Report</h1>
        </div>
        
        {/* ปุ่มเปิดแท็บใหม่ (เผื่ออยากดูเต็มๆ จอจริงๆ) */}
        <a 
          href="/drift_report.html" 
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 text-xs font-medium text-blue-600 hover:text-blue-700 bg-blue-50 px-3 py-1.5 rounded-md hover:bg-blue-100 transition-colors"
        >
          Open Original File <ExternalLink className="w-3 h-3" />
        </a>
      </header>

      {/* --- Iframe Content --- */}
      <div className="flex-1 w-full bg-white relative">
        <iframe 
          src="/drift_report_20251123_203921.html" 
          title="Drift Report"
          className="w-full h-full border-none absolute inset-0"
        />
      </div>
    </div>
  );
}