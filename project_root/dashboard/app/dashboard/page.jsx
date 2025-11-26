"use client"; // เพิ่ม use client เผื่อไว้สำหรับ interactive component ในอนาคต
import Link from 'next/link';
import { useState } from 'react';
import Image from 'next/image';
import { Bell, User, X, FileText } from 'lucide-react';

export default function DashboardPage() {
  // --- State สำหรับ Modal ---
  // ถ้าเป็น null = ปิด Modal
  // ถ้ามีข้อมูล = เปิด Modal แสดงข้อมูลนั้น
  
  const [modalData, setModalData] = useState(null);

  const cardHeight = "h-[300px]";
  const miniCardHeight = "h-36";

  // ฟังก์ชันเปิด Modal
  const openModal = (title, src) => {
    setModalData({ title, src });
  };

  // ฟังก์ชันปิด Modal
  const closeModal = () => {
    setModalData(null);
  };

  return (
    <div className="min-h-screen bg-[#F8F9FA] p-4 md:p-6 font-sans text-slate-800 relative">
      
      {/* --- Modal Overlay (แสดงเมื่อ modalData ไม่เป็น null) --- */}
      {modalData && (
        <ImageModal 
          isOpen={!!modalData} 
          onClose={closeModal} 
          title={modalData.title} 
          src={modalData.src} 
        />
      )}

{/* --- Header --- */}
      <header className="mb-6 flex items-center justify-between bg-white px-6 py-3 shadow-sm rounded-xl border border-slate-100">
        
        {/* ส่วนซ้าย: ชื่อ + ปุ่มดู Report */}
        <div className="flex items-center gap-4">
          <h1 className="text-lg md:text-xl font-bold text-slate-800">Air Quality Analytics</h1>
          
          {/* ปุ่มใหม่: Link ไปหน้า Drift Report */}
          <Link 
            href="/drift" 
            className="hidden md:flex items-center gap-2 text-xs font-medium text-white bg-slate-800 hover:bg-slate-700 px-3 py-1.5 rounded-lg transition-colors shadow-sm"
          >
            <FileText className="w-3 h-3" />
            View Drift Report
          </Link>
        </div>

        {/* ส่วนขวา: Profile เหมือนเดิม */}
        <div className="flex items-center gap-3">
          <button className="rounded-full p-2 text-slate-400 hover:bg-slate-50 hover:text-slate-600 transition">
            <Bell className="h-5 w-5" />
          </button>
          <div className="h-9 w-9 rounded-full bg-slate-200 flex items-center justify-center overflow-hidden border border-slate-100 shadow-sm">
            <User className="h-5 w-5 text-slate-500" />
          </div>
        </div>
      </header>

      {/* --- Main Grid Content --- */}
      <main className="grid grid-cols-12 gap-4">

        {/* === ROW 1 === */}
        <DashboardCard 
          title="Overall Air Quality Index (AQI) Trend" 
          className={`col-span-12 lg:col-span-8 ${cardHeight}`}
          headerAction={<SelectDropdown />}
          // ส่งฟังก์ชัน onClick เพื่อเปิด Modal
          onClick={() => openModal("Overall Air Quality Index (AQI) Trend", "/eda_output/aqi_time_trend.png")}
        >
          <div className="relative h-full w-full">
            <Image src="/eda_output/aqi_time_trend.png" alt="AQI Trend" fill className="object-contain" />
          </div>
        </DashboardCard>

        <DashboardCard 
          title="Top 10 Stations / Bias" 
          className={`col-span-12 lg:col-span-4 ${cardHeight}`}
          onClick={() => openModal("Top 10 Stations / Bias", "/eda_output/station_bias.png")}
        >
          <div className="relative h-full w-full">
            <Image src="../../../eda_output/station_bias.png" alt="Top Stations" fill className="object-contain" />
          </div>
        </DashboardCard>

        {/* === ROW 2 === */}
        <DashboardCard 
          title="Correlation Matrix (Heatmap)" 
          className={`col-span-12 lg:col-span-6 ${cardHeight}`}
          onClick={() => openModal("Correlation Matrix (Heatmap)", "/eda_output/corr_heatmap.png")}
        >
          <div className="relative h-full w-full rounded-lg overflow-hidden flex items-center justify-center">
            <Image src="../../../eda_output/corr_heatmap.png" alt="Correlation Matrix" fill className="object-contain p-1" />
          </div>
        </DashboardCard>

        <DashboardCard 
          title="Correlation: AQI vs PM2.5" 
          className={`col-span-12 lg:col-span-6 ${cardHeight}`}
          onClick={() => openModal("Correlation: AQI vs PM2.5", "/eda_output/scatter_aqi_pm25.png")}
        >
          <div className="relative h-full w-full rounded-lg overflow-hidden">
            <Image src="../../../eda_output/scatter_aqi_pm25.png" alt="Scatter Plot" fill className="object-contain" />
          </div>
        </DashboardCard>

        {/* === ROW 3 (Mini Cards) === */}
        <div className="col-span-12 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          <MiniDistCard title="PM10 Dist." src="../../../eda_output/dist_pm10.png" height={miniCardHeight} onClick={() => openModal("PM10 Distribution", "../../../eda_output/dist_pm10.png")} />
          <MiniDistCard title="PM2.5 Dist." src="../../../eda_output/dist_pm25.png" height={miniCardHeight} onClick={() => openModal("PM2.5 Distribution", "../../../eda_output/dist_pm25.png")} />
          <MiniDistCard title="AQI Dist." src="../../../eda_output/dist_aqi.png" height={miniCardHeight} onClick={() => openModal("AQI Distribution", "../../../eda_output/dist_aqi.png")} />
          <MiniDistCard title="Temp (T) Dist." src="../../../eda_output/dist_t.png" height={miniCardHeight} onClick={() => openModal("Temperature Distribution", "../../../eda_output/dist_t.png")} />
          <MiniDistCard title="Humidity (H) Dist." src="../../../eda_output/dist_h.png" height={miniCardHeight} onClick={() => openModal("Humidity Distribution", "../../../eda_output/dist_h.png")} />
          <MiniDistCard title="Missing Values" src="../../../eda_output/missing_values.png" height={miniCardHeight} onClick={() => openModal("Missing Values", "../../../eda_output/missing_values.png")} />
        </div>

      </main>
    </div>
  );
}

// --- Component: Image Modal (ส่วนที่เพิ่มเข้ามา) ---
function ImageModal({ isOpen, onClose, title, src }) {
  if (!isOpen) return null;

  return (
    // Backdrop (พื้นหลังสีดำจางๆ)
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 md:p-8 transition-opacity duration-300"
      onClick={onClose} // คลิกพื้นหลังเพื่อปิด
    >
      {/* Modal Content (กล่องสีขาว) */}
      <div 
        className="bg-white rounded-xl shadow-2xl w-full max-w-5xl h-[60vh] md:h-[80vh] flex flex-col overflow-hidden animate-in fade-in zoom-in duration-200"
        onClick={(e) => e.stopPropagation()} // ป้องกันคลิกกล่องแล้วปิด
      >
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100">
          <h3 className="text-lg font-semibold text-gray-800">{title}</h3>
          <button 
            onClick={onClose}
            className="p-2 rounded-full hover:bg-gray-100 text-gray-500 transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Modal Body (รูปภาพ) */}
        <div className="flex-1 relative bg-gray-50 p-4">
          <Image 
            src={src} 
            alt={title} 
            fill 
            className="object-contain" // รูปจะขยายเต็มกล่องแต่รักษาสัดส่วน
            quality={100}
          />
        </div>
      </div>
    </div>
  );
}

// --- Reusable Components (เพิ่ม onClick และ Cursor Pointer) ---

function DashboardCard({ title, children, className = "", headerAction, onClick }) {
  return (
    <div 
      className={`flex flex-col bg-white rounded-xl shadow-[0_2px_8px_rgba(0,0,0,0.04)] border border-slate-100 p-4 ${className} transition-all duration-200 hover:shadow-md hover:border-blue-200 cursor-pointer group`}
      onClick={onClick}
    >
      <div className="flex items-center justify-between mb-2 truncate shrink-0">
        <h3 className="text-[13px] font-semibold text-slate-700 uppercase tracking-wider truncate mr-2 group-hover:text-blue-600 transition-colors">{title}</h3>
        {/* ใส่ onClick stopPropagation ที่ปุ่ม Dropdown เพื่อไม่ให้เผลอเปิด Modal เวลาเลือกเมนู */}
        <div onClick={(e) => e.stopPropagation()}>
          {headerAction}
        </div>
      </div>
      <div className="flex-1 relative min-h-0 rounded-md overflow-hidden">
        {children}
      </div>
    </div>
  );
}

function MiniDistCard({ title, src, height, onClick }) {
  return (
    <div 
      className={`flex flex-col bg-white rounded-xl shadow-[0_2px_5px_rgba(0,0,0,0.03)] border border-slate-100 p-3 ${height} hover:shadow-md hover:border-blue-200 hover:-translate-y-1 transition-all cursor-pointer group`}
      onClick={onClick}
    >
      <span className="text-[11px] font-medium text-slate-500 mb-1 truncate shrink-0 group-hover:text-blue-600 transition-colors">{title}</span>
      <div className="relative flex-1 w-full rounded-md overflow-hidden">
        <Image 
          src={src} 
          alt={title} 
          fill 
          className="object-contain" 
        />
      </div>
    </div>
  );
}

function SelectDropdown() {
  return (
    <div className="relative shrink-0">
      <select className="appearance-none bg-white border border-slate-200 text-slate-600 text-xs rounded-md py-1 pl-2 pr-6 cursor-pointer focus:outline-none focus:ring-1 focus:ring-blue-500 hover:border-blue-400 transition-colors">
        <option>Last 30 Days</option>
        <option>This Week</option>
      </select>
      <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-1 text-slate-400">
        <svg className="fill-current h-3 w-3" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z"/></svg>
      </div>
    </div>
  );
}