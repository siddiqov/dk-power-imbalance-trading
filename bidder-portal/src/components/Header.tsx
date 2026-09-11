import { ZoneSelector } from './ZoneSelector';
import { Activity } from 'lucide-react';
import { PortalViewMode } from '../types';

interface HeaderProps {
  priceArea: 'DK1' | 'DK2';
  onPriceAreaChange: (zone: 'DK1' | 'DK2') => void;
  viewMode?: PortalViewMode;
  dateStr?: string;
}

export function Header({ priceArea, onPriceAreaChange, viewMode = 'live', dateStr }: HeaderProps) {
  return (
    <header className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
      <div className="flex items-center gap-4">
        <div className="p-3 bg-indigo-600/20 rounded-xl border border-indigo-500/30">
          <Activity className="w-8 h-8 text-indigo-400" />
        </div>
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">Intraday Quarter Signals</h1>
          <p className="text-slate-400 font-medium mt-1">
            {viewMode === 'today'
              ? `Transformer-TFT Model — Full Day 96 Quarters (${dateStr || 'Today'})`
              : 'Transformer-TFT Model — Next 6 Tradeable Quarters'}
          </p>
        </div>
      </div>
      
      <ZoneSelector activeZone={priceArea} onChange={onPriceAreaChange} />
    </header>
  );
}
