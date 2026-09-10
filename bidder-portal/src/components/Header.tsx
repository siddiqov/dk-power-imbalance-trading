import { ZoneSelector } from './ZoneSelector';
import { Activity } from 'lucide-react';

interface HeaderProps {
  priceArea: 'DK1' | 'DK2';
  onPriceAreaChange: (zone: 'DK1' | 'DK2') => void;
}

export function Header({ priceArea, onPriceAreaChange }: HeaderProps) {
  return (
    <header className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
      <div className="flex items-center gap-4">
        <div className="p-3 bg-indigo-600/20 rounded-xl border border-indigo-500/30">
          <Activity className="w-8 h-8 text-indigo-400" />
        </div>
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">Intraday Quarter Signals</h1>
          <p className="text-slate-400 font-medium mt-1">Transformer-TFT Model — Next 8 Tradeable Quarters</p>
        </div>
      </div>
      
      <ZoneSelector activeZone={priceArea} onChange={onPriceAreaChange} />
    </header>
  );
}
