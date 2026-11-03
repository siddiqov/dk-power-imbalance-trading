import { ZoneSelector } from './ZoneSelector';
import { DenmarkClock } from './DenmarkClock';
import { Activity, Clock } from 'lucide-react';
import { PortalViewMode } from '../types';

interface HeaderProps {
  priceArea: 'DK1' | 'DK2';
  onPriceAreaChange: (zone: 'DK1' | 'DK2') => void;
  viewMode?: PortalViewMode;
  dateStr?: string;
  isDayAhead?: boolean;
  selectedModel?: string;
  isAllScreen?: boolean;
}

export function Header({
  priceArea,
  onPriceAreaChange,
  viewMode = 'live',
  dateStr,
  isDayAhead = false,
  selectedModel = 'StormHunter',
  isAllScreen = false,
}: HeaderProps) {
  const displayModelName =
    priceArea === 'DK1'
      ? 'StormHunter (Conformal LightGBM)'
      : selectedModel === 'StormHunter'
      ? 'StormHunter'
      : selectedModel === 'StormFusion'
      ? 'StormFusion (Multi-AI)'
      : selectedModel === 'NeuralFlash AI'
      ? 'NeuralFlash AI (Deep Transformer)'
      : selectedModel;

  return (
    <header className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
      {/* Title & Description */}
      <div className="flex items-center gap-4">
        <div className={`p-3 rounded-xl border ${isDayAhead ? 'bg-amber-600/20 border-amber-500/30' : 'bg-indigo-600/20 border-indigo-500/30'}`}>
          {isDayAhead ? (
            <Clock className="w-8 h-8 text-amber-400" />
          ) : (
            <Activity className="w-8 h-8 text-indigo-400" />
          )}
        </div>
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
              {isDayAhead ? 'Day-Ahead Auction Signals' : 'Intraday Quarter Signals'}
            </h1>
          </div>
          <p className="text-slate-400 font-medium text-xs sm:text-sm mt-1">
            <span className="text-indigo-300 font-semibold">{displayModelName}</span> —{' '}
            {viewMode === 'range'
              ? `Custom Range (${dateStr})`
              : 'Actionable 4 Quarters (Delivery in 2h 15m)'}
          </p>
        </div>
      </div>

      {/* Right Header: Denmark Clock & Zone Selector (customer view only) */}
      <div className="flex items-center gap-3 flex-wrap self-stretch lg:self-auto">
        {!isAllScreen && (
          <ZoneSelector activeZone={priceArea} onChange={onPriceAreaChange} />
        )}
        <DenmarkClock />
      </div>
    </header>
  );
}
