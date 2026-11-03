import { ZoneSelector } from './ZoneSelector';
import { DenmarkClock } from './DenmarkClock';
import { Activity, Clock, ShieldCheck } from 'lucide-react';
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
  const displayModel = priceArea === 'DK1' && !isAllScreen ? 'StormHunter' : selectedModel;

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
          <div className="flex items-center gap-2.5 flex-wrap">
            <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
              {isDayAhead ? 'Day-Ahead Auction Signals' : 'Intraday Quarter Signals'}
            </h1>
            {priceArea === 'DK1' && !isAllScreen && (
              <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/30">
                <ShieldCheck className="w-3.5 h-3.5" />
                StormHunter Champion
              </span>
            )}
          </div>
          <p className="text-slate-400 font-medium text-xs sm:text-sm mt-1">
            {displayModel} Algorithm —{' '}
            {viewMode === 'range'
              ? `Custom Range (${dateStr})`
              : '4 Delivery Quarters (135m Gate)'}
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
