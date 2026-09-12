import { useState, useEffect } from 'react';
import { Clock } from 'lucide-react';

export function DenmarkClock() {
  const [timeStr, setTimeStr] = useState<string>('');
  const [dateStr, setDateStr] = useState<string>('');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      try {
        const timeFormatter = new Intl.DateTimeFormat('en-GB', {
          timeZone: 'Europe/Copenhagen',
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
          hour12: false,
        });
        const dateFormatter = new Intl.DateTimeFormat('en-GB', {
          timeZone: 'Europe/Copenhagen',
          day: '2-digit',
          month: 'short',
        });
        setTimeStr(timeFormatter.format(now));
        setDateStr(dateFormatter.format(now));
      } catch {
        setTimeStr(now.toLocaleTimeString());
      }
    };

    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex items-center gap-2.5 bg-slate-900/60 px-4 py-2.5 rounded-lg border border-slate-700/60 shadow-inner">
      <Clock className="w-4 h-4 text-cyan-400 animate-pulse" />
      <div className="flex flex-col text-left">
        <span className="text-[10px] uppercase font-semibold text-slate-400 tracking-wider flex items-center gap-1">
          <span>🇩🇰 Denmark (CET)</span>
          {dateStr && <span className="text-slate-500">• {dateStr}</span>}
        </span>
        <span className="text-sm sm:text-base font-mono font-bold text-cyan-300 tracking-tight">
          {timeStr || '--:--:--'}
        </span>
      </div>
    </div>
  );
}
