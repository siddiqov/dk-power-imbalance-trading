import { useState, useEffect } from 'react';
import { RefreshCw, AlertTriangle, CheckCircle2 } from 'lucide-react';

interface LastUpdatedProps {
  timestamp: Date | null;
}

export function LastUpdated({ timestamp }: LastUpdatedProps) {
  const [minutesAgo, setMinutesAgo] = useState<number>(0);

  useEffect(() => {
    if (!timestamp) return;

    const calculateTime = () => {
      const diffInMs = Date.now() - timestamp.getTime();
      setMinutesAgo(Math.floor(diffInMs / 60000));
    };

    calculateTime();
    const interval = setInterval(calculateTime, 30000); // Check every 30s

    return () => clearInterval(interval);
  }, [timestamp]);

  if (!timestamp) return null;

  const isDelayed = minutesAgo > 25;

  return (
    <div className="flex flex-wrap items-center gap-2">
      {isDelayed ? (
        <div className="flex items-center gap-1.5 text-xs text-amber-300 bg-amber-500/10 px-3 py-1.5 rounded-lg border border-amber-500/30">
          <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
          <span>Feed Delayed — Showing Last Confirmed Batch</span>
        </div>
      ) : (
        <div className="flex items-center gap-1.5 text-xs text-emerald-300 bg-emerald-500/10 px-3 py-1.5 rounded-lg border border-emerald-500/30">
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
          <span>Live 135m Gate</span>
        </div>
      )}
      <div className="flex items-center gap-1.5 text-xs text-slate-400 bg-slate-900/40 px-3 py-1.5 rounded-lg border border-slate-700/40">
        <RefreshCw className="w-3.5 h-3.5 text-slate-400 shrink-0" />
        <span>Updated {minutesAgo === 0 ? 'just now' : `${minutesAgo}m ago`}</span>
      </div>
    </div>
  );
}
