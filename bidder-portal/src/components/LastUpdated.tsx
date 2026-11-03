import { useState, useEffect } from 'react';
import { RefreshCw, Radio } from 'lucide-react';

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

  const isStale = minutesAgo >= 25;

  return (
    <div className="flex items-center gap-2.5 flex-wrap">
      <div className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border ${
        isStale
          ? 'bg-amber-500/10 text-amber-400 border-amber-500/30'
          : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
      }`}>
        <Radio className={`w-3.5 h-3.5 ${isStale ? 'text-amber-400' : 'text-emerald-400 animate-pulse'}`} />
        <span>{isStale ? 'Feed Transition' : 'Live 135m Gate'}</span>
      </div>

      <div className="flex items-center gap-2 text-xs text-slate-400 bg-slate-900/40 px-3 py-1.5 rounded-lg border border-slate-700/40">
        <RefreshCw className="w-3.5 h-3.5 text-slate-500" />
        <span>
          Updated {minutesAgo === 0 ? 'just now' : `${minutesAgo} min ago`}
        </span>
      </div>
    </div>
  );
}
