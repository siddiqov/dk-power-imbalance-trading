import { useState, useEffect } from 'react';
import { RefreshCw } from 'lucide-react';

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

  return (
    <div className="flex items-center gap-2 text-sm text-slate-400 bg-slate-900/30 px-4 py-2 rounded-lg border border-slate-700/30">
      <RefreshCw className="w-4 h-4" />
      <span>
        Updated {minutesAgo === 0 ? 'just now' : `${minutesAgo} min ago`}
      </span>
    </div>
  );
}
