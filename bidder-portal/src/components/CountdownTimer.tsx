import { useState, useEffect } from 'react';
import { QuarterPrediction } from '../types';
import { Timer } from 'lucide-react';

interface CountdownTimerProps {
  nextQuarter?: QuarterPrediction;
  onExpire?: () => void;
}

export function CountdownTimer({ nextQuarter, onExpire }: CountdownTimerProps) {
  const [timeLeft, setTimeLeft] = useState<{ hours: number; minutes: number; seconds: number; isNegative: boolean } | null>(null);

  useEffect(() => {
    if (!nextQuarter) {
      setTimeLeft(null);
      return;
    }

    let hasExpired = false;

    const calculateTimeLeft = () => {
      // Gate closure is exactly 120 mins (2 hours) before delivery start (Nord Pool Close time)
      const deliveryStart = new Date(nextQuarter.time_dk).getTime();
      const gateClosure = deliveryStart - 120 * 60 * 1000;
      const now = Date.now();
      
      const difference = gateClosure - now;
      const isNegative = difference <= 0;
      const absDiff = Math.abs(difference);

      if (isNegative && !hasExpired) {
        hasExpired = true;
        onExpire?.();
      }

      const hours = Math.floor((absDiff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
      const minutes = Math.floor((absDiff % (1000 * 60 * 60)) / (1000 * 60));
      const seconds = Math.floor((absDiff % (1000 * 60)) / 1000);

      setTimeLeft({ hours, minutes, seconds, isNegative });
    };

    calculateTimeLeft();
    const timer = setInterval(calculateTimeLeft, 1000);

    return () => clearInterval(timer);
  }, [nextQuarter]);

  if (!nextQuarter || !timeLeft) {
    return <div className="flex items-center text-slate-500 gap-2"><Timer className="w-5 h-5"/> Waiting for next gate...</div>;
  }

  const { hours, minutes, seconds, isNegative } = timeLeft;
  
  // Total minutes until gate closure
  const totalMinutesLeft = (isNegative ? -1 : 1) * (hours * 60 + minutes);
  
  let colorClass = 'text-emerald-400';
  if (totalMinutesLeft < 1) {
    colorClass = 'text-red-500 animate-pulse';
  } else if (totalMinutesLeft < 5) {
    colorClass = 'text-amber-400';
  }

  const formatUnit = (unit: number) => unit.toString().padStart(2, '0');

  return (
    <div className="flex items-center gap-3 bg-slate-900/50 px-5 py-3 rounded-lg border border-slate-700/50">
      <Timer className={`w-5 h-5 ${colorClass}`} />
      <div className="flex flex-col">
        <span className="text-xs text-slate-400 font-medium uppercase tracking-wider">
          Gate closes in ({nextQuarter.quarter_label} · QH-{nextQuarter.quarter_index.toString().padStart(2, '0')})
        </span>
        <span className={`text-xl font-mono font-bold tracking-tight ${colorClass}`}>
          {isNegative ? '-' : ''}{formatUnit(hours)}h {formatUnit(minutes)}m {formatUnit(seconds)}s
        </span>
      </div>
    </div>
  );
}
