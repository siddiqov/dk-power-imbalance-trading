import { useState, useEffect } from 'react';
import { QuarterPrediction } from '../types';
import { Timer } from 'lucide-react';

interface CountdownTimerProps {
  targetQuarter?: QuarterPrediction;
  nextQuarter?: QuarterPrediction;
  lastLockedQuarter?: QuarterPrediction;
  onExpire?: () => void;
}

export function CountdownTimer({
  targetQuarter,
  nextQuarter,
  lastLockedQuarter,
  onExpire,
}: CountdownTimerProps) {
  const activeQuarter = targetQuarter || nextQuarter || lastLockedQuarter;
  const [timeLeft, setTimeLeft] = useState<{
    hours: number;
    minutes: number;
    seconds: number;
    isNegative: boolean;
  } | null>(null);

  useEffect(() => {
    if (!activeQuarter || !activeQuarter.time_dk) {
      setTimeLeft(null);
      return;
    }

    let hasExpired = false;

    const calculateTimeLeft = () => {
      const deliveryStart = new Date(activeQuarter.time_dk).getTime();
      const now = Date.now();

      const difference = deliveryStart - now;
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
  }, [activeQuarter, onExpire]);

  if (!activeQuarter || !timeLeft) {
    return (
      <div className="flex items-center text-slate-500 gap-2">
        <Timer className="w-5 h-5 text-indigo-400 animate-spin" />
        <span className="text-xs">Synchronizing physical gate timer...</span>
      </div>
    );
  }

  const { hours, minutes, seconds, isNegative } = timeLeft;
  const totalMinutesLeft = (isNegative ? -1 : 1) * (hours * 60 + minutes);

  let colorClass = 'text-emerald-400';
  if (totalMinutesLeft < 15) {
    colorClass = 'text-amber-400 animate-pulse';
  } else if (totalMinutesLeft < 60) {
    colorClass = 'text-indigo-300';
  }

  const formatUnit = (unit: number) => unit.toString().padStart(2, '0');

  return (
    <div className="flex items-center gap-3 bg-slate-900/50 px-5 py-3 rounded-lg border border-slate-700/50">
      <Timer className={`w-5 h-5 ${colorClass}`} />
      <div className="flex flex-col">
        <span className="text-xs text-slate-400 font-medium uppercase tracking-wider">
          Delivery begins in ({activeQuarter.quarter_label} · QH-{String(activeQuarter.quarter_index).padStart(2, '0')})
        </span>
        <span className={`text-xl font-mono font-bold tracking-tight ${colorClass}`}>
          {isNegative ? '-' : ''}{formatUnit(hours)}h {formatUnit(minutes)}m {formatUnit(seconds)}s
        </span>
      </div>
    </div>
  );
}
