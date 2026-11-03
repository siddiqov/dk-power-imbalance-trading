import { useState, useEffect } from 'react';
import { QuarterPrediction } from '../types';
import { Timer } from 'lucide-react';

interface CountdownTimerProps {
  /**
   * The upcoming batch's first quarter (delivery starting in ~2h 15m)
   * or the last locked quarter.
   */
  targetQuarter?: QuarterPrediction;
  lastLockedQuarter?: QuarterPrediction;
  onExpire?: () => void;
}

export function CountdownTimer({ targetQuarter, lastLockedQuarter, onExpire }: CountdownTimerProps) {
  const activeQuarter = targetQuarter || lastLockedQuarter;
  const [timeLeft, setTimeLeft] = useState<{ hours: number; minutes: number; seconds: number; isNegative: boolean } | null>(null);

  useEffect(() => {
    if (!activeQuarter) {
      setTimeLeft(null);
      return;
    }

    let hasExpired = false;

    const calculateTimeLeft = () => {
      const deliveryStart = new Date(activeQuarter.time_dk).getTime();
      const now = Date.now();

      // Time until the delivery of this batch starts
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
  }, [activeQuarter]);

  if (!activeQuarter || !timeLeft) {
    return (
      <div className="flex items-center text-slate-500 gap-2">
        <Timer className="w-5 h-5" />
        <span>Waiting for upcoming quarter batch...</span>
      </div>
    );
  }

  const { hours, minutes, seconds, isNegative } = timeLeft;
  const totalMinutesLeft = (isNegative ? -1 : 1) * (hours * 60 + minutes);

  let colorClass = 'text-emerald-400';
  if (totalMinutesLeft < 15) {
    colorClass = 'text-red-500 animate-pulse';
  } else if (totalMinutesLeft < 60) {
    colorClass = 'text-amber-400';
  }

  const formatUnit = (unit: number) => unit.toString().padStart(2, '0');

  // Format delivery start time (e.g., 10:00)
  let startTimeLabel = '';
  try {
    const d = new Date(activeQuarter.time_dk);
    startTimeLabel = d.toLocaleTimeString('da-DK', { timeZone: 'Europe/Copenhagen', hour: '2-digit', minute: '2-digit' });
  } catch {
    startTimeLabel = activeQuarter.quarter_label;
  }

  return (
    <div className="flex items-center gap-3 bg-slate-900/50 px-5 py-3 rounded-lg border border-slate-700/50">
      <Timer className={`w-5 h-5 ${colorClass}`} />
      <div className="flex flex-col">
        <span className="text-xs text-slate-400 font-medium uppercase tracking-wider">
          {isNegative ? 'Batch Active Since' : 'Batch Delivery Starts At'}{' '}
          <span className="text-slate-200 font-bold">{startTimeLabel} ({activeQuarter.quarter_label})</span>
        </span>
        <span className={`text-xl font-mono font-bold tracking-tight ${colorClass}`}>
          {isNegative ? '-' : ''}{formatUnit(hours)}h {formatUnit(minutes)}m {formatUnit(seconds)}s
        </span>
      </div>
    </div>
  );
}
