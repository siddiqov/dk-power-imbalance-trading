import { useState, useEffect } from 'react';
import { QuarterPrediction } from '../types';
import { Timer } from 'lucide-react';

interface CountdownTimerProps {
  /**
   * The LAST locked quarter in the live view.
   * We count down to its delivery_start + 30 min, i.e. the next gate closure
   * (next batch gate = last locked quarter's delivery end + 15 min gate window).
   */
  lastLockedQuarter?: QuarterPrediction;
  onExpire?: () => void;
}

export function CountdownTimer({ lastLockedQuarter, onExpire }: CountdownTimerProps) {
  const [timeLeft, setTimeLeft] = useState<{ hours: number; minutes: number; seconds: number; isNegative: boolean } | null>(null);

  useEffect(() => {
    if (!lastLockedQuarter) {
      setTimeLeft(null);
      return;
    }

    let hasExpired = false;

    const calculateTimeLeft = () => {
      // Next gate closure = last locked quarter delivery_end + 15 min.
      // (i.e. the first quarter of the next batch must be locked before its gate closes,
      //  which is 2 hours before delivery = next_delivery_start - 120 min.
      //  Simplified: last_delivery_end = last_delivery_start + 15 min;
      //  next gate target = last_delivery_end + 15 min.)
      const lastDeliveryStart = new Date(lastLockedQuarter.time_dk).getTime();
      const nextGateClosure = lastDeliveryStart + 30 * 60 * 1000; // +30 min
      const now = Date.now();

      const difference = nextGateClosure - now;
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
  }, [lastLockedQuarter]);

  if (!lastLockedQuarter || !timeLeft) {
    return <div className="flex items-center text-slate-500 gap-2"><Timer className="w-5 h-5"/> Waiting for locked decisions...</div>;
  }

  const { hours, minutes, seconds, isNegative } = timeLeft;

  // Total minutes until next gate closure
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
          Next gate closure · after Q{lastLockedQuarter.quarter_label}
        </span>
        <span className={`text-xl font-mono font-bold tracking-tight ${colorClass}`}>
          {isNegative ? '-' : ''}{formatUnit(hours)}h {formatUnit(minutes)}m {formatUnit(seconds)}s
        </span>
      </div>
    </div>
  );
}
