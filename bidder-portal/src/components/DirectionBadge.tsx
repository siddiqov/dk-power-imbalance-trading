import { ArrowUp, ArrowDown, ArrowRight, ShieldAlert } from 'lucide-react';

interface DirectionBadgeProps {
  decision: string;
}

export function DirectionBadge({ decision }: DirectionBadgeProps) {
  const isCircuitBreaker = decision.includes('Circuit Breaker');
  const isCrashOverride = decision.includes('Crash Override') || decision.includes('CRASH PRED');
  const isBuy = !isCircuitBreaker && !isCrashOverride && (decision.includes('BUY') || decision.includes('Long'));
  const isSell = !isCircuitBreaker && !isCrashOverride && (decision.includes('SELL') || decision.includes('Short'));
  
  let Icon = ArrowRight;
  let colorClass = 'bg-gray-500/20 text-gray-400 border-gray-500/30';
  let label = 'HOLD';

  if (isCircuitBreaker) {
    Icon = ShieldAlert;
    colorClass = 'bg-amber-500/20 text-amber-400 border-amber-500/40 font-bold shadow-sm shadow-amber-500/10';
    label = 'HOLD (Circuit Breaker)';
  } else if (isCrashOverride) {
    Icon = ArrowDown;
    colorClass = 'bg-rose-500/30 text-rose-300 border-rose-500/50 font-bold shadow-sm shadow-rose-500/20';
    label = '🔥 SELL (Crash Override)';
  } else if (isBuy) {
    Icon = ArrowUp;
    colorClass = 'bg-green-500/20 text-green-400 border-green-500/30';
    label = 'BUY Spot (Long)';
  } else if (isSell) {
    Icon = ArrowDown;
    colorClass = 'bg-red-500/20 text-red-400 border-red-500/30';
    label = 'SELL Spot (Short)';
  } else {
    // If it's some form of HOLD
    label = decision; // 'HOLD' or other
  }

  return (
    <div className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border ${colorClass}`}>
      <Icon className="w-3.5 h-3.5 shrink-0" />
      <span>{label}</span>
    </div>
  );
}

