import { ArrowUp, ArrowDown, ArrowRight } from 'lucide-react';

interface DirectionBadgeProps {
  decision: string;
}

export function DirectionBadge({ decision }: DirectionBadgeProps) {
  const isBuy = decision.includes('BUY') || decision.includes('Long');
  const isSell = decision.includes('SELL') || decision.includes('Short');
  
  let Icon = ArrowRight;
  let colorClass = 'bg-gray-500/20 text-gray-400 border-gray-500/30';
  let label = 'HOLD';

  if (isBuy) {
    Icon = ArrowUp;
    colorClass = 'bg-green-500/20 text-green-400 border-green-500/30';
    label = 'BUY Spot (Long)';
  } else if (isSell) {
    Icon = ArrowDown;
    colorClass = 'bg-red-500/20 text-red-400 border-red-500/30';
    label = 'SELL Spot (Short)';
  } else {
    // If it's some form of HOLD
    label = decision; // 'HOLD' or 'HOLD (Spike Shield Protected)'
  }

  return (
    <div className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border ${colorClass}`}>
      <Icon className="w-3.5 h-3.5" />
      <span>{label}</span>
    </div>
  );
}
