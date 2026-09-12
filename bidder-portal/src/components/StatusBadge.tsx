import { CheckCircle2, Clock } from 'lucide-react';

interface StatusBadgeProps {
  status?: string | null;
}

export function StatusBadge({ status }: StatusBadgeProps) {
  const isSettled = status?.toUpperCase() === 'SETTLED';

  if (isSettled) {
    return (
      <div className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
        <CheckCircle2 className="w-3 h-3" />
        <span>Settled</span>
      </div>
    );
  }

  return (
    <div className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-semibold bg-amber-500/15 text-amber-400 border border-amber-500/30">
      <Clock className="w-3 h-3" />
      <span>Pending</span>
    </div>
  );
}
