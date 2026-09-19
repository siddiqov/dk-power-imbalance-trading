import { CheckCircle2, Clock, Lock, AlertTriangle } from 'lucide-react';

interface StatusBadgeProps {
  status?: string | null;
}

export function StatusBadge({ status }: StatusBadgeProps) {
  const s = (status || '').toUpperCase();

  if (s === 'SETTLED' || s === 'SETTLED_AUDITED') {
    return (
      <div className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
        <CheckCircle2 className="w-3 h-3" />
        <span>Settled</span>
      </div>
    );
  }

  if (s === 'LOCKED' || s === 'LOCKED_PENDING') {
    return (
      <div className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/40">
        <Lock className="w-3 h-3" />
        <span>Locked</span>
      </div>
    );
  }

  if (s === 'MISSED') {
    return (
      <div className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-semibold bg-amber-500/15 text-amber-400 border border-amber-500/30">
        <AlertTriangle className="w-3 h-3" />
        <span>Missed Gate</span>
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
