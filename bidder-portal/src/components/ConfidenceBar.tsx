interface ConfidenceBarProps {
  pUp: number;
  pDown: number;
}

export function ConfidenceBar({ pUp, pDown }: ConfidenceBarProps) {
  return (
    <div className="inline-flex items-center justify-center font-mono text-sm">
      <span className="text-emerald-400 font-medium">{pUp.toFixed(1)}%</span>
      <span className="text-slate-300 mx-1.5">/</span>
      <span className="text-rose-400 font-medium">{pDown.toFixed(1)}%</span>
    </div>
  );
}
