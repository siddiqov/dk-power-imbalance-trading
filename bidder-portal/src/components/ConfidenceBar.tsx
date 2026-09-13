interface ConfidenceBarProps {
  pUp: number;
  pDown: number;
}

export function ConfidenceBar({ pUp, pDown }: ConfidenceBarProps) {
  const upVal = pUp <= 1.0 ? pUp * 100 : pUp;
  const downVal = pDown <= 1.0 ? pDown * 100 : pDown;

  return (
    <div className="inline-flex items-center justify-center font-mono text-sm">
      <span className="text-emerald-400 font-medium">{upVal.toFixed(1)}%</span>
      <span className="text-slate-300 mx-1.5">/</span>
      <span className="text-rose-400 font-medium">{downVal.toFixed(1)}%</span>
    </div>
  );
}
