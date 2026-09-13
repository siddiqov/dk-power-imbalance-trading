interface ConfidenceBarProps {
  pUp: number;
  pDown: number;
}

export function ConfidenceBar({ pUp, pDown }: ConfidenceBarProps) {
  const upVal = Number(pUp ?? 0);
  const downVal = Number(pDown ?? 0);

  return (
    <div className="inline-flex items-center justify-center font-mono text-sm">
      <span className="text-emerald-400 font-medium">{upVal.toFixed(1)}%</span>
      <span className="text-slate-300 mx-1.5">/</span>
      <span className="text-rose-400 font-medium">{downVal.toFixed(1)}%</span>
    </div>
  );
}
