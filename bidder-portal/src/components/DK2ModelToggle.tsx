import { Zap, Wind, Brain } from 'lucide-react';

export type DK2ModelKey = 'StormHunter' | 'StormFusion' | 'NeuralFlash AI';

interface DK2ModelToggleProps {
  selectedModel: DK2ModelKey;
  onModelChange: (model: DK2ModelKey) => void;
}

export function DK2ModelToggle({ selectedModel, onModelChange }: DK2ModelToggleProps) {
  const models: { id: DK2ModelKey; label: string; icon: typeof Zap; desc: string; tag: string }[] = [
    {
      id: 'StormHunter',
      label: 'StormHunter',
      icon: Zap,
      desc: 'Conformal LightGBM • Intraday Target',
      tag: 'High Alpha',
    },
    {
      id: 'StormFusion',
      label: 'StormFusion',
      icon: Wind,
      desc: 'Multi-AI Ensemble • Chronos-2 + TiRex',
      tag: 'Multi-AI',
    },
    {
      id: 'NeuralFlash AI',
      label: 'NeuralFlash AI',
      icon: Brain,
      desc: 'Deep Transformer • Zero-Shot Foundation',
      tag: 'Deep Learning',
    },
  ];

  return (
    <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 bg-slate-800/90 p-2 sm:p-2.5 rounded-xl border border-slate-700 shadow-md">
      <div className="flex items-center gap-2 px-2 text-xs text-slate-400 font-medium">
        <span className="uppercase tracking-wider text-[11px] text-slate-400 font-bold">DK2 Strategy:</span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-1.5 flex-1">
        {models.map((m) => {
          const Icon = m.icon;
          const isActive = selectedModel === m.id;

          return (
            <button
              key={m.id}
              type="button"
              onClick={() => onModelChange(m.id)}
              className={`flex items-center justify-between sm:justify-center gap-2 px-3 py-2 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                isActive
                  ? 'bg-gradient-to-r from-indigo-600 to-indigo-500 text-white shadow-md shadow-indigo-500/20 border border-indigo-400/40'
                  : 'bg-slate-900/60 text-slate-300 hover:text-white hover:bg-slate-750 border border-slate-700/50'
              }`}
            >
              <div className="flex items-center gap-1.5">
                <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-white' : 'text-slate-400'}`} />
                <span>{m.label}</span>
              </div>
              <span
                className={`text-[9px] uppercase px-1.5 py-0.5 rounded font-mono ${
                  isActive ? 'bg-indigo-700/60 text-indigo-100' : 'bg-slate-800 text-slate-400'
                }`}
              >
                {m.tag}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
