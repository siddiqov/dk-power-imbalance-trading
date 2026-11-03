import React from 'react';
import { Zap, Cpu, Sparkles } from 'lucide-react';

export type DK2ModelOption = 'StormHunter' | 'StormFusion' | 'NeuralFlash AI';

interface DK2ModelToggleProps {
  selectedModel: DK2ModelOption;
  onChange: (model: DK2ModelOption) => void;
  disabled?: boolean;
}

const MODELS: Array<{
  id: DK2ModelOption;
  name: string;
  tag: string;
  icon: React.ComponentType<{ className?: string }>;
  accentColor: string;
}> = [
  {
    id: 'StormHunter',
    name: 'StormHunter',
    tag: 'High Alpha',
    icon: Zap,
    accentColor: 'text-amber-400 bg-amber-500/10 border-amber-500/30',
  },
  {
    id: 'StormFusion',
    name: 'StormFusion',
    tag: 'Multi-AI',
    icon: Cpu,
    accentColor: 'text-indigo-400 bg-indigo-500/10 border-indigo-500/30',
  },
  {
    id: 'NeuralFlash AI',
    name: 'NeuralFlash AI',
    tag: 'Deep Learning',
    icon: Sparkles,
    accentColor: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30',
  },
];

export function DK2ModelToggle({ selectedModel, onChange, disabled = false }: DK2ModelToggleProps) {
  return (
    <div className="flex flex-col sm:flex-row items-start sm:items-center gap-2 bg-slate-850 p-1.5 rounded-xl border border-slate-750 shadow-md">
      <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider px-2">
        DK2 Algorithm:
      </span>
      <div className="grid grid-cols-3 gap-1.5 w-full sm:w-auto">
        {MODELS.map(({ id, name, tag, icon: Icon }) => {
          const isSelected = selectedModel === id;
          return (
            <button
              key={id}
              type="button"
              disabled={disabled}
              onClick={() => onChange(id)}
              className={`flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                isSelected
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30 border border-indigo-400/40 scale-[1.02]'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800 border border-transparent'
              } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
              title={`${name} (${tag})`}
            >
              <Icon className={`w-3.5 h-3.5 ${isSelected ? 'text-white' : 'text-slate-400'}`} />
              <span>{name}</span>
              <span
                className={`hidden md:inline-block text-[9px] px-1 py-0.2 rounded font-mono ${
                  isSelected ? 'bg-indigo-700/80 text-indigo-100' : 'bg-slate-800 text-slate-400'
                }`}
              >
                {tag}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
