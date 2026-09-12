import { Cpu } from 'lucide-react';

export const AVAILABLE_MODELS = [
  {
    id: 'Transformer-TFT',
    shortName: 'TFT',
    fullName: 'Transformer-TFT (Primary Quantile Attention)',
    badge: 'Primary',
  },
  {
    id: 'Stacking-MetaEnsemble',
    shortName: 'Meta-Stack',
    fullName: 'Stacking-MetaEnsemble (Ridge Meta-Learner)',
    badge: 'Meta',
  },
  {
    id: 'Hierarchical-LGBM+XGB',
    shortName: 'LGBM+XGB',
    fullName: 'Hierarchical-LGBM+XGB (Two-Stage Tree Model)',
    badge: 'Ensemble',
  },
  {
    id: 'Transfer-LightGBM',
    shortName: 'Transfer',
    fullName: 'Transfer-LightGBM (Target Domain Transfer)',
    badge: 'Transfer',
  },
  {
    id: 'Pure15m-CatBoost',
    shortName: 'CatBoost',
    fullName: 'Pure15m-CatBoost (Native 15-Minute Categorical)',
    badge: 'CatBoost',
  },
  {
    id: 'Deep-BiLSTM',
    shortName: 'BiLSTM',
    fullName: 'Deep-BiLSTM (Bidirectional Recurrent Neural Net)',
    badge: 'Neural',
  },
];

interface ModelSelectorProps {
  selectedModel: string;
  onModelChange: (model: string) => void;
}

export function ModelSelector({ selectedModel, onModelChange }: ModelSelectorProps) {
  const currentModel = AVAILABLE_MODELS.find((m) => m.id === selectedModel);

  return (
    <div className="flex flex-wrap items-center justify-between gap-2.5 w-full bg-slate-850 px-4 py-2.5 rounded-xl border border-slate-700/80 shadow-md">
      {/* Model Tag Pills */}
      <div className="flex items-center gap-2 flex-wrap">
        <div className="flex items-center gap-1.5 text-slate-400 text-xs font-bold uppercase tracking-wider mr-1">
          <Cpu className="w-3.5 h-3.5 text-indigo-400" />
          <span>Model:</span>
        </div>

        {AVAILABLE_MODELS.map((m) => {
          const isActive = selectedModel === m.id;
          return (
            <button
              key={m.id}
              type="button"
              onClick={() => onModelChange(m.id)}
              title={`${m.fullName} [${m.badge}]`}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                isActive
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30 scale-102 border border-indigo-400/30'
                  : 'bg-slate-800 text-slate-400 hover:text-slate-200 hover:bg-slate-750 border border-slate-700/60'
              }`}
            >
              <span>{m.shortName}</span>
              {isActive && (
                <span className="ml-1.5 text-[10px] px-1 py-0.2 rounded bg-indigo-950/80 text-indigo-200 border border-indigo-500/40 uppercase font-mono">
                  {m.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Active Model Indicator */}
      <div className="hidden sm:flex items-center gap-2 text-xs font-mono text-slate-400">
        <span>Active Model:</span>
        <span className="font-semibold text-white px-2 py-0.5 rounded bg-slate-900 border border-slate-750">
          {currentModel ? currentModel.fullName.split(' (')[0] : selectedModel}
        </span>
      </div>
    </div>
  );
}
