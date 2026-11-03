import { Cpu } from 'lucide-react';

export const AVAILABLE_MODELS = [
  {
    id: 'StormHunter',
    shortName: 'StormHunter',
    fullName: 'StormHunter (Conformal Sizing • Intraday Target)',
    badge: 'Storm',
  },
  {
    id: 'StormFusion',
    shortName: 'StormFusion',
    fullName: 'StormFusion (Multi-AI Ensemble • Chronos-2 + TiRex)',
    badge: 'Fusion',
  },
  {
    id: 'NeuralFlash AI',
    shortName: 'NeuralFlash AI',
    fullName: 'NeuralFlash AI (Deep Transformer • Zero-Shot Foundation)',
    badge: 'Neural',
  },
  {
    id: 'V4.1-HighAlpha',
    shortName: 'V4.1 High-Alpha',
    fullName: 'V4.1 High-Alpha Meta (Microstructure & Cross-Border Balancing)',
    badge: 'Alpha',
  },
  {
    id: 'V3.2-FlowAware',
    shortName: 'V3.2 Flow-Aware',
    fullName: 'V3.2 Flow-Aware Meta (Multi-Horizon Flow & Physical Circuit Breaker)',
    badge: 'V3.2',
  },
  {
    id: 'Transfer-LightGBM',
    shortName: 'Transfer-LightGBM',
    fullName: 'Transfer-LightGBM (Target Domain Transfer)',
    badge: 'LGBM',
  },
  {
    id: 'Transformer-TFT',
    shortName: 'TFT',
    fullName: 'Transformer-TFT (Primary Quantile Attention)',
    badge: 'Quantile',
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
    <div className="flex items-center gap-2">
      <div className="flex items-center gap-1.5 text-xs text-slate-400 font-medium">
        <Cpu className="w-3.5 h-3.5 text-indigo-400" />
        <span>Model:</span>
      </div>
      <div className="relative">
        <select
          value={selectedModel}
          onChange={(e) => onModelChange(e.target.value)}
          className="appearance-none bg-slate-900 border border-slate-700 hover:border-slate-600 rounded-lg pl-3 pr-8 py-1.5 text-xs font-semibold text-slate-200 cursor-pointer focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 transition-colors"
        >
          {AVAILABLE_MODELS.map((m) => (
            <option key={m.id} value={m.id} className="bg-slate-900 text-slate-200">
              {m.fullName}
            </option>
          ))}
        </select>
        <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-slate-400">
          <svg className="w-3 h-3 fill-current" viewBox="0 0 20 20">
            <path
              d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
              clipRule="evenodd"
              fillRule="evenodd"
            />
          </svg>
        </div>
      </div>
      {currentModel && (
        <span className="hidden sm:inline-block text-[10px] font-mono font-bold uppercase tracking-wider px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
          {currentModel.badge}
        </span>
      )}
    </div>
  );
}
