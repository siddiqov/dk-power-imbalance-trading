import { Cpu, TrendingUp, Clock } from 'lucide-react';

export const AVAILABLE_MODELS = [
  { id: 'Transformer-TFT', name: 'Transformer-TFT', badge: 'Primary' },
  { id: 'Stacking-MetaEnsemble', name: 'Stacking-MetaEnsemble', badge: 'Meta' },
  { id: 'Hierarchical-LGBM+XGB', name: 'Hierarchical-LGBM+XGB', badge: 'Ensemble' },
  { id: 'Transfer-LightGBM', name: 'Transfer-LightGBM', badge: 'Transfer' },
  { id: 'Pure15m-CatBoost', name: 'Pure15m-CatBoost', badge: 'CatBoost' },
  { id: 'Deep-BiLSTM', name: 'Deep-BiLSTM', badge: 'Neural' },
];

interface ModelSelectorProps {
  selectedModel: string;
  onModelChange: (model: string) => void;
  marketMode: 'INTRADAY_D0' | 'DAY_AHEAD_D1';
  onMarketModeChange: (mode: 'INTRADAY_D0' | 'DAY_AHEAD_D1') => void;
}

export function ModelSelector({
  selectedModel,
  onModelChange,
  marketMode,
  onMarketModeChange,
}: ModelSelectorProps) {
  return (
    <div className="flex flex-col lg:flex-row items-stretch lg:items-center justify-between gap-3 w-full bg-slate-800/90 p-3 rounded-xl border border-slate-700 shadow-md">
      {/* Market Mode Selector */}
      <div className="flex items-center gap-1.5 p-1 bg-slate-900 rounded-lg border border-slate-700/80">
        <button
          type="button"
          onClick={() => onMarketModeChange('INTRADAY_D0')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs sm:text-sm font-semibold transition-all cursor-pointer ${
            marketMode === 'INTRADAY_D0'
              ? 'bg-blue-600 text-white shadow-sm'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
          }`}
        >
          <TrendingUp className="w-4 h-4 text-blue-300" />
          <span>Intraday Continuous</span>
        </button>

        <button
          type="button"
          onClick={() => onMarketModeChange('DAY_AHEAD_D1')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs sm:text-sm font-semibold transition-all cursor-pointer ${
            marketMode === 'DAY_AHEAD_D1'
              ? 'bg-amber-600 text-white shadow-sm'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
          }`}
        >
          <Clock className="w-4 h-4 text-amber-300" />
          <span>Day-Ahead Auction</span>
        </button>
      </div>

      {/* Model Selection Tabs */}
      <div className="flex items-center gap-1.5 p-1 bg-slate-900 rounded-lg border border-slate-700/80 overflow-x-auto">
        <div className="flex items-center gap-1 px-2 text-slate-400 text-xs font-semibold uppercase tracking-wider">
          <Cpu className="w-3.5 h-3.5 text-indigo-400" />
          <span>Model:</span>
        </div>
        {AVAILABLE_MODELS.map((m) => (
          <button
            key={m.id}
            type="button"
            onClick={() => onModelChange(m.id)}
            className={`whitespace-nowrap px-3 py-1.5 rounded-md text-xs sm:text-sm font-medium transition-all cursor-pointer ${
              selectedModel === m.id
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
            }`}
          >
            {m.name}
          </button>
        ))}
      </div>
    </div>
  );
}
