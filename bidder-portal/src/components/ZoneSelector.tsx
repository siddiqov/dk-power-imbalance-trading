interface ZoneSelectorProps {
  activeZone: 'DK1' | 'DK2';
  onChange: (zone: 'DK1' | 'DK2') => void;
}

export function ZoneSelector({ activeZone, onChange }: ZoneSelectorProps) {
  return (
    <div className="flex bg-slate-800 p-1 rounded-lg border border-slate-700">
      {(['DK1', 'DK2'] as const).map((zone) => (
        <button
          key={zone}
          onClick={() => onChange(zone)}
          className={`px-6 py-2 rounded-md font-semibold transition-all ${
            activeZone === zone
              ? 'bg-indigo-600 text-white shadow-md'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'
          }`}
        >
          {zone}
        </button>
      ))}
    </div>
  );
}
