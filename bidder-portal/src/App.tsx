import { useState } from 'react';
import { Header } from './components/Header';
import { QuarterTable } from './components/QuarterTable';
import { CountdownTimer } from './components/CountdownTimer';
import { LastUpdated } from './components/LastUpdated';
import { usePredictions } from './hooks/usePredictions';

function App() {
  const [priceArea, setPriceArea] = useState<'DK1' | 'DK2'>('DK1');
  const { predictions, loading, error, lastUpdated, refetch } = usePredictions(priceArea);

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col items-center">
      <div className="w-full max-w-7xl px-4 py-8 flex flex-col gap-8">
        <Header priceArea={priceArea} onPriceAreaChange={setPriceArea} />
        
        <div className="flex flex-col md:flex-row justify-between items-center gap-4 bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-lg">
          <CountdownTimer nextQuarter={predictions[0]} onExpire={refetch} />
          <LastUpdated timestamp={lastUpdated} />
        </div>

        <QuarterTable predictions={predictions} loading={loading} error={error} />
      </div>
    </div>
  );
}

export default App;
