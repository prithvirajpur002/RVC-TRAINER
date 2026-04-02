import { useState, useEffect } from 'react';
import {
  Play, CheckCircle, AlertCircle, TrendingUp,
  BookOpen, ChevronRight, Info, Star,
} from 'lucide-react';
import { supabase } from '../lib/supabase';

// ── Types ─────────────────────────────────────────────────────────────────────

interface ExperimentRun {
  id: string;
  exp_id: string;
  dataset: string;
  config: string;
  epochs: number;
  batch_size: number;
  changed_from?: string;
  change_note?: string;
  change_variable?: string;
  rvc_commit?: string;
  status: 'pending' | 'running' | 'complete' | 'failed';
  scores?: Record<string, number>;
  created_at: string;
  completed_at?: string;
}

interface Decision {
  id: string;
  winner_exp_id: string;
  loser_exp_id?: string;
  reason_summary: string;
  next_planned_exp_id?: string;
  rationale?: string;
  created_at: string;
}

// ── Constants ─────────────────────────────────────────────────────────────────

const DATASETS = ['clean', 'natural', 'raw'] as const;
const CONFIGS  = ['baseline', 'high_quality'] as const;

const DATASET_COLORS: Record<string, string> = {
  clean:   'bg-sky-100 text-sky-800 border-sky-300',
  natural: 'bg-emerald-100 text-emerald-800 border-emerald-300',
  raw:     'bg-amber-100 text-amber-800 border-amber-300',
};

// ── Helper functions ──────────────────────────────────────────────────────────

function getCompositeScore(scores?: Record<string, number>): number | null {
  if (!scores) return null;
  // Weights: naturalness 0.35 + clarity 0.30 + identity (pitch_corr) 0.35
  // identity = pitch correlation — the real voice identity metric
  return (
    (scores.naturalness || 0) * 0.35 +
    (scores.clarity     || 0) * 0.30 +
    (scores.identity    || 0) * 0.35
  );
}

function getStatusBadge(status: string): string {
  switch (status) {
    case 'complete': return 'bg-green-100 text-green-800 border-green-300';
    case 'running':  return 'bg-blue-100 text-blue-800 border-blue-300';
    case 'failed':   return 'bg-red-100 text-red-800 border-red-300';
    default:         return 'bg-gray-100 text-gray-700 border-gray-300';
  }
}

function nextExpId(runs: ExperimentRun[]): string {
  if (runs.length === 0) return 'exp_001';
  const nums = runs
    .map(r => parseInt(r.exp_id.split('_')[1] || '0'))
    .filter(n => !isNaN(n));
  const max = Math.max(0, ...nums);
  return `exp_${String(max + 1).padStart(3, '0')}`;
}

// ── Dataset Coverage widget ───────────────────────────────────────────────────

function DatasetCoverage({ runs }: { runs: ExperimentRun[] }) {
  const completed = runs.filter(r => r.status === 'complete');
  const covered   = new Set(completed.map(r => r.dataset));

  return (
    <div className="bg-white rounded-lg border border-slate-200 p-5 mb-6">
      <h2 className="text-sm font-semibold text-slate-700 uppercase tracking-wider mb-3">
        Dataset Coverage
      </h2>
      <p className="text-xs text-slate-500 mb-3">
        Best practice: test all 3 dataset types before drawing conclusions.
        Untested = gap in your comparison.
      </p>
      <div className="flex gap-3">
        {DATASETS.map(d => (
          <div
            key={d}
            className={`flex-1 rounded-lg border p-3 text-center ${
              covered.has(d)
                ? DATASET_COLORS[d]
                : 'bg-slate-50 text-slate-400 border-slate-200'
            }`}
          >
            <div className="text-lg">{covered.has(d) ? '✅' : '⬜'}</div>
            <div className="text-xs font-semibold mt-1">{d}</div>
            <div className="text-xs mt-0.5">
              {covered.has(d)
                ? `${completed.filter(r => r.dataset === d).length} run(s)`
                : 'not tested'}
            </div>
          </div>
        ))}
      </div>
      {covered.size < 3 && (
        <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded p-2 mt-3">
          ⚠️ You haven't tested all 3 datasets yet. Compare them before drawing
          conclusions about your data.
        </p>
      )}
    </div>
  );
}

// ── Iteration Status widget ───────────────────────────────────────────────────

type LoopStep = 'define' | 'run' | 'evaluate' | 'decide' | 'next';

function IterationStatus({
  runs,
  decisions,
}: {
  runs: ExperimentRun[];
  decisions: Decision[];
}) {
  const latest       = runs[0];
  const latestStatus = latest?.status;
  const decidedIds   = new Set(
    decisions.flatMap(d => [d.winner_exp_id, d.loser_exp_id].filter(Boolean))
  );

  let step: LoopStep = 'define';
  let message        = 'No experiments yet — define your first one below.';

  if (latest) {
    if (latestStatus === 'pending') {
      step    = 'run';
      message = `${latest.exp_id} is pending. Click Run (on Kaggle) to start training.`;
    } else if (latestStatus === 'running') {
      step    = 'evaluate';
      message = `${latest.exp_id} is training. Wait for it to complete.`;
    } else if (latestStatus === 'complete' && !decidedIds.has(latest.exp_id)) {
      step    = 'decide';
      message = `${latest.exp_id} is done. Record your decision below before creating the next experiment.`;
    } else {
      step    = 'next';
      message = `Last decision recorded. Define the next experiment.`;
    }
  }

  const steps: { id: LoopStep; label: string }[] = [
    { id: 'define',   label: 'Define' },
    { id: 'run',      label: 'Run' },
    { id: 'evaluate', label: 'Evaluate' },
    { id: 'decide',   label: 'Decide' },
    { id: 'next',     label: 'Next' },
  ];

  const stepOrder = steps.map(s => s.id);
  const current   = stepOrder.indexOf(step);

  return (
    <div className="bg-white rounded-lg border border-slate-200 p-5 mb-6">
      <h2 className="text-sm font-semibold text-slate-700 uppercase tracking-wider mb-3">
        Iteration Loop
      </h2>

      <div className="flex items-center gap-1 mb-4">
        {steps.map((s, i) => (
          <div key={s.id} className="flex items-center gap-1">
            <div
              className={`px-2.5 py-1 rounded text-xs font-semibold ${
                i === current
                  ? 'bg-slate-900 text-white'
                  : i < current
                  ? 'bg-green-100 text-green-800 border border-green-300'
                  : 'bg-slate-100 text-slate-400'
              }`}
            >
              {s.label}
            </div>
            {i < steps.length - 1 && (
              <ChevronRight size={12} className="text-slate-300 flex-shrink-0" />
            )}
          </div>
        ))}
      </div>

      <p
        className={`text-sm rounded p-2 ${
          step === 'decide'
            ? 'bg-amber-50 border border-amber-200 text-amber-800'
            : 'bg-slate-50 text-slate-700'
        }`}
      >
        {message}
      </p>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function ExperimentRunner() {
  const [runs,      setRuns]      = useState<ExperimentRun[]>([]);
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [loading,   setLoading]   = useState(true);

  // New experiment form
  const [newDataset,     setNewDataset]     = useState<string>('clean');
  const [newConfig,      setNewConfig]      = useState<string>('baseline');
  const [newEpochs,      setNewEpochs]      = useState<number>(200);
  const [newBatchSize,   setNewBatchSize]   = useState<number>(6);
  const [newChangedFrom, setNewChangedFrom] = useState<string>('');
  const [newChangeNote,  setNewChangeNote]  = useState<string>('');
  const [newRvcCommit,   setNewRvcCommit]   = useState<string>('');

  // Decision form
  const [decWinner,   setDecWinner]   = useState('');
  const [decLoser,    setDecLoser]    = useState('');
  const [decReason,   setDecReason]   = useState('');
  const [decNextNote, setDecNextNote] = useState('');

  const [error,      setError]      = useState('');
  const [successMsg, setSuccessMsg] = useState('');

  useEffect(() => {
    loadData();
    const iv = setInterval(loadData, 30000); // 30s — training runs take hours
    return () => clearInterval(iv);
  }, []);

  async function loadData() {
    try {
      const [runsRes, decRes] = await Promise.all([
        supabase.from('experiment_runs').select('*').order('created_at', { ascending: false }),
        supabase.from('decisions').select('*').order('created_at', { ascending: false }),
      ]);

      if (runsRes.error) throw runsRes.error;
      if (decRes.error)  throw decRes.error;

      setRuns(runsRes.data || []);
      setDecisions(decRes.data || []);
    } catch (err) {
      console.error('Failed to load data:', err);
    } finally {
      setLoading(false);
    }
  }

  // ── Derive new experiment ID ────────────────────────────────────────────────
  const derivedExpId = nextExpId(runs);
  const isFirstExp   = runs.length === 0;

  // ── Check if a decision is required before creating next experiment ─────────
  const decidedIds     = new Set(
    decisions.flatMap(d => [d.winner_exp_id, d.loser_exp_id].filter(Boolean))
  );
  const latestComplete = runs.find(r => r.status === 'complete');
  const decisionNeeded =
    !isFirstExp &&
    latestComplete &&
    !decidedIds.has(latestComplete.exp_id);

  // ── Create experiment ───────────────────────────────────────────────────────
  async function createExperiment() {
    setError('');
    setSuccessMsg('');

    if (decisionNeeded) {
      setError(
        `Record a decision for ${latestComplete!.exp_id} before creating a new experiment.`
      );
      return;
    }

    if (!isFirstExp && !newChangedFrom) {
      setError('Select which experiment this builds on (Changed From).');
      return;
    }

    if (!isFirstExp && !newChangeNote.trim()) {
      setError('Describe what you changed and why (Change Note).');
      return;
    }

    if (runs.some(r => r.exp_id === derivedExpId)) {
      setError(`Experiment ${derivedExpId} already exists.`);
      return;
    }

    // Validate single-change rule
    if (!isFirstExp && newChangedFrom) {
      const prevRun = runs.find(r => r.exp_id === newChangedFrom);
      if (prevRun) {
        const diffs = (
          ['dataset', 'config', 'epochs', 'batch_size'] as const
        ).filter(k => {
          const prev = (prevRun as Record<string, unknown>)[k];
          const curr = { dataset: newDataset, config: newConfig, epochs: newEpochs, batch_size: newBatchSize }[k];
          return prev !== curr;
        });

        if (diffs.length === 0) {
          setError(`Identical to ${newChangedFrom}. Change exactly ONE variable.`);
          return;
        }
        if (diffs.length > 1) {
          setError(
            `Too many changes (${diffs.join(', ')}). Change exactly ONE variable from ${newChangedFrom}.`
          );
          return;
        }
      }
    }

    try {
      const { error: insertError } = await supabase.from('experiment_runs').insert({
        exp_id:          derivedExpId,
        dataset:         newDataset,
        config:          newConfig,
        epochs:          newEpochs,
        batch_size:      newBatchSize,
        changed_from:    newChangedFrom || null,
        change_note:     newChangeNote  || null,
        rvc_commit:      newRvcCommit   || null,
        change_variable: (() => {
          if (!newChangedFrom) return null;
          const prev = runs.find(r => r.exp_id === newChangedFrom);
          if (!prev) return null;
          const kv: Record<string, unknown> = { dataset: newDataset, config: newConfig, epochs: newEpochs, batch_size: newBatchSize };
          const diffs = (['dataset','config','epochs','batch_size'] as const)
            .filter(k => (prev as Record<string, unknown>)[k] !== kv[k]);
          return diffs[0] || null;
        })(),
        status:          'pending',
      });

      if (insertError) throw insertError;

      setSuccessMsg(
        `${derivedExpId} created. Now run it on Kaggle:\n` +
        `python experiment_runner.py run ${derivedExpId}`
      );
      setNewChangedFrom('');
      setNewChangeNote('');
      setNewRvcCommit('');
      await loadData();
    } catch (err) {
      setError(`Failed to create: ${err}`);
    }
  }

  // ── Record decision ─────────────────────────────────────────────────────────
  async function recordDecision() {
    setError('');
    setSuccessMsg('');

    if (!decWinner) {
      setError('Select a winner.');
      return;
    }
    if (!decReason.trim()) {
      setError('Reason cannot be empty. What did you learn?');
      return;
    }

    try {
      const { error } = await supabase.from('decisions').insert({
        winner_exp_id:      decWinner,
        loser_exp_id:       decLoser || null,
        reason_summary:     decReason,
        next_planned_exp_id: null,
        rationale:          decNextNote || null,
      });

      if (error) throw error;

      setSuccessMsg(`Decision recorded. ${decWinner} is the current best.`);
      setDecWinner('');
      setDecLoser('');
      setDecReason('');
      setDecNextNote('');
      await loadData();
    } catch (err) {
      setError(`Failed to record decision: ${err}`);
    }
  }

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-6">
      <div className="max-w-5xl mx-auto">

        {/* Header */}
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-slate-900">Experiment Control</h1>
          <p className="text-slate-500 text-sm mt-1">
            One variable per experiment · Record every decision · Compare all datasets
          </p>
        </div>

        {/* Alerts */}
        {error && (
          <div className="mb-5 p-4 bg-red-50 border border-red-200 rounded-lg flex gap-3">
            <AlertCircle className="text-red-600 flex-shrink-0 mt-0.5" size={18} />
            <p className="text-red-800 text-sm whitespace-pre-line">{error}</p>
          </div>
        )}
        {successMsg && (
          <div className="mb-5 p-4 bg-green-50 border border-green-200 rounded-lg flex gap-3">
            <CheckCircle className="text-green-600 flex-shrink-0 mt-0.5" size={18} />
            <p className="text-green-800 text-sm whitespace-pre-line">{successMsg}</p>
          </div>
        )}

        {/* Iteration Status + Dataset Coverage side by side */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-2">
          <IterationStatus runs={runs} decisions={decisions} />
          <DatasetCoverage runs={runs} />
        </div>

        {/* Decision required banner */}
        {decisionNeeded && (
          <div className="mb-6 p-4 bg-amber-50 border-2 border-amber-400 rounded-lg">
            <p className="font-semibold text-amber-900">
              🛑 Decision required before creating next experiment
            </p>
            <p className="text-sm text-amber-800 mt-1">
              {latestComplete!.exp_id} finished. Scroll down to record your
              decision — which model won and why — then come back here.
            </p>
          </div>
        )}

        {/* ── Create Experiment ─────────────────────────────────────────────── */}
        <div className="bg-white rounded-lg border border-slate-200 p-6 mb-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-semibold text-slate-900">
              Define Experiment{' '}
              <span className="font-mono text-slate-500 text-sm">{derivedExpId}</span>
            </h2>
            {decisionNeeded && (
              <span className="text-xs font-semibold text-red-700 bg-red-50 border border-red-200 px-2 py-1 rounded">
                Blocked — record decision first
              </span>
            )}
          </div>

          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-xs font-semibold text-slate-600 mb-1 uppercase tracking-wider">
                Dataset
              </label>
              <select
                value={newDataset}
                onChange={e => setNewDataset(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm text-slate-900"
              >
                {DATASETS.map(d => (
                  <option key={d} value={d}>{d}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-600 mb-1 uppercase tracking-wider">
                Config
              </label>
              <select
                value={newConfig}
                onChange={e => setNewConfig(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm text-slate-900"
              >
                {CONFIGS.map(c => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-600 mb-1 uppercase tracking-wider">
                Epochs
              </label>
              <input
                type="number"
                value={newEpochs}
                onChange={e => setNewEpochs(parseInt(e.target.value) || 200)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm text-slate-900"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-600 mb-1 uppercase tracking-wider">
                Batch Size
              </label>
              <input
                type="number"
                value={newBatchSize}
                onChange={e => setNewBatchSize(parseInt(e.target.value) || 6)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm text-slate-900"
              />
            </div>

            <div className="col-span-2">
              <label className="block text-xs font-semibold text-slate-600 mb-1 uppercase tracking-wider">
                RVC Commit{' '}
                <span className="font-normal text-slate-400 normal-case tracking-normal">
                  (optional — auto-detected at run time if blank)
                </span>
              </label>
              <input
                type="text"
                value={newRvcCommit}
                onChange={e => setNewRvcCommit(e.target.value)}
                placeholder="e.g. abc123def456  →  run: git -C /path/to/rvc rev-parse HEAD"
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm text-slate-900 font-mono"
              />
              <p className="text-xs text-slate-400 mt-1">
                Ensures exp_001 and exp_003 are on the same RVC version — if the repo
                updates between runs, scores aren't comparable.
              </p>
            </div>
          </div>

          {/* Changed From — only shown for exp_002+ */}
          {!isFirstExp && (
            <div className="mt-2 pt-4 border-t border-slate-100 grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-semibold text-slate-600 mb-1 uppercase tracking-wider">
                  Changed From <span className="text-red-500">*</span>
                </label>
                <select
                  value={newChangedFrom}
                  onChange={e => setNewChangedFrom(e.target.value)}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm text-slate-900"
                >
                  <option value="">— select parent experiment —</option>
                  {runs.map(r => (
                    <option key={r.exp_id} value={r.exp_id}>
                      {r.exp_id} ({r.dataset}, {r.config}, {r.epochs}e)
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-600 mb-1 uppercase tracking-wider">
                  Change Note <span className="text-red-500">*</span>
                </label>
                <input
                  type="text"
                  value={newChangeNote}
                  onChange={e => setNewChangeNote(e.target.value)}
                  placeholder="e.g. testing natural dataset instead of clean"
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm text-slate-900"
                />
              </div>

              {/* Show what will change */}
              {newChangedFrom && (() => {
                const prev = runs.find(r => r.exp_id === newChangedFrom);
                if (!prev) return null;
                const kv: Record<string, unknown> = { dataset: newDataset, config: newConfig, epochs: newEpochs, batch_size: newBatchSize };
                const diffs = (['dataset','config','epochs','batch_size'] as const)
                  .filter(k => (prev as Record<string, unknown>)[k] !== kv[k])
                  .map(k => `${k}: ${(prev as Record<string,unknown>)[k]} → ${kv[k]}`);

                return (
                  <div className="col-span-2">
                    <div className={`text-xs rounded p-2 ${
                      diffs.length === 1
                        ? 'bg-green-50 border border-green-200 text-green-800'
                        : diffs.length === 0
                        ? 'bg-amber-50 border border-amber-200 text-amber-800'
                        : 'bg-red-50 border border-red-200 text-red-800'
                    }`}>
                      {diffs.length === 1
                        ? `✅ Single change: ${diffs[0]}`
                        : diffs.length === 0
                        ? '⚠️ No changes from parent — change exactly one variable'
                        : `❌ ${diffs.length} changes detected: ${diffs.join(', ')} — change only ONE`}
                    </div>
                  </div>
                );
              })()}
            </div>
          )}

          <button
            onClick={createExperiment}
            disabled={!!decisionNeeded}
            className={`mt-4 w-full font-semibold py-2.5 px-4 rounded-lg flex items-center justify-center gap-2 transition-colors ${
              decisionNeeded
                ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                : 'bg-slate-900 hover:bg-slate-800 text-white'
            }`}
          >
            <Play size={15} />
            Create {derivedExpId}
          </button>
        </div>

        {/* ── Experiment Runs Table ─────────────────────────────────────────── */}
        <div className="bg-white rounded-lg border border-slate-200 p-6 mb-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
            <TrendingUp size={18} />
            Experiment Runs
          </h2>

          {loading ? (
            <p className="text-slate-500 text-sm">Loading…</p>
          ) : runs.length === 0 ? (
            <p className="text-slate-500 text-sm">No experiments yet.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 border-b border-slate-200">
                  <tr>
                    <th className="px-3 py-2 text-left font-semibold text-slate-700">ID</th>
                    <th className="px-3 py-2 text-left font-semibold text-slate-700">Dataset</th>
                    <th className="px-3 py-2 text-left font-semibold text-slate-700">Config</th>
                    <th className="px-3 py-2 text-left font-semibold text-slate-700">Params</th>
                    <th className="px-3 py-2 text-left font-semibold text-slate-700">Changed from</th>
                    <th className="px-3 py-2 text-left font-semibold text-slate-700">Status</th>
                    <th className="px-3 py-2 text-right font-semibold text-slate-700">Score</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map(run => {
                    const composite = getCompositeScore(run.scores);
                    const isWinner  = decisions.some(d => d.winner_exp_id === run.exp_id);
                    return (
                      <tr key={run.id} className="border-b border-slate-100 hover:bg-slate-50">
                        <td className="px-3 py-2 font-mono font-semibold text-slate-900">
                          {run.exp_id}
                          {isWinner && <Star size={11} className="inline ml-1 text-amber-500" fill="currentColor" />}
                        </td>
                        <td className="px-3 py-2">
                          <span className={`px-2 py-0.5 text-xs font-semibold rounded border ${DATASET_COLORS[run.dataset] || ''}`}>
                            {run.dataset}
                          </span>
                        </td>
                        <td className="px-3 py-2 text-slate-600 text-xs">{run.config}</td>
                        <td className="px-3 py-2 text-xs text-slate-500">
                          <div>{run.epochs}e · b{run.batch_size}</div>
                          {run.rvc_commit && (
                            <div className="font-mono text-slate-300 text-[10px] mt-0.5" title="RVC repo commit">
                              {run.rvc_commit.slice(0, 8)}
                            </div>
                          )}
                        </td>
                        <td className="px-3 py-2 text-xs text-slate-500">
                          {run.changed_from ? (
                            <span>
                              <span className="font-mono">{run.changed_from}</span>
                              {run.change_variable && (
                                <span className="ml-1 text-slate-400">[{run.change_variable}]</span>
                              )}
                              {run.change_note && (
                                <div className="text-slate-400 truncate max-w-[140px]" title={run.change_note}>
                                  {run.change_note}
                                </div>
                              )}
                            </span>
                          ) : (
                            <span className="text-slate-300">baseline</span>
                          )}
                        </td>
                        <td className="px-3 py-2">
                          <span className={`px-2 py-0.5 text-xs font-semibold rounded border ${getStatusBadge(run.status)}`}>
                            {run.status}
                          </span>
                        </td>
                        <td className="px-3 py-2 text-right">
                          {composite !== null && run.status === 'complete' ? (
                            <div>
                              <div className="font-bold text-slate-900">{composite.toFixed(3)}</div>
                              {run.scores?.identity !== undefined && (
                                <div
                                  className="text-[10px] text-slate-400 mt-0.5"
                                  title="Pitch correlation (F0 similarity) — real identity metric"
                                >
                                  pitch {run.scores.identity.toFixed(2)}
                                </div>
                              )}
                            </div>
                          ) : (
                            <span className="text-slate-400">—</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* ── Record Decision ───────────────────────────────────────────────── */}
        <div className={`bg-white rounded-lg border p-6 mb-6 ${
          decisionNeeded ? 'border-amber-400 ring-2 ring-amber-200' : 'border-slate-200'
        }`}>
          <h2 className="text-lg font-semibold text-slate-900 mb-1 flex items-center gap-2">
            <BookOpen size={18} />
            Record Decision
            {decisionNeeded && (
              <span className="text-xs font-bold text-amber-700 bg-amber-100 border border-amber-300 px-2 py-0.5 rounded ml-1">
                REQUIRED NOW
              </span>
            )}
          </h2>
          <p className="text-xs text-slate-500 mb-4">
            After comparing two experiments, record which one won and why.
            This is REQUIRED before creating the next experiment.
          </p>

          <div className="grid grid-cols-2 gap-4 mb-3">
            <div>
              <label className="block text-xs font-semibold text-slate-600 mb-1 uppercase tracking-wider">
                Winner <span className="text-red-500">*</span>
              </label>
              <select
                value={decWinner}
                onChange={e => setDecWinner(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm text-slate-900"
              >
                <option value="">— select winner —</option>
                {runs.filter(r => r.status === 'complete').map(r => (
                  <option key={r.exp_id} value={r.exp_id}>
                    {r.exp_id} ({r.dataset}, {getCompositeScore(r.scores)?.toFixed(3) ?? '?'})
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-600 mb-1 uppercase tracking-wider">
                Loser (optional)
              </label>
              <select
                value={decLoser}
                onChange={e => setDecLoser(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm text-slate-900"
              >
                <option value="">— none —</option>
                {runs.filter(r => r.status === 'complete' && r.exp_id !== decWinner).map(r => (
                  <option key={r.exp_id} value={r.exp_id}>
                    {r.exp_id} ({r.dataset})
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="mb-3">
            <label className="block text-xs font-semibold text-slate-600 mb-1 uppercase tracking-wider">
              Reason — what did you learn? <span className="text-red-500">*</span>
            </label>
            <textarea
              value={decReason}
              onChange={e => setDecReason(e.target.value)}
              placeholder="e.g. Natural dataset produced more human-sounding output on test_1. Clean mode over-compressed the voice."
              rows={2}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm text-slate-900 resize-none"
            />
          </div>

          <div className="mb-4">
            <label className="block text-xs font-semibold text-slate-600 mb-1 uppercase tracking-wider">
              Next experiment rationale (optional)
            </label>
            <input
              type="text"
              value={decNextNote}
              onChange={e => setDecNextNote(e.target.value)}
              placeholder="e.g. Will try high_quality config with natural dataset next"
              className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm text-slate-900"
            />
          </div>

          <button
            onClick={recordDecision}
            className="w-full bg-slate-900 hover:bg-slate-800 text-white font-semibold py-2.5 px-4 rounded-lg flex items-center justify-center gap-2 transition-colors"
          >
            <CheckCircle size={15} />
            Record Decision
          </button>
        </div>

        {/* ── Decision Log ──────────────────────────────────────────────────── */}
        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
            <Info size={18} />
            Decision Log
          </h2>

          {decisions.length === 0 ? (
            <p className="text-slate-500 text-sm">
              No decisions yet. Compare experiments and record winners above.
            </p>
          ) : (
            <div className="space-y-3">
              {decisions.map(d => (
                <div key={d.id} className="p-3 bg-slate-50 rounded border border-slate-200">
                  <div className="flex justify-between items-start mb-1">
                    <span className="text-sm font-semibold text-slate-900">
                      <Star size={12} className="inline text-amber-500 mr-1" fill="currentColor" />
                      {d.winner_exp_id}
                      {d.loser_exp_id && (
                        <span className="font-normal text-slate-500"> beat {d.loser_exp_id}</span>
                      )}
                    </span>
                    <span className="text-xs text-slate-400">
                      {new Date(d.created_at).toLocaleDateString()}
                    </span>
                  </div>
                  <p className="text-sm text-slate-700">{d.reason_summary}</p>
                  {d.rationale && (
                    <p className="text-xs text-slate-500 mt-1 italic">Next: {d.rationale}</p>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
