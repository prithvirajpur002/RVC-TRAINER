import { useState, useEffect, useCallback } from 'react';
import { Play, CheckCircle, AlertCircle, TrendingUp, Trophy, LogIn } from 'lucide-react';
import { supabase } from '../lib/supabase';

interface ExperimentRun {
  id: string;
  exp_id: string;
  dataset: string;
  config: string;
  epochs: number;
  batch_size: number;
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

export default function ExperimentRunner() {
  const [runs, setRuns] = useState<ExperimentRun[]>([]);
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [loading, setLoading] = useState(true);
  const [authed, setAuthed] = useState<boolean | null>(null);

  const [newExpId, setNewExpId] = useState('exp_001');
  const [newDataset, setNewDataset] = useState('clean');
  const [newConfig, setNewConfig] = useState('baseline');
  const [newEpochs, setNewEpochs] = useState(200);
  const [newBatchSize, setNewBatchSize] = useState(6);
  const [submitting, setSubmitting] = useState(false);

  const [decWinner, setDecWinner] = useState('');
  const [decLoser, setDecLoser] = useState('');
  const [decReason, setDecReason] = useState('');
  const [decNext, setDecNext] = useState('');
  const [decRationale, setDecRationale] = useState('');
  const [decSubmitting, setDecSubmitting] = useState(false);

  const [error, setError] = useState('');
  const [successMsg, setSuccessMsg] = useState('');

  // ── Auth check ────────────────────────────────────────────────────────────
  useEffect(() => {
    supabase.auth.getUser().then(({ data }) => {
      setAuthed(!!data.user);
    });
    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      setAuthed(!!session?.user);
    });
    return () => listener.subscription.unsubscribe();
  }, []);

  // ── Data loading ──────────────────────────────────────────────────────────
  const loadData = useCallback(async () => {
    try {
      const [runsRes, decisionsRes] = await Promise.all([
        supabase.from('experiment_runs').select('*').order('created_at', { ascending: false }),
        supabase.from('decisions').select('*').order('created_at', { ascending: false }),
      ]);
      if (runsRes.error) throw runsRes.error;
      if (decisionsRes.error) throw decisionsRes.error;
      setRuns(runsRes.data || []);
      setDecisions(decisionsRes.data || []);
    } catch (err) {
      console.error('Failed to load data:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  // ── Adaptive polling: 10 s when a run is active, 30 s otherwise ──────────
  useEffect(() => {
    if (!authed) return;
    loadData();
    const getInterval = () =>
      runs.some((r) => r.status === 'running') ? 10_000 : 30_000;
    let timer = setInterval(loadData, getInterval());
    return () => clearInterval(timer);
  }, [authed, loadData, runs]);

  // ── Helpers ───────────────────────────────────────────────────────────────
  function advanceExpId(current: string) {
    const parts = current.split('_');
    const num = parseInt(parts[1], 10);
    // Fix: guard against NaN so auto-increment never produces "exp_NaN"
    if (!isNaN(num)) {
      setNewExpId(`exp_${(num + 1).toString().padStart(3, '0')}`);
    }
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'complete': return 'bg-green-100 text-green-800 border-green-300';
      case 'running':  return 'bg-blue-100 text-blue-800 border-blue-300';
      case 'failed':   return 'bg-red-100 text-red-800 border-red-300';
      default:         return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  }

  function getCompositeScore(scores?: Record<string, number>): number {
    if (!scores) return 0;
    return (
      (scores.naturalness || 0) * 0.45 +
      (scores.clarity     || 0) * 0.35 +
      (scores.identity    || 0) * 0.20
    );
  }

  // ── Create experiment ─────────────────────────────────────────────────────
  async function createExperiment() {
    setError('');
    setSuccessMsg('');

    if (!newExpId.match(/^exp_\d+$/)) {
      setError('Experiment ID must be in the format exp_001, exp_002, etc.');
      return;
    }
    if (runs.some((r) => r.exp_id === newExpId)) {
      setError(`Experiment ${newExpId} already exists — choose a different ID.`);
      return;
    }

    setSubmitting(true);
    try {
      const { error: insertError } = await supabase.from('experiment_runs').insert({
        exp_id: newExpId,
        dataset: newDataset,
        config: newConfig,
        epochs: newEpochs,
        batch_size: newBatchSize,
        status: 'pending',
      });
      if (insertError) throw insertError;
      setSuccessMsg(`Created ${newExpId}. Run training with: python rdp/main.py --only ${newExpId}`);
      advanceExpId(newExpId);
      await loadData();
    } catch (err) {
      setError(`Failed to create experiment: ${err}`);
    } finally {
      setSubmitting(false);
    }
  }

  // ── Record decision ───────────────────────────────────────────────────────
  async function recordDecision() {
    setError('');
    setSuccessMsg('');

    if (!decWinner.trim()) {
      setError('Winner experiment ID is required.');
      return;
    }
    if (!decReason.trim()) {
      setError('Reason summary is required.');
      return;
    }

    setDecSubmitting(true);
    try {
      const { error: insertError } = await supabase.from('decisions').insert({
        winner_exp_id:        decWinner.trim(),
        loser_exp_id:         decLoser.trim() || null,
        reason_summary:       decReason.trim(),
        next_planned_exp_id:  decNext.trim()  || null,
        rationale:            decRationale.trim() || null,
      });
      if (insertError) throw insertError;
      setSuccessMsg(`Decision recorded — winner: ${decWinner}`);
      setDecWinner('');
      setDecLoser('');
      setDecReason('');
      setDecNext('');
      setDecRationale('');
      await loadData();
    } catch (err) {
      setError(`Failed to record decision: ${err}`);
    } finally {
      setDecSubmitting(false);
    }
  }

  // ── Not authenticated ─────────────────────────────────────────────────────
  if (authed === false) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center p-8">
        <div className="bg-white rounded-lg border border-slate-200 p-10 max-w-sm w-full text-center">
          <LogIn className="mx-auto mb-4 text-slate-400" size={40} />
          <h2 className="text-xl font-semibold text-slate-900 mb-2">Sign in required</h2>
          <p className="text-slate-500 text-sm mb-6">
            You need to be signed in to view and manage experiments.
          </p>
          <button
            onClick={() => supabase.auth.signInWithOAuth({ provider: 'github' })}
            className="w-full bg-slate-900 hover:bg-slate-800 text-white font-semibold py-2 px-4 rounded-lg transition-colors"
          >
            Sign in with GitHub
          </button>
        </div>
      </div>
    );
  }

  // ── Loading auth state ────────────────────────────────────────────────────
  if (authed === null) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center">
        <p className="text-slate-500">Loading...</p>
      </div>
    );
  }

  // ── Main UI ───────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-8">
      <div className="max-w-5xl mx-auto">
        <h1 className="text-4xl font-bold text-slate-900 mb-2">Experiment Control</h1>
        <p className="text-slate-600 mb-8">
          Manual, traceable experiment management. One variable per experiment.
        </p>

        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex gap-3">
            <AlertCircle className="text-red-600 flex-shrink-0 mt-0.5" size={20} />
            <p className="text-red-800">{error}</p>
          </div>
        )}
        {successMsg && (
          <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg flex gap-3">
            <CheckCircle className="text-green-600 flex-shrink-0 mt-0.5" size={20} />
            <p className="text-green-800">{successMsg}</p>
          </div>
        )}

        {/* ── Define experiment ── */}
        <div className="bg-white rounded-lg border border-slate-200 p-6 mb-8">
          <h2 className="text-xl font-semibold text-slate-900 mb-4">Define New Experiment</h2>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Experiment ID</label>
              <input
                type="text"
                value={newExpId}
                onChange={(e) => setNewExpId(e.target.value)}
                placeholder="exp_001"
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-slate-900 focus:outline-none focus:ring-2 focus:ring-slate-400"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Dataset</label>
              <select
                value={newDataset}
                onChange={(e) => setNewDataset(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-slate-900"
              >
                <option value="clean">clean</option>
                <option value="natural">natural</option>
                <option value="raw">raw</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Config</label>
              <select
                value={newConfig}
                onChange={(e) => setNewConfig(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-slate-900"
              >
                <option value="baseline">baseline</option>
                <option value="high_quality">high_quality</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Epochs</label>
              <input
                type="number"
                value={newEpochs}
                onChange={(e) => setNewEpochs(parseInt(e.target.value, 10))}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-slate-900"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Batch Size</label>
              <input
                type="number"
                value={newBatchSize}
                onChange={(e) => setNewBatchSize(parseInt(e.target.value, 10))}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-slate-900"
              />
            </div>
          </div>
          <button
            onClick={createExperiment}
            disabled={submitting}
            className="w-full bg-slate-900 hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-2 px-4 rounded-lg flex items-center justify-center gap-2 transition-colors"
          >
            {submitting ? (
              <span className="animate-spin inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full" />
            ) : (
              <Play size={16} />
            )}
            {submitting ? 'Creating…' : 'Create Experiment'}
          </button>
          <p className="mt-2 text-xs text-slate-400 text-center">
            After creating, start training: <code className="bg-slate-100 px-1 rounded">python rdp/main.py --only {newExpId}</code>
          </p>
        </div>

        {/* ── Experiment runs ── */}
        <div className="bg-white rounded-lg border border-slate-200 p-6 mb-8">
          <h2 className="text-xl font-semibold text-slate-900 mb-4 flex items-center gap-2">
            <TrendingUp size={20} />
            Experiment Runs
          </h2>
          {loading ? (
            <p className="text-slate-500">Loading…</p>
          ) : runs.length === 0 ? (
            <p className="text-slate-500">No experiments yet.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 border-b border-slate-200">
                  <tr>
                    <th className="px-4 py-2 text-left font-semibold text-slate-900">ID</th>
                    <th className="px-4 py-2 text-left font-semibold text-slate-900">Dataset</th>
                    <th className="px-4 py-2 text-left font-semibold text-slate-900">Config</th>
                    <th className="px-4 py-2 text-left font-semibold text-slate-900">Params</th>
                    <th className="px-4 py-2 text-left font-semibold text-slate-900">Status</th>
                    <th className="px-4 py-2 text-right font-semibold text-slate-900">Natural</th>
                    <th className="px-4 py-2 text-right font-semibold text-slate-900">Clarity</th>
                    <th className="px-4 py-2 text-right font-semibold text-slate-900">Identity</th>
                    <th className="px-4 py-2 text-right font-semibold text-slate-900">Composite</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((run) => (
                    <tr key={run.id} className="border-b border-slate-100 hover:bg-slate-50">
                      <td className="px-4 py-2 font-mono text-slate-900">{run.exp_id}</td>
                      <td className="px-4 py-2 text-slate-600">{run.dataset}</td>
                      <td className="px-4 py-2 text-slate-600">{run.config}</td>
                      <td className="px-4 py-2 text-xs text-slate-600">{run.epochs}e, b{run.batch_size}</td>
                      <td className="px-4 py-2">
                        <span className={`px-2 py-1 text-xs font-semibold rounded border ${getStatusColor(run.status)}`}>
                          {run.status}
                        </span>
                      </td>
                      {/* Individual scores — only shown when complete */}
                      <td className="px-4 py-2 text-right text-slate-700">
                        {run.scores && run.status === 'complete'
                          ? (run.scores.naturalness ?? 0).toFixed(3) : '—'}
                      </td>
                      <td className="px-4 py-2 text-right text-slate-700">
                        {run.scores && run.status === 'complete'
                          ? (run.scores.clarity ?? 0).toFixed(3) : '—'}
                      </td>
                      <td className="px-4 py-2 text-right text-slate-700">
                        {run.scores && run.status === 'complete'
                          ? (run.scores.identity ?? 0).toFixed(3) : '—'}
                      </td>
                      <td className="px-4 py-2 text-right font-bold text-slate-900">
                        {run.scores && run.status === 'complete'
                          ? getCompositeScore(run.scores).toFixed(3) : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* ── Record decision ── */}
        <div className="bg-white rounded-lg border border-slate-200 p-6 mb-8">
          <h2 className="text-xl font-semibold text-slate-900 mb-1 flex items-center gap-2">
            <Trophy size={20} />
            Record Decision
          </h2>
          <p className="text-sm text-slate-500 mb-4">
            After listening to experiments, record which one won and why. This is your iteration memory.
          </p>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Winner <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                value={decWinner}
                onChange={(e) => setDecWinner(e.target.value)}
                placeholder="exp_002"
                list="exp-ids"
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-slate-900 focus:outline-none focus:ring-2 focus:ring-slate-400"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Loser (optional)</label>
              <input
                type="text"
                value={decLoser}
                onChange={(e) => setDecLoser(e.target.value)}
                placeholder="exp_001"
                list="exp-ids"
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-slate-900 focus:outline-none focus:ring-2 focus:ring-slate-400"
              />
            </div>
            <div className="col-span-2">
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Reason <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                value={decReason}
                onChange={(e) => setDecReason(e.target.value)}
                placeholder="e.g. Natural dataset sounded more authentic, less robotic"
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-slate-900 focus:outline-none focus:ring-2 focus:ring-slate-400"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Next planned experiment</label>
              <input
                type="text"
                value={decNext}
                onChange={(e) => setDecNext(e.target.value)}
                placeholder="exp_003"
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-slate-900 focus:outline-none focus:ring-2 focus:ring-slate-400"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Rationale for next step</label>
              <input
                type="text"
                value={decRationale}
                onChange={(e) => setDecRationale(e.target.value)}
                placeholder="e.g. Try high_quality config with natural dataset"
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-slate-900 focus:outline-none focus:ring-2 focus:ring-slate-400"
              />
            </div>
          </div>
          <button
            onClick={recordDecision}
            disabled={decSubmitting}
            className="w-full bg-emerald-700 hover:bg-emerald-600 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-2 px-4 rounded-lg flex items-center justify-center gap-2 transition-colors"
          >
            {decSubmitting ? (
              <span className="animate-spin inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full" />
            ) : (
              <CheckCircle size={16} />
            )}
            {decSubmitting ? 'Saving…' : 'Record Decision'}
          </button>
          {/* Datalist for autocomplete */}
          <datalist id="exp-ids">
            {runs.map((r) => <option key={r.id} value={r.exp_id} />)}
          </datalist>
        </div>

        {/* ── Decision log ── */}
        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h2 className="text-xl font-semibold text-slate-900 mb-4">Decision Log</h2>
          {loading ? (
            <p className="text-slate-500">Loading…</p>
          ) : decisions.length === 0 ? (
            <p className="text-slate-500">No decisions recorded yet. Compare experiments above and record the winner.</p>
          ) : (
            <div className="space-y-3">
              {decisions.map((decision) => (
                <div key={decision.id} className="p-3 bg-slate-50 rounded border border-slate-200">
                  <div className="flex justify-between items-start mb-1">
                    <span className="font-semibold text-slate-900">
                      Winner: <span className="font-mono text-emerald-700">{decision.winner_exp_id}</span>
                      {decision.loser_exp_id && (
                        <span className="text-slate-400 font-normal"> vs <span className="font-mono">{decision.loser_exp_id}</span></span>
                      )}
                    </span>
                    <span className="text-xs text-slate-400 whitespace-nowrap ml-4">
                      {new Date(decision.created_at).toLocaleDateString()}
                    </span>
                  </div>
                  <p className="text-sm text-slate-700 mb-1">{decision.reason_summary}</p>
                  {decision.next_planned_exp_id && (
                    <p className="text-xs text-slate-500">
                      Next: <span className="font-mono">{decision.next_planned_exp_id}</span>
                      {decision.rationale && <> — {decision.rationale}</>}
                    </p>
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
