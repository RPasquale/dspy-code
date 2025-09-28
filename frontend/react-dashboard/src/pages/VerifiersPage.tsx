import { FormEvent, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import Card from '../components/Card';
import { api } from '../api/client';
import type { VerifierSummary } from '../api/types';

const VerifiersPage = () => {
  const queryClient = useQueryClient();
  const { data } = useQuery({ queryKey: ['verifiers'], queryFn: api.getVerifiers, refetchInterval: 20000 });
  const [newVerifierName, setNewVerifierName] = useState('');
  const [newVerifierTool, setNewVerifierTool] = useState('run_tests');
  const [newVerifierStatus, setNewVerifierStatus] = useState('active');
  const [newVerifierDescription, setNewVerifierDescription] = useState('');

  const update = useMutation({
    mutationFn: api.updateVerifier,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['verifiers'] })
  });

  const createVerifier = useMutation({
    mutationFn: (payload: { name: string; description?: string; tool?: string; status?: string }) => api.createVerifier(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['verifiers'] });
      setNewVerifierName('');
      setNewVerifierDescription('');
    }
  });

  const deleteVerifier = useMutation({
    mutationFn: (name: string) => api.deleteVerifier(name),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['verifiers'] })
  });

  const handleCreate = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const name = newVerifierName.trim();
    if (!name || createVerifier.isPending) return;
    createVerifier.mutate({
      name,
      tool: newVerifierTool,
      status: newVerifierStatus,
      description: newVerifierDescription.trim() || undefined
    });
  };
  const verifiers = (data?.verifiers ?? []) as VerifierSummary[];
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900 mb-2">Verifiers</h1>
        <p className="text-slate-600">Monitor and manage DSPy verifier performance and status</p>
      </div>
      
      <Card title="All Verifiers" subtitle={`Active: ${data?.total_active ?? 0} | Avg Accuracy: ${(data?.avg_accuracy ?? 0).toFixed(1)}%`}>
        <form onSubmit={handleCreate} className="grid grid-cols-1 md:grid-cols-5 gap-3 mb-4">
          <input
            className="rounded-md border border-slate-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-blue-500"
            placeholder="Verifier name"
            value={newVerifierName}
            onChange={(event) => setNewVerifierName(event.target.value)}
            required
          />
          <input
            className="rounded-md border border-slate-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-blue-500"
            placeholder="Description"
            value={newVerifierDescription}
            onChange={(event) => setNewVerifierDescription(event.target.value)}
          />
          <select
            className="rounded-md border border-slate-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-blue-500"
            value={newVerifierTool}
            onChange={(event) => setNewVerifierTool(event.target.value)}
          >
            {['run_tests', 'lint', 'build', 'patch'].map((tool) => (
              <option key={tool} value={tool}>{tool}</option>
            ))}
          </select>
          <select
            className="rounded-md border border-slate-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-blue-500"
            value={newVerifierStatus}
            onChange={(event) => setNewVerifierStatus(event.target.value)}
          >
            {['active', 'paused', 'disabled'].map((option) => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
          <button
            type="submit"
            disabled={createVerifier.isPending || !newVerifierName.trim()}
            className="inline-flex items-center justify-center rounded-md bg-slate-900 px-3 py-2 text-sm font-medium text-white shadow-sm hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {createVerifier.isPending ? 'Adding…' : 'Add Verifier'}
          </button>
        </form>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-slate-200">
            <thead className="bg-slate-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Name</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Accuracy</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Status</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Checks</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Issues</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Avg Time</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider"></th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-slate-200">
              {verifiers.map((v) => (
                <tr key={v.name} className="hover:bg-slate-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-slate-900">{v.name}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">{v.accuracy.toFixed(1)}%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">
                    <select 
                      className="block w-full rounded-md border-slate-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm" 
                      value={v.status} 
                      onChange={(e) => update.mutate({ name: v.name, status: e.target.value })}
                    >
                      {['active', 'paused', 'disabled'].map((s) => (
                        <option key={s} value={s}>{s}</option>
                      ))}
                    </select>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">{v.checks_performed}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">{v.issues_found}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">{v.avg_execution_time.toFixed(2)}s</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">
                    <button
                      type="button"
                      className="inline-flex items-center px-3 py-1 border border-red-200 shadow-sm text-sm leading-4 font-medium rounded-md text-red-700 bg-white hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 disabled:cursor-not-allowed disabled:opacity-60"
                      onClick={() => { if (!deleteVerifier.isPending && window.confirm(`Remove ${v.name}?`)) { deleteVerifier.mutate(v.name); } }}
                      disabled={deleteVerifier.isPending && deleteVerifier.variables === v.name}
                    >
                      {deleteVerifier.isPending && deleteVerifier.variables === v.name ? 'Deleting…' : 'Delete'}
                    </button>
                  </td>
                </tr>
              ))}
              {verifiers.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-6 py-4 text-center text-sm text-slate-500">No verifiers found</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
};

export default VerifiersPage;
