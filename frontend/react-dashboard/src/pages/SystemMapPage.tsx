import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import Card from '../components/Card';
import StatusPill from '../components/StatusPill';
import { api } from '../api/client';
import type { DataFlow, SystemNode } from '../api/types';
import { useQuery } from '@tanstack/react-query';

const colorForType = (type: string) => {
  switch (type) {
    case 'agent':
      return '#3b82f6';
    case 'llm':
      return '#8b5cf6';
    case 'message_broker':
      return '#f97316';
    case 'compute_master':
      return '#fbbf24';
    case 'compute_worker':
      return '#10b981';
    case 'coordination':
      return '#ec4899';
    case 'ml_training':
      return '#06b6d4';
    default:
      return '#6b7280';
  }
};

const SystemMapPage = () => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [selectedNode, setSelectedNode] = useState<SystemNode | null>(null);
  const topologyQuery = useQuery({
    queryKey: ['system-topology'],
    queryFn: api.getSystemTopology,
    refetchInterval: 30000
  });

  const nodes = topologyQuery.data?.nodes ?? [];
  const flows = topologyQuery.data?.data_flows ?? [];

  useEffect(() => {
    if (!selectedNode && nodes.length > 0) {
      setSelectedNode(nodes[0]);
    }
  }, [nodes, selectedNode]);

  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = svgRef.current.clientWidth || 800;
    const height = 520;
    svg.attr('viewBox', `0 0 ${width} ${height}`);

    const simulation = d3
      .forceSimulation(nodes)
      .force(
        'link',
        d3
          .forceLink<DataFlow, SystemNode>(flows as DataFlow[])
          .id((d: unknown) => (d as SystemNode).id)
          .distance(150)
          .strength(0.4)
      )
      .force('charge', d3.forceManyBody().strength(-380))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(80));

    const link = svg
      .append('g')
      .attr('stroke', 'rgba(148,163,184,0.5)')
      .attr('stroke-width', 1.5)
      .selectAll('line')
      .data(flows)
      .enter()
      .append('line')
      .attr('stroke-dasharray', '4 4');

    const nodeGroup = svg
      .append('g')
      .selectAll('g')
      .data(nodes)
      .enter()
      .append('g')
      .call(
        d3
          .drag<Element, SystemNode>()
          .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on('drag', (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      );

    nodeGroup
      .append('circle')
      .attr('r', 26)
      .attr('fill', (d) => colorForType(d.type))
      .attr('stroke', 'rgba(15,23,42,0.9)')
      .attr('stroke-width', 3)
      .on('click', (_, d) => setSelectedNode(d));

    nodeGroup
      .append('text')
      .attr('x', 0)
      .attr('y', 42)
      .attr('text-anchor', 'middle')
      .attr('fill', 'rgba(15,23,42,0.85)')
      .attr('font-size', 12)
      .text((d) => d.name);

    simulation.on('tick', () => {
      link
        .attr('x1', (d) => (d.source as SystemNode).x ?? 0)
        .attr('y1', (d) => (d.source as SystemNode).y ?? 0)
        .attr('x2', (d) => (d.target as SystemNode).x ?? 0)
        .attr('y2', (d) => (d.target as SystemNode).y ?? 0);

      nodeGroup.attr('transform', (d) => `translate(${d.x ?? 0}, ${d.y ?? 0})`);
    });

    return () => {
      simulation.stop();
    };
  }, [nodes, flows]);

  const clusterInfo = topologyQuery.data?.cluster_info;

  const selectedFlows = useMemo(() => {
    if (!selectedNode) return [];
    return flows.filter((flow) => flow.source === selectedNode.id || flow.target === selectedNode.id);
  }, [flows, selectedNode]);

  return (
    <>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-slate-900 mb-2">System Architecture Map</h1>
              <p className="text-slate-600">Interactive topology exploring container health, resource usage, and streaming connections.</p>
            </div>
            {clusterInfo && (
              <StatusPill status="ok" text={`${clusterInfo.healthy_nodes}/${clusterInfo.total_nodes} healthy`} />
            )}
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Topology Map */}
          <div className="lg:col-span-2">
            <Card title="System Topology" subtitle="Force-directed graph of running components">
              <div className="p-6">
                <svg 
                  ref={svgRef} 
                  className="w-full h-[520px] border border-slate-200 rounded-lg bg-slate-50"
                />
              </div>
            </Card>
          </div>

          {/* Right Column */}
          <div className="space-y-8">
            {/* Cluster Summary */}
            <Card title="Cluster Summary" subtitle="Aggregated metrics from topology">
              {clusterInfo ? (
                <div className="space-y-4">
                  <div className="flex justify-between items-center p-3 bg-slate-50 rounded-lg">
                    <span className="text-sm text-slate-600">Total Nodes</span>
                    <strong className="text-lg">{clusterInfo.total_nodes}</strong>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-slate-50 rounded-lg">
                    <span className="text-sm text-slate-600">Healthy Nodes</span>
                    <strong className="text-lg">{clusterInfo.healthy_nodes}</strong>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-slate-50 rounded-lg">
                    <span className="text-sm text-slate-600">Total CPU Cores</span>
                    <strong className="text-lg">{clusterInfo.total_cpu_cores}</strong>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-slate-50 rounded-lg">
                    <span className="text-sm text-slate-600">Total Memory</span>
                    <strong className="text-lg">{clusterInfo.total_memory_gb} GB</strong>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-slate-50 rounded-lg">
                    <span className="text-sm text-slate-600">Network Throughput</span>
                    <strong className="text-lg">{clusterInfo.network_throughput_mbps} Mbps</strong>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8">
                  <div className="text-slate-400 text-sm">Waiting for cluster metadata…</div>
                </div>
              )}
            </Card>

            {/* Nodes List */}
            <Card title="System Nodes" subtitle="Click on a node in the graph to highlight details">
              <div className="space-y-3 max-h-80 overflow-y-auto">
                {nodes.map((node) => (
                  <button
                    key={node.id}
                    className={`w-full flex items-center gap-3 p-3 rounded-lg border transition-all ${
                      selectedNode?.id === node.id 
                        ? 'border-blue-300 bg-blue-50' 
                        : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50'
                    }`}
                    onClick={() => setSelectedNode(node)}
                    type="button"
                  >
                    <div 
                      className="w-3 h-3 rounded-full" 
                      style={{ backgroundColor: colorForType(node.type) }}
                    />
                    <div className="flex-1 text-left">
                      <div className="font-medium text-slate-900">{node.name}</div>
                      <div className="text-xs text-slate-500 uppercase tracking-wide">{node.type}</div>
                    </div>
                    <StatusPill status={node.status} text={`${(node.cpu_usage || 0).toFixed(0)}% CPU`} />
                  </button>
                ))}
              </div>
            </Card>
          </div>
        </div>

        {/* Connections Section */}
        <div className="mt-8">
          <Card title="Node Connections" subtitle="Throughput and latency for selected node">
            {selectedNode ? (
              <div className="space-y-4">
                <div className="flex justify-between items-center p-4 bg-slate-50 rounded-lg">
                  <div>
                    <h3 className="font-semibold text-slate-900">{selectedNode.name}</h3>
                    <p className="text-sm text-slate-600">{selectedNode.host}:{selectedNode.port || '—'}</p>
                  </div>
                </div>
                <div className="space-y-3">
                  {selectedFlows.map((flow) => (
                    <div key={`${flow.source}-${flow.target}-${flow.type}`} className="flex justify-between items-center p-3 bg-white border border-slate-200 rounded-lg">
                      <div>
                        <div className="font-medium text-slate-900">{flow.type}</div>
                        <div className="text-sm text-slate-600">
                          {flow.source === selectedNode.id ? '→' : '←'}
                          {flow.source === selectedNode.id ? flow.target : flow.source}
                        </div>
                      </div>
                      <div className="text-right text-sm text-slate-600">
                        <div>{flow.throughput} msgs/min</div>
                        <div>{flow.latency.toFixed(2)} ms</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="text-center py-8">
                <div className="text-slate-400 text-sm">Select a node to view its streams and connections.</div>
              </div>
            )}
          </Card>
        </div>
      </div>
    </>
  );
};

export default SystemMapPage;