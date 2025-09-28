import React from 'react';

type Point = { output: number; cost: number };

interface Props {
  points: Point[];
  width?: number;
  height?: number;
}

const ParetoChart: React.FC<Props> = ({ points, width = 360, height = 220 }) => {
  if (!points || points.length === 0) {
    return <div>No Pareto points.</div>;
  }
  const pad = 24;
  const xs = points.map(p => p.cost);
  const ys = points.map(p => p.output);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const sx = (x: number) => pad + ((x - minX) / (maxX - minX || 1)) * (width - 2 * pad);
  const sy = (y: number) => (height - pad) - ((y - minY) / (maxY - minY || 1)) * (height - 2 * pad);
  const [zoom, setZoom] = React.useState(1);
  const [offset, setOffset] = React.useState({ x: 0, y: 0 });
  const dragging = React.useRef<null | { x: number; y: number }>(null);
  const onWheel: React.WheelEventHandler<SVGSVGElement> = (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    setZoom(z => Math.min(5, Math.max(0.5, z * factor)));
  };
  const onMouseDown: React.MouseEventHandler<SVGSVGElement> = (e) => { dragging.current = { x: e.clientX, y: e.clientY }; };
  const onMouseUp: React.MouseEventHandler<SVGSVGElement> = () => { dragging.current = null; };
  const onMouseMove: React.MouseEventHandler<SVGSVGElement> = (e) => {
    if (dragging.current) {
      const dx = e.clientX - dragging.current.x; const dy = e.clientY - dragging.current.y;
      dragging.current = { x: e.clientX, y: e.clientY };
      setOffset(o => ({ x: o.x + dx, y: o.y + dy }));
    }
  };
  return (
    <svg width={width} height={height} role="img" aria-label="Pareto Chart" onWheel={onWheel} onMouseDown={onMouseDown} onMouseUp={onMouseUp} onMouseMove={onMouseMove} style={{ cursor: 'grab', userSelect: 'none' }}>
      {/* Axes */}
      <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#586e75" />
      <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#586e75" />
      {/* Ticks */}
      {[0, 0.25, 0.5, 0.75, 1].map((t, i) => {
        const x = pad + t * (width - 2 * pad);
        const y = (height - pad) - t * (height - 2 * pad);
        return (
          <g key={i}>
            <line x1={x} y1={height - pad} x2={x} y2={height - pad + 4} stroke="#586e75" />
            <line x1={pad - 4} y1={y} x2={pad} y2={y} stroke="#586e75" />
          </g>
        );
      })}
      {/* Points with zoom/pan */}
      <g transform={`translate(${offset.x}, ${offset.y}) scale(${zoom})`}>
        {points.map((p, idx) => (
          <g key={idx}>
            <circle cx={sx(p.cost)} cy={sy(p.output)} r={3} fill="#2aa198" />
            <title>{`Score: ${p.output.toFixed(3)}\nCost: ${p.cost.toFixed(2)}s`}</title>
          </g>
        ))}
      </g>
      {/* Legend */}
      <g transform={`translate(${width - pad - 90}, ${pad})`}>
        <rect x={0} y={-14} width={90} height={20} fill="#002b36" stroke="#586e75" />
        <circle cx={10} cy={-4} r={3} fill="#2aa198" />
        <text x={20} y={-1} fontSize={10} fill="#93a1a1">Pareto point</text>
      </g>
      {/* Labels */}
      <text x={width / 2} y={height - 4} textAnchor="middle" fontSize="10" fill="#93a1a1">Cost (s)</text>
      <text x={12} y={12} fontSize="10" fill="#93a1a1">Score</text>
    </svg>
  );
};

export default ParetoChart;
