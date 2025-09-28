import {
  Chart,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Filler } from 'chart.js';

let registered = false;

export const ensureChartsRegistered = () => {
  if (!registered) {
    Chart.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, ArcElement, Title, Tooltip, Legend, Filler);
    Chart.defaults.color = 'rgba(230, 222, 207, 0.88)';
    Chart.defaults.borderColor = 'rgba(107, 114, 128, 0.25)';
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    registered = true;
  }
};
