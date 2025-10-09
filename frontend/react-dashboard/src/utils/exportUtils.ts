import jsPDF from 'jspdf';
import 'jspdf-autotable';

export interface ExportOptions {
  filename?: string;
  includeHeaders?: boolean;
  dateFormat?: string;
  timezone?: string;
}

export interface ExportColumn {
  key: string;
  title: string;
  width?: number;
  render?: (value: any, record: any) => string;
}

// CSV Export
export const exportToCSV = (
  data: any[],
  columns: ExportColumn[],
  options: ExportOptions = {}
): void => {
  const {
    filename = `export-${new Date().toISOString().split('T')[0]}.csv`,
    includeHeaders = true
  } = options;

  const csvContent = [
    // Headers
    ...(includeHeaders ? [columns.map(col => col.title).join(',')] : []),
    // Data rows
    ...data.map(record => 
      columns.map(col => {
        const value = col.render ? col.render(record[col.key], record) : record[col.key];
        // Escape CSV values
        const stringValue = String(value || '');
        if (stringValue.includes(',') || stringValue.includes('"') || stringValue.includes('\n')) {
          return `"${stringValue.replace(/"/g, '""')}"`;
        }
        return stringValue;
      }).join(',')
    )
  ].join('\n');

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
  URL.revokeObjectURL(link.href);
};

// JSON Export
export const exportToJSON = (
  data: any[],
  options: ExportOptions = {}
): void => {
  const {
    filename = `export-${new Date().toISOString().split('T')[0]}.json`
  } = options;

  const jsonContent = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonContent], { type: 'application/json' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
  URL.revokeObjectURL(link.href);
};

// PDF Export
export const exportToPDF = (
  data: any[],
  columns: ExportColumn[],
  options: ExportOptions & {
    title?: string;
    subtitle?: string;
    orientation?: 'portrait' | 'landscape';
    pageSize?: 'A4' | 'A3' | 'letter';
  } = {}
): void => {
  const {
    filename = `export-${new Date().toISOString().split('T')[0]}.pdf`,
    title = 'Data Export',
    subtitle,
    orientation = 'portrait',
    pageSize = 'A4'
  } = options;

  const doc = new jsPDF({
    orientation,
    unit: 'mm',
    format: pageSize
  });

  // Add title
  doc.setFontSize(18);
  doc.text(title, 14, 20);

  // Add subtitle
  if (subtitle) {
    doc.setFontSize(12);
    doc.text(subtitle, 14, 30);
  }

  // Add export date
  doc.setFontSize(10);
  doc.text(`Exported on: ${new Date().toLocaleString()}`, 14, 40);

  // Prepare table data
  const tableData = data.map(record => 
    columns.map(col => {
      const value = col.render ? col.render(record[col.key], record) : record[col.key];
      return String(value || '');
    })
  );

  const tableHeaders = columns.map(col => col.title);

  // Add table
  (doc as any).autoTable({
    head: [tableHeaders],
    body: tableData,
    startY: 50,
    styles: {
      fontSize: 8,
      cellPadding: 3
    },
    headStyles: {
      fillColor: [74, 158, 255],
      textColor: 255,
      fontStyle: 'bold'
    },
    alternateRowStyles: {
      fillColor: [248, 250, 252]
    },
    columnStyles: columns.reduce((acc, col, index) => {
      if (col.width) {
        acc[index] = { cellWidth: col.width };
      }
      return acc;
    }, {} as any)
  });

  // Save PDF
  doc.save(filename);
};

// Excel Export (using CSV format for simplicity)
export const exportToExcel = (
  data: any[],
  columns: ExportColumn[],
  options: ExportOptions = {}
): void => {
  const {
    filename = `export-${new Date().toISOString().split('T')[0]}.xlsx`
  } = options;

  // Create HTML table for Excel
  const htmlContent = `
    <html>
      <head>
        <meta charset="utf-8">
        <style>
          table { border-collapse: collapse; width: 100%; }
          th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
          th { background-color: #4a9eff; color: white; font-weight: bold; }
          tr:nth-child(even) { background-color: #f8fafc; }
        </style>
      </head>
      <body>
        <table>
          <thead>
            <tr>
              ${columns.map(col => `<th>${col.title}</th>`).join('')}
            </tr>
          </thead>
          <tbody>
            ${data.map(record => `
              <tr>
                ${columns.map(col => {
                  const value = col.render ? col.render(record[col.key], record) : record[col.key];
                  return `<td>${String(value || '')}</td>`;
                }).join('')}
              </tr>
            `).join('')}
          </tbody>
        </table>
      </body>
    </html>
  `;

  const blob = new Blob([htmlContent], { type: 'application/vnd.ms-excel' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
  URL.revokeObjectURL(link.href);
};

// Chart Export
export const exportChartToPNG = (
  canvas: HTMLCanvasElement,
  options: ExportOptions = {}
): void => {
  const {
    filename = `chart-${new Date().toISOString().split('T')[0]}.png`
  } = options;

  const link = document.createElement('a');
  link.download = filename;
  link.href = canvas.toDataURL('image/png');
  link.click();
};

export const exportChartToSVG = (
  svgElement: SVGElement,
  options: ExportOptions = {}
): void => {
  const {
    filename = `chart-${new Date().toISOString().split('T')[0]}.svg`
  } = options;

  const svgData = new XMLSerializer().serializeToString(svgElement);
  const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(svgBlob);
  link.download = filename;
  link.click();
  URL.revokeObjectURL(link.href);
};

// Bulk Export
export const exportMultipleFormats = async (
  data: any[],
  columns: ExportColumn[],
  formats: ('csv' | 'json' | 'pdf' | 'excel')[],
  options: ExportOptions = {}
): Promise<void> => {
  const promises = formats.map(format => {
    switch (format) {
      case 'csv':
        return new Promise<void>(resolve => {
          exportToCSV(data, columns, options);
          resolve();
        });
      case 'json':
        return new Promise<void>(resolve => {
          exportToJSON(data, options);
          resolve();
        });
      case 'pdf':
        return new Promise<void>(resolve => {
          exportToPDF(data, columns, options);
          resolve();
        });
      case 'excel':
        return new Promise<void>(resolve => {
          exportToExcel(data, columns, options);
          resolve();
        });
      default:
        return Promise.resolve();
    }
  });

  await Promise.all(promises);
};
