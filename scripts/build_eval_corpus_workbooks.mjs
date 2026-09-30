// Source-asset authoring only. Normal eval runs copy the frozen workbooks.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const root = process.env.EVAL_CORPUS_ROOT ?? path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../evals/datasets/real_corpus');
const source = JSON.parse(await fs.readFile(path.join(root, 'source.json'), 'utf8'));
const qa = process.env.EVAL_CORPUS_QA;
if (!qa) throw new Error('EVAL_CORPUS_QA is required for workbook verification');
await fs.mkdir(qa, { recursive: true });
await fs.mkdir(path.join(root, 'assets'), { recursive: true });
for (const doc of source.documents.filter(d => d.filename.endsWith('.xlsx'))) {
  const outputPath = path.join(root, 'assets', doc.filename);
  try { await fs.access(outputPath); throw new Error(`Refusing to overwrite ${outputPath}`); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  const workbook = Workbook.create();
  const sheet = workbook.worksheets.add(doc.sheet);
  sheet.showGridLines = false;
  const lastCol = String.fromCharCode(64 + doc.headers.length);
  const lastRow = doc.rows.length + 3;
  const range = sheet.getRange(`A1:${lastCol}${lastRow}`);
  range.format.font = { name: 'Arial', size: 11, color: '#172B4D' };
  range.format.columnWidth = 20;
  range.format.rowHeight = 29;
  range.format.verticalAlignment = 'center';
  sheet.getRange('A1').values = [[doc.title]];
  sheet.getRange('A1').format.font = { name: 'Arial', size: 16, bold: true };
  sheet.getRange(`A3:${lastCol}${lastRow}`).values = [doc.headers, ...doc.rows];
  sheet.getRange(`A3:${lastCol}3`).format = {
    fill: '#243B53', font: { name: 'Arial', size: 11, color: '#FFFFFF', bold: true },
    horizontalAlignment: 'center', verticalAlignment: 'center',
  };
  if (doc.document_key === 'plans') {
    sheet.getRange(`C4:D${lastRow}`).setNumberFormat('#,##0');
    sheet.getRange(`E1:E${lastRow}`).format.columnWidth = 30;
  } else {
    sheet.getRange(`D4:E${lastRow}`).setNumberFormat('#,##0');
  }
  workbook.recalculate();
  const values = await workbook.inspect({ kind: 'table', range: `${doc.sheet}!A3:${lastCol}${lastRow}`, include: 'values,formulas', tableMaxRows: 12, tableMaxCols: 6 });
  const errors = await workbook.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!', options: { useRegex: true, maxResults: 20 } });
  await fs.writeFile(path.join(qa, `${doc.document_key}-inspect.ndjson`), values.ndjson + '\n' + errors.ndjson);
  const image = await workbook.render({ sheetName: doc.sheet, range: `A1:${lastCol}${lastRow}`, scale: 2, format: 'png' });
  await fs.writeFile(path.join(qa, `${doc.document_key}.png`), new Uint8Array(await image.arrayBuffer()));
  await (await SpreadsheetFile.exportXlsx(workbook)).save(outputPath);
  console.log(doc.filename, 'exported with typed numeric values');
}
