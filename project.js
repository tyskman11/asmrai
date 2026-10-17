// Pool-Anbindung fuer ASMRAI: verteilte Transkription (Whisper im Browser, jeder Abschnitt wird doppelt geprueft).
// Eingang: pool/input (nur dort abgelegte Dateien werden verteilt), Ergebnis: pool/results/<datei>.transcript.json
const path = require('path');
const { createQueue } = require('../../server/chunk-queue');
const log = require('../../server/logger').for('pool-asmrai');
const queue = createQueue({ id: 'asmrai', dir: __dirname, model: 'onnx-community/whisper-base', language: null, sources: [path.join(__dirname, 'input')], log });

module.exports = {
  id: 'asmrai',
  name: 'ASMRAI',
  description: 'Lernt den Stil von ASMR-Creators und erzeugt daraus neue ASMR-Sessions. Im Pool: Sprach-Transkription von Audio-Abschnitten.',
  enabled: true,
  alive: 'http://127.0.0.1:8780/',   // Antwort = start.py laeuft; der Pool startet das Projekt nie selbst
  browserOnly: true,
  createTask: ctx => queue.nextTask(ctx.userId),
  hasWork: () => queue.hasWork(),
  queueSummary: () => queue.summary(),
  queueOverview: () => queue.overview(),
  getChunk: task => queue.getChunk(task),
  acceptResult: (task, result, ctx) => queue.accept(task, result, ctx.userId, ctx.trusted),
};
