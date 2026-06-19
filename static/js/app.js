const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const state = {
  dashboard: null,
  training: null,
  datasets: [],
  models: [],
  guided: [],
  replays: [],
  tournament: null,
  view: 'overview',
  game: null,
  color: 'white',
  orientation: 'white',
  strength: 'tactical',
  selected: null,
  moves: [],
  lastMove: null,
  network: null,
  networkNodes: [],
  mentor: { position: null, selected: null, move: null, history: [] },
  replay: null,
  replayIndex: 0,
  replayTimer: null,
};

const pieceFiles = {
  P: 'w_Pawn.png', N: 'w_Knight.png', B: 'w_Bishop.png', R: 'w_Rook.png', Q: 'w_Queen.png', K: 'w_King.png',
  p: 'b_Pawn.png', n: 'b_Knight.png', b: 'b_Bishop.png', r: 'b_Rook.png', q: 'b_Queen.png', k: 'b_King.png',
};

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({ ok: false, error: 'Resposta inválida do servidor.' }));
  if (!response.ok || !payload.ok) throw new Error(payload.error || `Erro ${response.status}`);
  return payload;
}

function toast(message, type = 'success') {
  const element = document.createElement('div');
  element.className = `toast ${type}`;
  element.textContent = message;
  $('#toastStack').appendChild(element);
  setTimeout(() => element.remove(), 3800);
}

function formatNumber(value) {
  const n = Number(value || 0);
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(n >= 100_000 ? 0 : 1)}k`;
  return n.toLocaleString('pt-BR');
}

function formatBytes(value) {
  if (!value) return '0 KB';
  if (value > 1_000_000) return `${(value / 1_000_000).toFixed(1)} MB`;
  return `${Math.ceil(value / 1000)} KB`;
}

function ago(date) {
  const seconds = Math.max(1, (Date.now() - new Date(date).getTime()) / 1000);
  if (seconds < 90) return 'agora';
  if (seconds < 3600) return `há ${Math.floor(seconds / 60)} min`;
  if (seconds < 86400) return `há ${Math.floor(seconds / 3600)} h`;
  return new Date(date).toLocaleDateString('pt-BR');
}

const viewNames = { overview: 'VISÃO GERAL', arena: 'ARENA', championship: 'CAMPEONATO', training: 'TREINAMENTO', mentor: 'MENTOR', datasets: 'DADOS PGN', replays: 'REPLAYS', network: 'REDE NEURAL', models: 'MODELOS' };

function changeView(view) {
  if (!viewNames[view]) return;
  state.view = view;
  $$('.view').forEach((el) => el.classList.toggle('active', el.id === `view-${view}`));
  $$('.nav-item').forEach((el) => el.classList.toggle('active', el.dataset.viewLink === view));
  $('#pageCrumb').textContent = viewNames[view];
  $('.sidebar').classList.remove('open');
  window.history.replaceState(null, '', `#${view}`);
  window.scrollTo({ top: 0, behavior: 'smooth' });
  if (view === 'network') loadNetwork();
  if (view === 'arena' && !state.game) newGame();
  if (view === 'models') renderModels();
  if (view === 'mentor') loadGuided(true);
  if (view === 'replays') loadReplays(true);
  if (view === 'championship') renderTournament(state.tournament || {});
}

async function loadDashboard(silent = false) {
  try {
    const payload = await api('/api/dashboard');
    state.dashboard = payload.summary;
    state.datasets = payload.datasets;
    state.models = payload.models;
    state.training = payload.training;
    renderDashboard();
    renderDatasets();
    renderTrainingDatasets();
    renderModels();
    renderModelSelectors();
    renderTraining(payload.training);
    await Promise.all([loadGuided(true), loadReplays(true)]);
    if (!silent) toast('Laboratório sincronizado.');
  } catch (error) {
    toast(error.message, 'error');
  }
}

function resultPills(results = {}) {
  return `<span class="result-pills"><i>${results['1-0'] || 0} W</i><i>${results['0-1'] || 0} B</i><i>${results['1/2-1/2'] || 0} D</i></span>`;
}

function renderDashboard() {
  const summary = state.dashboard || {};
  $('#statGames').textContent = formatNumber(summary.games);
  $('#statPositions').textContent = formatNumber(summary.positions);
  $('#statModels').textContent = formatNumber(summary.models);
  $('#statParameters').textContent = formatNumber(summary.parameters);
  $('#sideModelName').textContent = summary.active_model || 'Aurora';
  $('#sideModelMeta').textContent = `${formatNumber(summary.trained_positions)} posições · CPU`;
  $('#opponentName').textContent = summary.active_model || 'Aurora';
  const rows = state.datasets.slice(0, 4).map((dataset) => `<tr>
    <td><strong>${escapeHtml(dataset.name)}</strong><small>${escapeHtml((dataset.players || []).slice(0, 2).join(' × '))}</small></td>
    <td>${dataset.games}</td><td>${formatNumber(dataset.positions)}</td><td>${resultPills(dataset.results)}</td><td>${ago(dataset.imported_at)}</td>
  </tr>`).join('');
  $('#recentDatasets').innerHTML = rows || '<tr><td colspan="5">Nenhum PGN importado ainda.</td></tr>';
  drawMetricChart($('#overviewChart'), state.training?.metrics || []);
}

function escapeHtml(value = '') {
  return String(value).replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char]));
}

function drawMetricChart(canvas, metrics) {
  const valid = metrics.filter((m) => Number.isFinite(Number(m.loss)));
  $('#chartEmpty').classList.toggle('hidden', valid.length > 1);
  if (valid.length < 2) return;
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = rect.width * scale; canvas.height = rect.height * scale;
  const ctx = canvas.getContext('2d'); ctx.scale(scale, scale);
  const w = rect.width, h = rect.height, pad = 24;
  const values = valid.map((m) => Number(m.loss));
  const min = Math.min(...values), max = Math.max(...values, min + 0.001);
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = 'rgba(230,240,232,.06)'; ctx.lineWidth = 1;
  for (let i = 0; i < 5; i += 1) { const y = pad + (h - pad * 2) * i / 4; ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - pad, y); ctx.stroke(); }
  const points = values.map((value, index) => ({ x: pad + (w - pad * 2) * index / (values.length - 1), y: pad + (h - pad * 2) * (max - value) / (max - min) }));
  const gradient = ctx.createLinearGradient(0, pad, 0, h); gradient.addColorStop(0, 'rgba(134,231,184,.18)'); gradient.addColorStop(1, 'rgba(134,231,184,0)');
  ctx.beginPath(); ctx.moveTo(points[0].x, h - pad); points.forEach((p) => ctx.lineTo(p.x, p.y)); ctx.lineTo(points.at(-1).x, h - pad); ctx.fillStyle = gradient; ctx.fill();
  ctx.beginPath(); points.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)); ctx.strokeStyle = '#86e7b8'; ctx.lineWidth = 2; ctx.stroke();
  points.forEach((p) => { ctx.beginPath(); ctx.arc(p.x, p.y, 3, 0, Math.PI * 2); ctx.fillStyle = '#0d1411'; ctx.fill(); ctx.strokeStyle = '#86e7b8'; ctx.stroke(); });
}

function renderDatasets(filter = '') {
  const needle = filter.toLowerCase();
  const items = state.datasets.filter((d) => `${d.name} ${d.primary_player || ''} ${d.player_folder || ''} ${(d.players || []).join(' ')}`.toLowerCase().includes(needle));
  $('#datasetCount').textContent = `${state.datasets.length} conjunto${state.datasets.length === 1 ? '' : 's'}`;
  const groups = new Map();
  items.forEach((dataset) => { const player = dataset.primary_player || 'Não classificado'; if (!groups.has(player)) groups.set(player, []); groups.get(player).push(dataset); });
  $('#datasetGrid').innerHTML = [...groups.entries()].map(([player, datasets]) => `<div class="player-folder-header"><span>▰</span><div><small>PASTA DE JOGADOR</small><strong>${escapeHtml(player)}</strong></div><b>${datasets.length} arquivo${datasets.length === 1 ? '' : 's'}</b></div>${datasets.map((dataset) => `<article class="dataset-card">
    <div class="file-top"><span class="file-icon">PGN</span><span class="file-size">${formatBytes(dataset.size)}</span></div>
    <h3 title="${escapeHtml(dataset.name)}">${escapeHtml(dataset.name)}</h3>
    <p class="player-path">⌑ ${escapeHtml(dataset.player_folder || 'Aguardando classificação')}</p>
    <p title="${escapeHtml((dataset.players || []).join(', '))}">${dataset.primary_player_games || 0}/${dataset.games} partidas de ${escapeHtml(dataset.primary_player || 'jogador não identificado')}</p>
    <div class="dataset-metrics"><span><small>PARTIDAS</small><strong>${dataset.games}</strong></span><span><small>POSIÇÕES</small><strong>${formatNumber(dataset.positions)}</strong></span></div>
    <div class="dataset-actions"><button data-rename-dataset="${dataset.id}">✎ Renomear</button><button class="destructive" data-delete-dataset="${dataset.id}">⌫ Apagar</button></div>
  </article>`).join('')}`).join('') || '<div class="empty-library">Nenhum conjunto encontrado.</div>';
  $$('[data-rename-dataset]').forEach((button) => button.addEventListener('click', () => renameDataset(button.dataset.renameDataset)));
  $$('[data-delete-dataset]').forEach((button) => button.addEventListener('click', () => deleteDataset(button.dataset.deleteDataset)));
}

async function renameDataset(id) {
  const dataset = state.datasets.find((item) => item.id === id);
  const name = window.prompt('Novo nome do dataset:', dataset?.name || '');
  if (!name || name.trim() === dataset?.name) return;
  try { await api(`/api/datasets/${encodeURIComponent(id)}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: name.trim() }) }); toast('Dataset renomeado.'); await loadDashboard(true); }
  catch (error) { toast(error.message, 'error'); }
}

async function deleteDataset(id) {
  const dataset = state.datasets.find((item) => item.id === id);
  if (!window.confirm(`Apagar o dataset “${dataset?.name || id}” e seu arquivo PGN?`)) return;
  try { await api(`/api/datasets/${encodeURIComponent(id)}`, { method: 'DELETE' }); toast('Dataset apagado.'); await loadDashboard(true); }
  catch (error) { toast(error.message, 'error'); }
}

function renderTrainingDatasets() {
  $('#trainingDatasets').innerHTML = state.datasets.map((dataset) => `<label class="dataset-check"><input type="checkbox" name="dataset_ids" value="${dataset.id}" checked><span>${escapeHtml(dataset.name)}<small>${escapeHtml(dataset.primary_player || 'não classificado')}</small></span><small>${dataset.games} partidas</small></label>`).join('') || '<div class="empty-library">Importe um PGN para usar imitação.</div>';
}

function renderModelSelectors() {
  const options = state.models.map((model) => `<option value="${model.id}">${escapeHtml(model.config?.name || model.id)} · ${formatNumber(model.evaluation?.parameters)} params</option>`).join('');
  const base = $('#baseModelSelect');
  if (base) {
    const previous = base.value;
    base.innerHTML = `<option value="">Começar do zero</option>${options}`;
    if ([...base.options].some((option) => option.value === previous)) base.value = previous;
  }
  const checks = $('#championshipModels');
  if (checks) checks.innerHTML = state.models.map((model) => `<label class="dataset-check"><input type="checkbox" name="model_ids" value="${model.id}" checked><span>${escapeHtml(model.config?.name || model.id)}</span><small>${model.active ? 'ativo' : formatNumber(model.evaluation?.parameters)}</small></label>`).join('') || '<div class="empty-library">Treine ou importe um modelo primeiro.</div>';
}

async function uploadFiles(files) {
  const pgns = [...files].filter((file) => file.name.toLowerCase().endsWith('.pgn'));
  if (!pgns.length) return toast('Escolha arquivos com extensão .pgn.', 'error');
  const zone = $('#uploadZone'); zone.classList.add('uploading');
  const form = new FormData(); pgns.forEach((file) => form.append('files', file));
  try {
    const payload = await api('/api/datasets/import', { method: 'POST', body: form });
    const owners = [...new Set(payload.datasets.map((dataset) => dataset.primary_player))].join(', ');
    toast(`${payload.datasets.length} arquivo(s) organizado(s) em: ${owners}.`);
    await loadDashboard(true);
  } catch (error) { toast(error.message, 'error'); }
  finally { zone.classList.remove('uploading'); $('#pgnInput').value = ''; }
}

function trainingConfig(form) {
  const data = new FormData(form);
  return {
    name: data.get('name'), mode: data.get('mode'), hidden_layers: data.get('hidden_layers'), seed: Number(data.get('seed')),
    epochs: Number(data.get('epochs')), batch_size: Number(data.get('batch_size')), learning_rate: Number(data.get('learning_rate')),
    selfplay_episodes: Number(data.get('selfplay_episodes')), temperature: Number(data.get('temperature')),
    max_positions: Number(data.get('max_positions')), dataset_ids: data.getAll('dataset_ids'),
    include_guided: data.get('include_guided') === 'on', base_model_id: data.get('base_model_id') || null,
  };
}

async function startTraining(event) {
  event.preventDefault();
  try {
    const payload = await api('/api/training/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(trainingConfig(event.currentTarget)) });
    state.training = payload.training; renderTraining(payload.training); toast('Experimento iniciado.');
  } catch (error) { toast(error.message, 'error'); }
}

function renderTraining(training) {
  if (!training) return;
  const running = !!training.running;
  $('.monitor-card').classList.toggle('running', running);
  $('#trainingStage').textContent = String(training.stage || 'Pronto').toUpperCase();
  $('#trainingHeadline').textContent = running ? (training.config?.name || 'Treinando') : training.stage === 'Concluído' ? 'Checkpoint pronto' : training.stage === 'Falha' ? 'Algo precisa de atenção' : 'Aguardando configuração';
  $('#trainingSubline').textContent = training.error || (running ? 'Atualizando pesos no laboratório local.' : training.model_id ? `Salvo como ${training.model_id}` : 'Escolha os parâmetros e inicie quando estiver pronto.');
  $('#trainingProgress').style.width = `${training.progress || 0}%`;
  $('#sideProgress').style.width = `${training.progress || 0}%`;
  $('#progressValue').textContent = `${training.progress || 0}%`;
  $('#lossValue').textContent = training.loss == null ? '—' : Number(training.loss).toFixed(4);
  $('#epochValue').textContent = training.epoch || '—';
  $('#stopTrainingButton').classList.toggle('hidden', !running);
  $('#startTrainingButton').disabled = running;
  $('#trainingConsole').innerHTML = (training.logs || []).map((log) => `<p><time>${escapeHtml(log.time)}</time>${escapeHtml(log.message)}</p>`).join('');
  drawMetricChart($('#overviewChart'), training.metrics || []);
}

async function pollTraining() {
  try {
    const payload = await api('/api/training/status');
    const wasRunning = state.training?.running;
    state.training = payload.training;
    renderTraining(payload.training);
    if (wasRunning && !payload.training.running) {
      await loadDashboard(true);
      toast(payload.training.error ? payload.training.error : 'Treinamento concluído e checkpoint salvo.', payload.training.error ? 'error' : 'success');
    }
  } catch (_) { /* transient server restart */ }
}

function parseFen(fen) {
  const board = {};
  const rows = fen.split(' ')[0].split('/');
  rows.forEach((row, rowIndex) => {
    let file = 0;
    for (const char of row) {
      if (/\d/.test(char)) file += Number(char);
      else { board[`${'abcdefgh'[file]}${8 - rowIndex}`] = char; file += 1; }
    }
  });
  return board;
}

function renderBoard() {
  const board = parseFen(state.game?.fen || '8/8/8/8/8/8/8/8 w - - 0 1');
  const files = state.orientation === 'white' ? [...'abcdefgh'] : [...'hgfedcba'];
  const ranks = state.orientation === 'white' ? [8, 7, 6, 5, 4, 3, 2, 1] : [1, 2, 3, 4, 5, 6, 7, 8];
  const targets = state.selected ? (state.game?.legal_moves || []).filter((m) => m.startsWith(state.selected)).map((m) => m.slice(2, 4)) : [];
  $('#chessboard').innerHTML = ranks.flatMap((rank, row) => files.map((file, col) => {
    const square = `${file}${rank}`, piece = board[square], isLight = (file.charCodeAt(0) - 97 + rank) % 2 === 1;
    const classes = ['square', isLight ? 'light' : 'dark', state.selected === square ? 'selected' : '', targets.includes(square) ? 'target' : '', piece ? 'has-piece' : '', state.lastMove?.includes(square) ? 'last' : ''].filter(Boolean).join(' ');
    const showCoord = col === 0 || row === 7;
    return `<button class="${classes}" data-square="${square}" data-coord="${showCoord ? square : ''}" aria-label="${square}">${piece ? `<img class="piece" draggable="false" src="/static/assets/pieces/${pieceFiles[piece]}" alt="${piece}">` : ''}</button>`;
  })).join('');
  $$('.square').forEach((el) => el.addEventListener('click', () => selectSquare(el.dataset.square, board)));
}

function selectSquare(square, board) {
  if (!state.game || state.game.game_over) return;
  const userTurn = (state.game.turn === 'white') === (state.color === 'white');
  if (!userTurn) return;
  if (state.selected) {
    const candidates = state.game.legal_moves.filter((move) => move.startsWith(state.selected + square));
    if (candidates.length) { playMove(candidates.find((m) => m.endsWith('q')) || candidates[0]); return; }
  }
  const piece = board[square];
  const own = piece && ((piece === piece.toUpperCase()) === (state.color === 'white'));
  state.selected = own ? square : null;
  renderBoard();
}

async function newGame() {
  try {
    state.strength = $('#arenaStrength')?.value || 'tactical';
    state.orientation = state.color;
    const payload = await api('/api/play/new', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ color: state.color, strength: state.strength }) });
    state.game = payload.game; state.selected = null; state.moves = []; state.lastMove = payload.game.ai_move;
    if (payload.game.ai_move) state.moves.push(payload.game.ai_move);
    $('#playerColor').textContent = state.color === 'white' ? 'brancas' : 'pretas';
    renderBoard(); renderMoveLog(); updateGameStatus();
  } catch (error) { toast(error.message, 'error'); }
}

async function playMove(move) {
  const ownMove = move; state.selected = null;
  try {
    const payload = await api('/api/play/move', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: state.game.session_id, move }) });
    state.game = payload.game; state.moves.push(ownMove); if (payload.game.ai_move) state.moves.push(payload.game.ai_move);
    state.lastMove = payload.game.ai_move || ownMove; renderBoard(); renderMoveLog(); updateGameStatus();
  } catch (error) { toast(error.message, 'error'); renderBoard(); }
}

function renderMoveLog() {
  if (!state.moves.length) { $('#moveLog').innerHTML = '<div class="empty-state"><span>♙</span><p>Os lances aparecerão aqui.</p></div>'; return; }
  const rows = [];
  for (let i = 0; i < state.moves.length; i += 2) rows.push(`<div class="move-row"><span>${i / 2 + 1}.</span><b>${state.moves[i] || ''}</b><b>${state.moves[i + 1] || ''}</b></div>`);
  $('#moveLog').innerHTML = rows.join(''); $('#moveLog').scrollTop = $('#moveLog').scrollHeight;
}

function updateGameStatus() {
  const game = state.game;
  $('#gameState').textContent = game?.game_over ? (game.result || 'FIM') : game?.check ? 'XEQUE' : 'EM JOGO';
  $('#gameHint').textContent = game?.game_over ? `Resultado ${game.result}` : game?.ai_move ? `Rede jogou ${game.ai_move}` : 'Sua vez';
}

function renderPositionBoard(containerSelector, fen, options = {}) {
  const container = $(containerSelector); if (!container) return;
  const board = parseFen(fen || '8/8/8/8/8/8/8/8 w - - 0 1');
  const orientation = options.orientation || 'white';
  const files = orientation === 'white' ? [...'abcdefgh'] : [...'hgfedcba'];
  const ranks = orientation === 'white' ? [8, 7, 6, 5, 4, 3, 2, 1] : [1, 2, 3, 4, 5, 6, 7, 8];
  const targets = options.targets || [];
  container.innerHTML = ranks.flatMap((rank, row) => files.map((file, col) => {
    const square = `${file}${rank}`, piece = board[square], isLight = (file.charCodeAt(0) - 97 + rank) % 2 === 1;
    const classes = ['square', isLight ? 'light' : 'dark', options.selected === square ? 'selected' : '', targets.includes(square) ? 'target' : '', piece ? 'has-piece' : '', options.lastMove?.includes(square) ? 'last' : ''].filter(Boolean).join(' ');
    const showCoord = col === 0 || row === 7;
    return `<button class="${classes}" data-square="${square}" data-coord="${showCoord ? square : ''}" aria-label="${square}">${piece ? `<img class="piece" draggable="false" src="/static/assets/pieces/${pieceFiles[piece]}" alt="${piece}">` : ''}</button>`;
  })).join('');
  if (options.onClick) $$('.square', container).forEach((element) => element.addEventListener('click', () => options.onClick(element.dataset.square, board)));
  else $$('.square', container).forEach((element) => { element.disabled = true; });
}

async function loadGuided(silent = false) {
  try {
    const payload = await api('/api/guided'); state.guided = payload.examples;
    if (!state.mentor.position) {
      const position = await api('/api/guided/position'); state.mentor.position = position.position;
    }
    renderMentor(); renderGuidedExamples();
  } catch (error) { if (!silent) toast(error.message, 'error'); }
}

function renderMentor() {
  const mentor = state.mentor, position = mentor.position;
  if (!position) return;
  const targets = mentor.selected ? position.legal_moves.filter((move) => move.startsWith(mentor.selected)).map((move) => move.slice(2, 4)) : [];
  renderPositionBoard('#mentorBoard', position.fen, { selected: mentor.selected, targets, onClick: selectMentorSquare });
  $('#mentorTurn').textContent = position.turn === 'white' ? 'Brancas' : 'Pretas';
  $('#mentorMove').textContent = mentor.move || 'Selecione origem e destino';
  $('#saveGuided').disabled = !mentor.move;
  $('#mentorCount').textContent = `${state.guided.length} exemplo${state.guided.length === 1 ? '' : 's'}`;
}

function selectMentorSquare(square, board) {
  const mentor = state.mentor, position = mentor.position;
  if (mentor.selected) {
    const candidates = position.legal_moves.filter((move) => move.startsWith(mentor.selected + square));
    if (candidates.length) {
      mentor.move = candidates.find((move) => move.endsWith('q')) || candidates[0]; mentor.selected = null; renderMentor(); return;
    }
  }
  const piece = board[square], whiteTurn = position.turn === 'white';
  const own = piece && ((piece === piece.toUpperCase()) === whiteTurn);
  mentor.selected = own ? square : null; mentor.move = null; renderMentor();
}

async function saveGuidedExample(event) {
  event.preventDefault();
  if (!state.mentor.move) return;
  const data = new FormData(event.currentTarget);
  try {
    const payload = await api('/api/guided/examples', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({
      fen: state.mentor.position.fen, move: state.mentor.move, note: data.get('note'), priority: Number(data.get('priority')),
    }) });
    state.mentor.history.push(state.mentor.position); state.mentor.position = payload.next_position; state.mentor.selected = null; state.mentor.move = null; state.guided = payload.examples;
    event.currentTarget.elements.note.value = ''; renderMentor(); renderGuidedExamples(); renderTrainingDatasets(); toast(`Lance ${payload.example.san} adicionado ao Mentor.`);
  } catch (error) { toast(error.message, 'error'); }
}

async function resetMentor() {
  const payload = await api('/api/guided/position'); state.mentor = { position: payload.position, selected: null, move: null, history: [] }; renderMentor();
}

function renderGuidedExamples() {
  const grid = $('#guidedExamples'); if (!grid) return;
  grid.innerHTML = state.guided.map((example) => `<article class="guided-card"><div class="guided-card-head"><code>${escapeHtml(example.san)} · ${escapeHtml(example.move)}</code><b>${example.priority}×</b></div><p>${escapeHtml(example.note || 'Sem comentário — demonstração direta.')}</p><button data-delete-guided="${example.id}">remover exemplo</button></article>`).join('') || '<div class="empty-library">Selecione um lance no tabuleiro para criar a primeira demonstração.</div>';
  $$('[data-delete-guided]').forEach((button) => button.addEventListener('click', async () => {
    try { const payload = await api(`/api/guided/examples/${button.dataset.deleteGuided}`, { method: 'DELETE' }); state.guided = payload.examples; renderGuidedExamples(); renderMentor(); }
    catch (error) { toast(error.message, 'error'); }
  }));
}

function tournamentConfig(form) {
  const data = new FormData(form);
  return { model_ids: data.getAll('model_ids'), rounds: Number(data.get('rounds')), max_plies: Number(data.get('max_plies')),
    temperature: Number(data.get('temperature')), search_level: data.get('search_level') };
}

async function startTournament(event) {
  event.preventDefault();
  try { const payload = await api('/api/tournament/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(tournamentConfig(event.currentTarget)) }); state.tournament = payload.tournament; renderTournament(state.tournament); toast('Campeonato iniciado.'); }
  catch (error) { toast(error.message, 'error'); }
}

function renderTournament(data = {}) {
  if (!$('#standingsBody')) return;
  const running = !!data.running;
  $('#championshipBadge').innerHTML = `<i></i>${escapeHtml(String(data.stage || 'Pronto').toUpperCase())}`;
  $('#championshipStage').textContent = data.stage || 'Aguardando competidores';
  $('#championshipProgress').textContent = `${data.progress || 0}%`;
  $('#championshipProgressBar').style.width = `${data.progress || 0}%`;
  $('#startChampionship').disabled = running; $('#stopChampionship').classList.toggle('hidden', !running);
  $('#standingsBody').innerHTML = (data.standings || []).map((player) => `<tr><td>${escapeHtml(player.name)}</td><td>${player.points.toFixed(1)}</td><td>${player.played}</td><td>${player.wins}/${player.draws}/${player.losses}</td><td>${player.elo.toFixed(1)}</td></tr>`).join('') || '<tr><td colspan="5">A classificação aparecerá aqui.</td></tr>';
  $('#championshipGames').innerHTML = (data.games || []).map((game) => `<div class="match-row"><small>R${game.round}</small><strong>${escapeHtml(game.white)}</strong><span class="match-result">${game.result}</span><strong>${escapeHtml(game.black)}</strong><button data-watch-replay="${game.replay_id}">assistir</button></div>`).join('') || '<div class="empty-library">Nenhuma partida disputada.</div>';
  $$('[data-watch-replay]').forEach((button) => button.addEventListener('click', () => { changeView('replays'); loadReplay(button.dataset.watchReplay); }));
}

async function pollTournament() {
  try {
    const payload = await api('/api/tournament/status'); const wasRunning = state.tournament?.running;
    state.tournament = payload.tournament; renderTournament(state.tournament);
    if (wasRunning && !state.tournament.running) { await loadReplays(true); toast(state.tournament.error || 'Campeonato concluído.', state.tournament.error ? 'error' : 'success'); }
  } catch (_) { /* restart */ }
}

async function loadReplays(silent = false) {
  try { const payload = await api('/api/replays'); state.replays = payload.replays; renderReplayList(); if (!state.replay && state.replays.length && state.view === 'replays') loadReplay(state.replays[0].id); }
  catch (error) { if (!silent) toast(error.message, 'error'); }
}

function renderReplayList() {
  const list = $('#replayList'); if (!list) return;
  const filter = $('#replayFilter')?.value || 'all'; const items = state.replays.filter((item) => filter === 'all' || item.source === filter);
  $('#replayTotal').textContent = `${state.replays.length} partida${state.replays.length === 1 ? '' : 's'}`;
  list.innerHTML = items.map((item) => `<button class="replay-item ${state.replay?.id === item.id ? 'active' : ''}" data-replay-id="${item.id}"><i>${item.source === 'championship' ? '♛' : '⌁'}</i><span><strong>${escapeHtml(item.white)} × ${escapeHtml(item.black)}</strong><small>${item.source === 'championship' ? 'campeonato' : 'treino'} · ${item.plies} lances · ${ago(item.created_at)}</small></span><b>${item.result}</b></button>`).join('') || '<div class="empty-library">Nenhum replay nesta categoria.</div>';
  $$('[data-replay-id]').forEach((button) => button.addEventListener('click', () => loadReplay(button.dataset.replayId)));
}

async function loadReplay(id) {
  try { const payload = await api(`/api/replays/${id}`); state.replay = payload.replay; state.replayIndex = 0; stopReplay(); renderReplayList(); renderReplay(); }
  catch (error) { toast(error.message, 'error'); }
}

function renderReplay() {
  const replay = state.replay;
  if (!replay) { renderPositionBoard('#replayBoard', '8/8/8/8/8/8/8/8 w - - 0 1'); return; }
  const index = Math.max(0, Math.min(state.replayIndex, replay.fens.length - 1)); state.replayIndex = index;
  const last = index ? replay.moves[index - 1] : null;
  renderPositionBoard('#replayBoard', replay.fens[index], { lastMove: last });
  $('#replaySource').textContent = replay.source === 'championship' ? 'CAMPEONATO' : 'TREINO POR AUTOJOGO';
  $('#replayPlayers').textContent = `${replay.white} × ${replay.black}`; $('#replayResult').textContent = replay.result;
  $('#replayPly').textContent = `${index}/${replay.moves.length}`; $('#replaySan').textContent = index ? `${replay.sans[index - 1]} · ${replay.moves[index - 1]}` : 'posição inicial';
}

function replayStep(delta) { if (!state.replay) return; state.replayIndex = Math.max(0, Math.min(state.replay.moves.length, state.replayIndex + delta)); renderReplay(); if (state.replayIndex >= state.replay.moves.length) stopReplay(); }
function stopReplay() { if (state.replayTimer) clearInterval(state.replayTimer); state.replayTimer = null; if ($('#replayPlay')) $('#replayPlay').textContent = '▶'; }
function toggleReplay() { if (!state.replay) return; if (state.replayTimer) return stopReplay(); $('#replayPlay').textContent = 'Ⅱ'; state.replayTimer = setInterval(() => replayStep(1), 850); }

async function loadNetwork() {
  try {
    const fen = state.game?.fen ? `?fen=${encodeURIComponent(state.game.fen)}` : '';
    const payload = await api(`/api/network${fen}`); state.network = payload.network;
    const layerSizes = payload.network.layers.map((layer) => layer.total);
    $('#networkArchitecture').textContent = layerSizes.join(' → ');
    $('#networkParameters').textContent = formatNumber(payload.network.parameters);
    $('#networkPositions').textContent = formatNumber(payload.network.trained_positions);
    drawNetwork();
  } catch (error) { toast(error.message, 'error'); }
}

function drawNetwork() {
  if (!state.network) return;
  const canvas = $('#networkCanvas'), rect = canvas.getBoundingClientRect(), scale = window.devicePixelRatio || 1;
  canvas.width = rect.width * scale; canvas.height = rect.height * scale;
  const ctx = canvas.getContext('2d'); ctx.scale(scale, scale); ctx.clearRect(0, 0, rect.width, rect.height);
  const padX = Math.max(55, rect.width * .07), padY = 65;
  const positions = new Map(); state.networkNodes = [];
  state.network.layers.forEach((layer, layerIndex) => {
    const x = padX + (rect.width - padX * 2) * layerIndex / Math.max(1, state.network.layers.length - 1);
    layer.nodes.forEach((node, nodeIndex) => {
      const y = padY + (rect.height - padY * 2) * (nodeIndex + .5) / layer.nodes.length;
      positions.set(node.id, { x, y, node, layer: layer.name });
    });
  });
  state.network.edges.forEach((edge) => {
    const a = positions.get(edge.source), b = positions.get(edge.target); if (!a || !b) return;
    const magnitude = Math.min(1, Math.abs(edge.weight) * 7);
    ctx.beginPath(); ctx.moveTo(a.x, a.y); const dx = (b.x - a.x) * .48; ctx.bezierCurveTo(a.x + dx, a.y, b.x - dx, b.y, b.x, b.y);
    ctx.strokeStyle = edge.weight >= 0 ? `rgba(134,231,184,${.035 + magnitude * .22})` : `rgba(168,137,232,${.035 + magnitude * .22})`;
    ctx.lineWidth = .3 + magnitude * 1.25; ctx.stroke();
  });
  state.network.layers.forEach((layer, layerIndex) => {
    const sample = positions.get(layer.nodes[0]?.id); if (!sample) return;
    ctx.fillStyle = '#78857c'; ctx.font = '500 9px "DM Mono"'; ctx.textAlign = 'center';
    ctx.fillText(layer.name.toUpperCase(), sample.x, 29); ctx.fillStyle = '#4f5c54'; ctx.fillText(`${layer.total} NÓS`, sample.x, 44);
    layer.nodes.forEach((node) => {
      const p = positions.get(node.id), activation = Math.min(1, Math.abs(node.activation) / 2), radius = 3.3 + activation * 4.5;
      ctx.beginPath(); ctx.arc(p.x, p.y, radius + 4, 0, Math.PI * 2); ctx.fillStyle = `rgba(134,231,184,${activation * .09})`; ctx.fill();
      ctx.beginPath(); ctx.arc(p.x, p.y, radius, 0, Math.PI * 2); ctx.fillStyle = activation > .35 ? '#86e7b8' : '#22342a'; ctx.fill(); ctx.strokeStyle = 'rgba(134,231,184,.45)'; ctx.lineWidth = 1; ctx.stroke();
      state.networkNodes.push({ ...p, radius: radius + 7 });
    });
  });
}

function networkHover(event) {
  const rect = event.currentTarget.getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top;
  const found = state.networkNodes.find((p) => Math.hypot(x - p.x, y - p.y) <= p.radius), tip = $('#networkTooltip');
  if (!found) { tip.style.display = 'none'; return; }
  tip.style.display = 'block'; tip.style.left = `${Math.min(rect.width - 170, x + 12)}px`; tip.style.top = `${Math.max(12, y - 18)}px`;
  tip.innerHTML = `${escapeHtml(found.layer)} · #${found.node.index}<br>ativação ${found.node.activation.toFixed(4)}`;
}

function renderModels() {
  $('#modelGrid').innerHTML = state.models.map((model) => {
    const config = model.config || {}, evaluation = model.evaluation || {};
    return `<article class="model-card ${model.active ? 'active' : ''}"><div class="model-top"><span class="file-icon">NPZ</span>${model.active ? '<span class="model-badge">ATIVO</span>' : `<span class="file-size">${formatBytes(model.size)}</span>`}</div><h3>${escapeHtml(config.name || model.id)}</h3><p>${escapeHtml(model.id)} · ${model.parent_model_id ? `filho de ${escapeHtml(model.parent_model_id)}` : ago(model.created_at)}</p><div class="model-score"><span><small>REGIME</small><strong>${escapeHtml(config.mode || '—')}</strong></span><span><small>ACURÁCIA TESTE</small><strong>${((evaluation.test_policy_accuracy || 0) * 100).toFixed(1)}%</strong></span><span><small>PARÂMETROS</small><strong>${formatNumber(evaluation.parameters)}</strong></span></div><div class="model-actions"><button data-load-model="${model.id}" ${model.active ? 'disabled' : ''}>${model.active ? 'Em uso' : 'Carregar'}</button><button data-finetune-model="${model.id}">Ajustar</button><a href="/api/models/${model.id}/download">Exportar</a><button class="destructive" data-delete-model="${model.id}">Apagar</button></div></article>`;
  }).join('') || '<div class="empty-library">Nenhum checkpoint ainda. Seu primeiro modelo aparecerá aqui.</div>';
  $$('[data-load-model]').forEach((button) => button.addEventListener('click', () => loadModel(button.dataset.loadModel)));
  $$('[data-finetune-model]').forEach((button) => button.addEventListener('click', () => { changeView('training'); $('#baseModelSelect').value = button.dataset.finetuneModel; $('#trainingForm').elements.name.value = `${state.models.find((m) => m.id === button.dataset.finetuneModel)?.config?.name || 'Modelo'} FT`; toast('Modelo base selecionado para ajuste fino.'); }));
  $$('[data-delete-model]').forEach((button) => button.addEventListener('click', () => deleteModel(button.dataset.deleteModel)));
}

async function loadModel(id) {
  try { await api(`/api/models/${encodeURIComponent(id)}/load`, { method: 'POST' }); toast('Checkpoint carregado na arena.'); await loadDashboard(true); }
  catch (error) { toast(error.message, 'error'); }
}

async function deleteModel(id) {
  const model = state.models.find((item) => item.id === id);
  const label = model?.config?.name || id;
  if (!window.confirm(`Apagar permanentemente o modelo “${label}”? Os replays serão preservados.`)) return;
  try { await api(`/api/models/${encodeURIComponent(id)}`, { method: 'DELETE' }); toast('Modelo apagado.'); await loadDashboard(true); }
  catch (error) { toast(error.message, 'error'); }
}

async function importModel(file) {
  if (!file) return;
  const form = new FormData(); form.append('file', file);
  try { await api('/api/models/import', { method: 'POST', body: form }); toast('Modelo base importado.'); await loadDashboard(true); }
  catch (error) { toast(error.message, 'error'); }
  finally { $('#modelInput').value = ''; }
}

function wireEvents() {
  $$('[data-view-link]').forEach((button) => button.addEventListener('click', (event) => { event.preventDefault(); changeView(button.dataset.viewLink); }));
  $('.mobile-menu').addEventListener('click', () => $('.sidebar').classList.toggle('open'));
  $('#refreshButton').addEventListener('click', () => loadDashboard());
  $('#browsePgn').addEventListener('click', () => $('#pgnInput').click());
  $('#pgnInput').addEventListener('change', (event) => uploadFiles(event.target.files));
  const zone = $('#uploadZone');
  ['dragenter', 'dragover'].forEach((name) => zone.addEventListener(name, (event) => { event.preventDefault(); zone.classList.add('dragging'); }));
  ['dragleave', 'drop'].forEach((name) => zone.addEventListener(name, (event) => { event.preventDefault(); zone.classList.remove('dragging'); }));
  zone.addEventListener('drop', (event) => uploadFiles(event.dataTransfer.files));
  $('#datasetSearch').addEventListener('input', (event) => renderDatasets(event.target.value));
  $('#trainingForm').addEventListener('submit', startTraining);
  $('#selectAllDatasets').addEventListener('click', () => $$('#trainingDatasets input[type="checkbox"]').forEach((input) => { input.checked = true; }));
  $('#clearDatasets').addEventListener('click', () => $$('#trainingDatasets input[type="checkbox"]').forEach((input) => { input.checked = false; }));
  $('#stopTrainingButton').addEventListener('click', async () => { try { await api('/api/training/stop', { method: 'POST' }); toast('Parada segura solicitada.'); } catch (error) { toast(error.message, 'error'); } });
  $('#newGameButton').addEventListener('click', newGame);
  $$('#colorChoice button').forEach((button) => button.addEventListener('click', () => { state.color = button.dataset.color; $$('#colorChoice button').forEach((b) => b.classList.toggle('active', b === button)); newGame(); }));
  $('#arenaStrength').addEventListener('change', newGame);
  $('#flipBoard').addEventListener('click', () => { state.orientation = state.orientation === 'white' ? 'black' : 'white'; renderBoard(); });
  $('#championshipForm').addEventListener('submit', startTournament);
  $('#stopChampionship').addEventListener('click', async () => { await api('/api/tournament/stop', { method: 'POST' }); toast('O campeonato encerrará após a partida atual.'); });
  $('#mentorForm').addEventListener('submit', saveGuidedExample);
  $('#resetMentor').addEventListener('click', resetMentor);
  $('#mentorForm').elements.priority.addEventListener('input', (event) => { $('#priorityValue').textContent = `${event.target.value}×`; });
  $('#replayFilter').addEventListener('change', renderReplayList);
  $('#replayStart').addEventListener('click', () => { state.replayIndex = 0; renderReplay(); });
  $('#replayPrev').addEventListener('click', () => replayStep(-1));
  $('#replayPlay').addEventListener('click', toggleReplay);
  $('#replayNext').addEventListener('click', () => replayStep(1));
  $('#replayEnd').addEventListener('click', () => { if (state.replay) { state.replayIndex = state.replay.moves.length; renderReplay(); stopReplay(); } });
  $('#importModel').addEventListener('click', () => $('#modelInput').click());
  $('#modelInput').addEventListener('change', (event) => importModel(event.target.files[0]));
  $('#refreshNetwork').addEventListener('click', loadNetwork);
  $('#networkCanvas').addEventListener('mousemove', networkHover);
  $('#networkCanvas').addEventListener('mouseleave', () => $('#networkTooltip').style.display = 'none');
  window.addEventListener('resize', () => { drawMetricChart($('#overviewChart'), state.training?.metrics || []); if (state.view === 'network') drawNetwork(); });
}

wireEvents();
changeView(location.hash.slice(1) || 'overview');
loadDashboard(true);
setInterval(pollTraining, 1400);
setInterval(pollTournament, 1700);
