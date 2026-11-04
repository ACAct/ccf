(function () {
  'use strict';

  const { ApiClient, loadSettings, saveSettings } = window.JiuwenAPI;
  let client = new ApiClient();
  let chartId = 0;

  const state = {
    route: 'tasks',
    tasks: [],
    routeSelections: { monitor: null, results: null, paper: null },
    selectedTaskId: null,
    experiment: null,
    experimentTaskId: null,
    experimentLoading: false,
    experimentRequest: 0,
    paper: null,
    paperTaskId: null,
    taskFilter: 'all',
    taskSearch: '',
    autoRefresh: true,
    autoScroll: true,
    pollTimer: null,
    lastRefresh: null,
    taskError: ''
  };

  const pageTitles = { tasks: '任务中心', monitor: '实验监控', results: '结果分析', paper: '论文展示' };
  const stageOrder = ['topic', 'research', 'method', 'experiment', 'paper'];
  const stages = [
    { id: 'topic', name: '选题拆解', description: '明确问题与研究假设', icon: 'layers' },
    { id: 'research', name: '文献调研', description: '检索、精读与证据整理', icon: 'book' },
    { id: 'method', name: '方法设计', description: '形成方法与实验方案', icon: 'flask' },
    { id: 'experiment', name: '实验验证', description: '执行实验并记录结果', icon: 'monitor' },
    { id: 'paper', name: '论文撰写', description: '生成论文与分析结论', icon: 'paper' }
  ];

  function icon(name, className = '') {
    return `<svg${className ? ` class="${className}"` : ''} aria-hidden="true"><use href="#icon-${name}"/></svg>`;
  }

  function escapeHTML(value) {
    return String(value ?? '').replace(/[&<>'"]/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char]));
  }

  function formatDate(value, withDate = false) {
    if (!value) return '—';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return date.toLocaleString('zh-CN', withDate
      ? { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false }
      : { hour: '2-digit', minute: '2-digit', hour12: false });
  }

  function formatPercent(value, digits = 1) {
    const number = Number(value);
    if (!Number.isFinite(number)) return '—';
    const percent = Math.abs(number) <= 1 ? number * 100 : number;
    return `${(percent * 1).toFixed(digits)}%`;
  }

  function formatNumber(value, digits = 4) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(digits) : '—';
  }

  function formatLR(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toExponential(1) : '—';
  }

  function formatDuration(seconds) {
    const number = Number(seconds);
    if (!Number.isFinite(number) || number < 0) return '—';
    if (number < 60) return `${Math.round(number)} 秒`;
    const hours = Math.floor(number / 3600);
    const minutes = Math.floor((number % 3600) / 60);
    const secs = Math.round(number % 60);
    return hours ? `${hours}h ${minutes}m` : `${minutes}m ${secs}s`;
  }

  function statusInfo(status) {
    const map = {
      running: ['运行中', 'running'], completed: ['已完成', 'completed'],
      queued: ['排队中', 'queued'], paused: ['已暂停', 'paused'],
      failed: ['异常', 'failed'], stopped: ['已停止', 'stopped'],
      pending: ['等待中', 'queued'], researching: ['运行中', 'running'], writing: ['运行中', 'running']
    };
    const [label, className] = map[status] || ['未开始', 'default'];
    return { label, className };
  }

  function statusPill(status) {
    const info = statusInfo(status);
    return `<span class="status-pill status-${info.className}">${info.label}</span>`;
  }

  function currentTask() {
    return state.tasks.find(task => task.id === state.selectedTaskId) || null;
  }

  function currentExperimentId(task = currentTask()) {
    return task ? task.experiment_id || task.experimentId || task.id : '';
  }

  function toast(message, type = 'success') {
    const region = document.getElementById('toast-region');
    const element = document.createElement('div');
    element.className = `toast ${type}`;
    element.innerHTML = `${icon(type === 'error' ? 'alert' : 'check')}<span>${escapeHTML(message)}</span>`;
    region.appendChild(element);
    setTimeout(() => element.remove(), 4200);
  }

  function loadingState(text = '正在读取实验数据…') {
    return `<div class="loading-state"><div><div class="spinner"></div>${escapeHTML(text)}</div></div>`;
  }

  function emptyState(iconName, title, description, action = '') {
    return `<div class="card empty-state"><div class="empty-state-icon">${icon(iconName)}</div><h2>${escapeHTML(title)}</h2><p>${escapeHTML(description)}</p>${action}</div>`;
  }

  function updateSourceUI() {
    const isDemo = client.mode === 'demo';
    const button = document.getElementById('data-source-button');
    button.querySelector('.status-dot').className = `status-dot${isDemo ? ' demo' : ''}`;
    document.getElementById('data-source-label').textContent = isDemo ? '演示数据' : '真实数据';
    document.getElementById('system-status-text').textContent = isDemo ? '演示工作区已就绪' : '真实接口已配置';
  }

  async function loadTasks() {
    state.taskError = '';
    try {
      state.tasks = await client.listTasks();
      if (state.selectedTaskId && !state.tasks.some(task => task.id === state.selectedTaskId)) state.selectedTaskId = null;
      document.getElementById('system-status-text').textContent = client.mode === 'demo' ? '演示工作区已就绪' : '任务接口已连接';
    } catch (error) {
      state.tasks = [];
      state.taskError = error.message;
      document.getElementById('system-status-text').textContent = '任务接口连接失败';
      toast(`任务列表读取失败：${error.message}`, 'error');
    }
    document.getElementById('nav-task-count').textContent = state.tasks.length;
    renderTaskCenter();
  }

  function renderTaskCenter() {
    const total = state.tasks.length;
    const running = state.tasks.filter(task => ['running', 'researching', 'writing'].includes(task.status)).length;
    const completed = state.tasks.filter(task => task.status === 'completed').length;
    const queued = state.tasks.filter(task => ['queued', 'pending'].includes(task.status)).length;
    const stats = [
      { label: '全部科研任务', value: total, icon: 'tasks', color: '', extra: '累计创建' },
      { label: '正在运行', value: running, icon: 'monitor', color: 'blue', extra: running ? 'Agent 执行中' : '当前空闲' },
      { label: '已完成任务', value: completed, icon: 'check', color: 'teal', extra: '结果可查看' },
      { label: '等待调度', value: queued, icon: 'clock', color: 'orange', extra: '按顺序执行' }
    ];
    document.getElementById('task-stats').innerHTML = stats.map(item => `
      <article class="stat-card"><div class="stat-icon ${item.color}">${icon(item.icon)}</div><div class="stat-copy"><div class="stat-value">${item.value}</div><div class="stat-label">${item.label}</div></div><span class="stat-extra">${item.extra}</span></article>
    `).join('');

    document.getElementById('global-pipeline').innerHTML = stages.map((stage, index) => `
      <div class="pipeline-item"><span class="pipeline-number">0${index + 1}</span><div class="pipeline-copy"><strong>${stage.name}</strong><span>${stage.description}</span></div></div>
    `).join('');

    const list = document.getElementById('task-list');
    if (state.taskError) {
      list.innerHTML = `<tr class="empty-row"><td colspan="6">${icon('alert')}<br>任务接口暂不可用：${escapeHTML(state.taskError)}<br><button class="button button-secondary button-small" data-action="retry-tasks" style="margin-top:12px">重新连接</button></td></tr>`;
    } else {
      const keyword = state.taskSearch.trim().toLowerCase();
      const filtered = state.tasks.filter(task => {
        const matchesStatus = state.taskFilter === 'all' || task.status === state.taskFilter || (state.taskFilter === 'running' && ['researching', 'writing'].includes(task.status));
        const matchesKeyword = !keyword || `${task.title} ${task.research_direction} ${task.id}`.toLowerCase().includes(keyword);
        return matchesStatus && matchesKeyword;
      });
      list.innerHTML = filtered.length ? filtered.map(task => {
        const stage = stages.find(item => item.id === task.stage) || stages[0];
        return `<tr>
          <td class="task-name-cell"><button class="task-name-button" data-open-task="${escapeHTML(task.id)}"><strong title="${escapeHTML(task.title)}">${escapeHTML(task.title)}</strong><small>${escapeHTML(task.id)}</small></button></td>
          <td><span class="direction-tag">${escapeHTML(task.research_direction || 'Agent 科研')}</span></td>
          <td>${statusPill(task.status)}</td>
          <td><span class="stage-cell">${icon(stage.icon)}${stage.name}</span></td>
          <td class="time-cell">${formatDate(task.updated_at || task.created_at, true)}</td>
          <td><button class="row-action" data-open-task="${escapeHTML(task.id)}" title="进入实验监控">${icon('chevron')}</button></td>
        </tr>`;
      }).join('') : `<tr class="empty-row"><td colspan="6">${total ? '没有符合当前筛选条件的任务' : '暂无科研任务，点击右上角“新建科研任务”开始。'}</td></tr>`;
      document.getElementById('task-list-footer').innerHTML = `<span>共 ${filtered.length} 个任务${filtered.length !== total ? `（全部 ${total} 个）` : ''}</span><span>${client.mode === 'demo' ? '演示数据仅用于界面预览' : '数据来自任务 API'}</span>`;
    }
  }

  function selectRouteTask(route, explicitId) {
    if (explicitId && state.tasks.some(task => task.id === explicitId)) state.routeSelections[route] = explicitId;
    if (!state.routeSelections[route] || !state.tasks.some(task => task.id === state.routeSelections[route])) {
      let preferred;
      if (route === 'monitor') preferred = state.tasks.find(task => task.status === 'running') || state.tasks[0];
      else preferred = state.tasks.find(task => task.status === 'completed') || state.tasks[0];
      state.routeSelections[route] = preferred ? preferred.id : null;
    }
    state.selectedTaskId = state.routeSelections[route];
  }

  async function router() {
    clearInterval(state.pollTimer);
    const raw = location.hash.replace(/^#\/?/, '');
    const [routePart, idPart] = raw.split('/');
    const route = pageTitles[routePart] ? routePart : 'tasks';
    state.route = route;
    const explicitId = idPart ? decodeURIComponent(idPart) : null;
    if (route !== 'tasks') selectRouteTask(route, explicitId);
    else if (explicitId) state.selectedTaskId = explicitId;

    document.querySelectorAll('.view').forEach(view => view.classList.toggle('active', view.id === `view-${route}`));
    document.querySelectorAll('.nav-item').forEach(item => item.classList.toggle('active', item.dataset.route === route));
    document.getElementById('page-title').textContent = pageTitles[route];
    document.title = `${pageTitles[route]} · Jiuwen Research`;
    document.getElementById('sidebar').classList.remove('open');

    if (route === 'tasks') {
      renderTaskCenter();
      return;
    }
    if (!currentTask()) {
      const html = emptyState(route === 'paper' ? 'paper' : 'flask', '还没有可查看的科研任务', '请先创建一个科研任务，或直接返回任务中心查看演示任务。', '<button class="button button-primary" data-go-route="tasks">返回任务中心</button>');
      document.getElementById(`${route}-content`).innerHTML = html;
      return;
    }
    if (route === 'paper') await loadPaper();
    else await loadExperiment();
    setupPolling();
  }

  function taskSelector() {
    return `<div class="toolbar-select-wrap"><label>当前实验</label><select class="experiment-select" data-experiment-selector aria-label="切换实验">${state.tasks.map(task => `<option value="${escapeHTML(task.id)}"${task.id === state.selectedTaskId ? ' selected' : ''}>${escapeHTML(task.title)}</option>`).join('')}</select></div>`;
  }

  function parseLogLine(line) {
    let data = null;
    try {
      const parsed = JSON.parse(line);
      if (parsed && typeof parsed === 'object') data = parsed;
    } catch (_) { /* Parse key-value logs below. */ }
    if (!data) {
      data = {};
      const pattern = /(epoch|train_loss|train_acc|val_loss|val_acc|val_f1|lr|duration_seconds|elapsed_seconds|duration)["']?\s*[:=]\s*["']?(-?[\d.]+(?:e[+-]?\d+)?)/gi;
      let match;
      while ((match = pattern.exec(line))) data[match[1].toLowerCase()] = Number(match[2]);
      const timestamp = line.match(/\[?(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})/);
      if (timestamp) data.timestamp = timestamp[1];
    }
    const keys = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_f1', 'lr', 'duration_seconds', 'elapsed_seconds', 'duration'];
    keys.forEach(key => {
      if (data[key] !== undefined && data[key] !== null && data[key] !== '') data[key] = Number(data[key]);
    });
    if (data.epoch === undefined && data.train_loss === undefined && data.val_loss === undefined) return null;
    data.raw = line;
    return data;
  }

  function parseLogs(text) {
    return String(text || '').split(/\r?\n/).filter(line => line.trim()).map(parseLogLine).filter(Boolean);
  }

  function renderStageStepper(task) {
    let activeStage = stageOrder.indexOf(task.stage);
    if (activeStage < 0) activeStage = task.status === 'completed' ? 4 : 0;
    return `<div class="card stage-stepper">${stages.map((stage, index) => {
      const done = task.status === 'completed' || index < activeStage;
      const current = task.status !== 'completed' && index === activeStage;
      return `<div class="stage-step ${done ? 'done' : current ? 'current' : ''}"><span class="stage-step-dot">${done ? icon('check') : index + 1}</span><span>${stage.name}</span></div>`;
    }).join('')}</div>`;
  }

  function lineChart(series, labels, options = {}) {
    const validSeries = series.filter(item => item.values.some(value => Number.isFinite(Number(value))));
    if (!validSeries.length || !labels.length) return '<div class="curve-missing">暂无可绘制的逐轮实验数据</div>';
    const width = 720, height = 270;
    const margin = { left: 48, right: 18, top: 18, bottom: 35 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const values = validSeries.flatMap(item => item.values.map(Number).filter(Number.isFinite));
    let min = options.min !== undefined ? options.min : Math.min(...values);
    let max = options.max !== undefined ? options.max : Math.max(...values);
    if (options.min === undefined) min = Math.max(0, min - (max - min || Math.max(Math.abs(max), 1)) * .15);
    if (options.max === undefined) max += (max - min || Math.max(Math.abs(max), 1)) * .12;
    if (max === min) max = min + 1;
    const x = index => margin.left + (labels.length === 1 ? plotWidth / 2 : index * plotWidth / (labels.length - 1));
    const y = value => margin.top + (max - Number(value)) * plotHeight / (max - min);
    const grids = Array.from({ length: 5 }, (_, index) => {
      const value = max - index * (max - min) / 4;
      const yy = margin.top + index * plotHeight / 4;
      const label = options.percent ? `${Math.round(value * 100)}%` : value.toFixed(options.decimals ?? 2);
      return `<line x1="${margin.left}" y1="${yy}" x2="${width - margin.right}" y2="${yy}" stroke="#edf0f4" stroke-width="1"/><text x="${margin.left - 9}" y="${yy + 3}" text-anchor="end">${label}</text>`;
    }).join('');
    const labelStep = Math.max(1, Math.ceil(labels.length / 6));
    const xLabels = labels.map((label, index) => (index % labelStep === 0 || index === labels.length - 1)
      ? `<text x="${x(index)}" y="${height - 10}" text-anchor="middle">${escapeHTML(label)}</text>` : '').join('');
    const id = `chart-gradient-${chartId++}`;
    const paths = validSeries.map((item, seriesIndex) => {
      const points = item.values.map((value, index) => `${x(index).toFixed(2)},${y(value).toFixed(2)}`).join(' ');
      const area = seriesIndex === 0 && options.area
        ? `<defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${item.color}" stop-opacity=".18"/><stop offset="1" stop-color="${item.color}" stop-opacity="0"/></linearGradient></defs><polygon points="${margin.left},${margin.top + plotHeight} ${points} ${x(item.values.length - 1)},${margin.top + plotHeight}" fill="url(#${id})" stroke="none"/>` : '';
      const dots = item.values.length <= 35 ? item.values.map((value, index) => `<circle cx="${x(index)}" cy="${y(value)}" r="2.5" fill="#fff" stroke="${item.color}" stroke-width="1.5"/>`).join('') : '';
      return `${area}<polyline points="${points}" fill="none" stroke="${item.color}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>${dots}`;
    }).join('');
    return `<svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="实验趋势图">${grids}${xLabels}${paths}</svg><div class="chart-legend">${validSeries.map(item => `<span><i style="background:${item.color}"></i>${escapeHTML(item.name)}</span>`).join('')}</div>`;
  }

  function resourceItem(iconName, label, percent, detail, color = '') {
    const safePercent = Math.min(100, Math.max(0, Number(percent) || 0));
    return `<div class="resource-item"><div class="resource-item-top">${icon(iconName)}<span>${label}</span><strong>${Math.round(safePercent)}%</strong></div><div class="resource-bar"><span style="width:${safePercent}%;${color ? `background:${color}` : ''}"></span></div><div class="resource-sub"><span>${detail}</span><span>${safePercent >= 90 ? '接近上限' : '运行正常'}</span></div></div>`;
  }

  function monitorActions(task) {
    const buttons = [];
    if (task.status === 'running') buttons.push(`<button class="button button-secondary button-small" data-task-action="pause">${icon('pause')}暂停</button>`);
    if (task.status === 'paused') buttons.push(`<button class="button button-primary button-small" data-task-action="resume">${icon('play')}继续</button>`);
    if (['queued', 'pending', 'stopped'].includes(task.status)) buttons.push(`<button class="button button-primary button-small" data-task-action="start">${icon('play')}启动</button>`);
    if (['running', 'paused', 'queued', 'pending'].includes(task.status)) buttons.push(`<button class="button button-danger button-small" data-task-action="stop">${icon('stop')}停止</button>`);
    return buttons.join('');
  }

  function renderMonitor() {
    const task = currentTask();
    const experiment = state.experiment;
    if (!task || !experiment) return;
    const config = experiment.config || {};
    const history = parseLogs(experiment.log);
    const latest = history[history.length - 1] || task.latest_metrics || {};
    const epochs = Number(config.epochs || task.epochs || 0);
    const currentEpoch = Number(latest.epoch ?? task.current_epoch ?? 0);
    if (Number.isFinite(currentEpoch)) task.current_epoch = currentEpoch;
    const progress = epochs ? Math.min(100, currentEpoch / epochs * 100) : Number(task.progress) || 0;
    const resources = experiment.resources || task.resources || {};
    const rawLines = String(experiment.log || '').split(/\r?\n/).filter(line => line.trim());
    const visibleLines = rawLines.slice(-100);
    const totalSeconds = history.reduce((sum, item) => sum + Number(item.duration_seconds || item.duration || 0), 0) || resources.elapsed_seconds;
    const warnings = experiment.warnings || [];
    const configItems = [
      ['基础模型', config.model || config.model_name || task.model],
      ['实验策略', config.strategy || task.strategy],
      ['数据比例', config.data_ratio !== undefined ? formatPercent(config.data_ratio, 0) : '—'],
      ['训练轮次', epochs || '—'],
      ['批次大小', config.batch_size || '—'],
      ['随机种子', config.seed ?? '—']
    ];
    const metrics = [
      ['Train Loss', formatNumber(latest.train_loss), '训练集损失', 'monitor', ''],
      ['Train Accuracy', formatPercent(latest.train_acc), '训练集准确率', 'results', 'metric-trend'],
      ['Validation Loss', formatNumber(latest.val_loss), '验证集损失', 'monitor', ''],
      ['Val Accuracy', formatPercent(latest.val_acc), '验证集准确率', 'results', 'metric-trend'],
      ['Val Macro F1', formatPercent(latest.val_f1), '验证集综合指标', 'check', 'metric-trend']
    ];
    const labels = history.map(item => `E${item.epoch ?? ''}`);
    const chart = lineChart([
      { name: 'Train Loss', color: '#635bdf', values: history.map(item => item.train_loss) },
      { name: 'Validation Loss', color: '#159c8c', values: history.map(item => item.val_loss) }
    ], labels, { area: true, decimals: 2 });

    document.getElementById('monitor-content').innerHTML = `
      <div class="page-toolbar"><div class="page-toolbar-left">${taskSelector()}</div><div class="page-toolbar-right"><label class="refresh-label"><input type="checkbox" id="auto-refresh"${state.autoRefresh ? ' checked' : ''}>5 秒自动刷新</label><button class="button button-secondary button-small" data-action="refresh-experiment">${icon('refresh')}刷新</button></div></div>
      ${task.error ? `<div class="error-banner">${icon('alert')}<span>${escapeHTML(task.error)}</span></div>` : ''}
      ${warnings.length ? `<div class="warning-banner">${icon('alert')}<span>部分实验文件暂未读取：${escapeHTML(warnings.join('；'))}</span></div>` : ''}
      <section class="card experiment-banner"><div class="experiment-title"><div class="experiment-title-row"><h2>${escapeHTML(task.title)}</h2>${statusPill(task.status)}</div><p>${escapeHTML(currentExperimentId(task))} · ${escapeHTML(task.research_direction || '')}</p></div><div class="experiment-actions">${monitorActions(task)}</div></section>
      ${renderStageStepper(task)}
      <section class="card config-card"><div class="card-header"><div><h3>实验配置</h3><p>config.json · 参数来源可追溯</p></div><span class="soft-badge">${escapeHTML(config.strategy || task.strategy || 'Baseline')}</span></div><div class="config-grid">${configItems.map(([label, value]) => `<div class="config-item"><span>${label}</span><strong title="${escapeHTML(value ?? '—')}">${escapeHTML(value ?? '—')}</strong></div>`).join('')}</div><div class="progress-block"><div class="progress-meta"><span>训练进度 <strong>Epoch ${currentEpoch} / ${epochs || '—'}</strong></span><span>LR ${formatLR(latest.lr)} · 最近耗时 ${formatDuration(latest.duration_seconds || latest.duration)}</span></div><div class="progress-track"><div class="progress-fill" style="width:${progress}%"></div></div></div></section>
      <div class="metric-grid monitor-metrics">${metrics.map(([label, value, foot, iconName, trend]) => `<article class="card metric-card"><div class="metric-label"><span>${label}</span>${icon(iconName)}</div><div class="metric-value">${value}</div><div class="metric-foot ${trend}">${foot}</div></article>`).join('')}</div>
      <div class="monitor-grid"><section class="card chart-card"><div class="card-header"><div><h3>损失变化</h3><p>按 Epoch 实时绘制最近训练日志</p></div><span class="soft-badge">${history.length} 个数据点</span></div><div class="chart-container">${chart}</div></section>
      <section class="card resource-card"><div class="card-header"><div><h3>资源消耗</h3><p>任务执行节点实时状态</p></div>${icon('gpu')}</div><div class="resource-list">
        ${resourceItem('gpu', 'GPU 利用率', resources.gpu_percent ?? resources.gpu_utilization ?? 0, '计算资源')}
        ${resourceItem('database', 'GPU 显存', (Number(resources.gpu_memory_used || 0) / Number(resources.gpu_memory_total || 1)) * 100, `${resources.gpu_memory_used ?? 0} / ${resources.gpu_memory_total ?? '—'} GB`, '#3578e5')}
        ${resourceItem('monitor', 'CPU 利用率', resources.cpu_percent ?? resources.cpu_utilization ?? 0, `${resources.ram_used ?? '—'} / ${resources.ram_total ?? '—'} GB 内存`, '#159c8c')}
      </div><div class="resource-summary"><span>累计训练耗时</span><strong>${formatDuration(totalSeconds)}</strong></div></section></div>
      <section class="card log-card"><div class="card-header"><div><h3>运行日志</h3><p>logs/${escapeHTML(currentExperimentId(task))}.log${state.lastRefresh ? ` · 更新于 ${formatDate(state.lastRefresh)}` : ''}</p></div><div class="log-header-actions"><label><input type="checkbox" id="auto-scroll"${state.autoScroll ? ' checked' : ''}>自动滚动</label>${icon('terminal')}</div></div><pre class="log-window" id="log-window">${visibleLines.length ? visibleLines.map(line => `<div class="log-line${/error|exception|traceback/i.test(line) ? ' error' : ''}">${escapeHTML(line)}</div>`).join('') : '<div class="log-empty">任务尚未产生训练日志</div>'}</pre></section>
    `;
    if (state.autoScroll) requestAnimationFrame(() => {
      const logWindow = document.getElementById('log-window');
      if (logWindow) logWindow.scrollTop = logWindow.scrollHeight;
    });
  }

  async function loadExperiment() {
    const task = currentTask();
    if (!task || !['monitor', 'results'].includes(state.route)) return;
    const requestId = ++state.experimentRequest;
    const route = state.route;
    state.experimentLoading = true;
    const container = document.getElementById(`${route}-content`);
    if (state.experimentTaskId !== task.id) container.innerHTML = loadingState();
    try {
      const experiment = await client.getExperiment(task);
      if (requestId !== state.experimentRequest) return;
      state.experiment = experiment;
      state.experimentTaskId = task.id;
      state.lastRefresh = new Date().toISOString();
      if (state.route === 'monitor') renderMonitor();
      else if (state.route === 'results') renderResults();
    } catch (error) {
      if (requestId !== state.experimentRequest) return;
      state.experiment = null;
      state.experimentTaskId = task.id;
      container.innerHTML = emptyState('alert', '实验数据读取失败', error.message, '<button class="button button-secondary" data-action="refresh-experiment">重新读取</button>');
      toast(error.message, 'error');
    } finally {
      if (requestId === state.experimentRequest) state.experimentLoading = false;
    }
  }

  function metricValue(metrics, split, name, average = 'macro') {
    const group = metrics && (metrics[split] || metrics[`${split}_metrics`]);
    if (!group) return NaN;
    const direct = group[name];
    if (direct && typeof direct === 'object') return Number(direct[average]);
    const flat = group[`${name}_${average}`];
    if (flat !== undefined) return Number(flat);
    if (name === 'accuracy') return Number(direct);
    return NaN;
  }

  function resultMetricCard(label, testValue, valValue, iconName) {
    const delta = Number(testValue) - Number(valValue);
    return `<article class="card result-metric-card"><div class="result-metric-top"><span>${label}</span>${icon(iconName)}</div><div class="result-metric-value">${formatPercent(testValue)}</div><div class="result-metric-foot"><span>验证集 ${formatPercent(valValue)}</span><strong>${Number.isFinite(delta) ? `${delta >= 0 ? '+' : ''}${(delta * 100).toFixed(1)} pp` : '—'}</strong></div></article>`;
  }

  function metricsTable(metrics) {
    const rows = [
      ['Accuracy', 'accuracy'], ['Precision', 'precision'], ['Recall', 'recall'], ['F1 Score', 'f1']
    ];
    return `<table class="data-table metric-table"><thead><tr><th>指标</th><th>Val Macro</th><th>Val Weighted</th><th>Test Macro</th><th>Test Weighted</th></tr></thead><tbody>${rows.map(([label, key]) => `<tr><td><strong>${label}</strong></td>${['val', 'test'].map(split => ['macro', 'weighted'].map(avg => {
      const value = key === 'accuracy' ? metricValue(metrics, split, key) : metricValue(metrics, split, key, avg);
      return `<td>${formatPercent(value)}</td>`;
    }).join('')).join('')}</tr>`).join('')}</tbody></table>`;
  }

  function renderResults() {
    const task = currentTask();
    const experiment = state.experiment;
    if (!task || !experiment) return;
    const metrics = experiment.metrics;
    if (!metrics) {
      document.getElementById('results-content').innerHTML = `<div class="page-toolbar"><div class="page-toolbar-left">${taskSelector()}</div><div class="page-toolbar-right"><button class="button button-secondary button-small" data-action="refresh-experiment">${icon('refresh')}刷新</button></div></div>${emptyState('results', '实验指标尚未生成', 'metrics.json 将在实验完成并校验后出现在此处。你可以先前往实验监控查看实时训练状态。', '<button class="button button-primary" data-go-route="monitor">查看实验监控</button>')}`;
      return;
    }
    const history = parseLogs(experiment.log);
    const labels = history.map(item => `E${item.epoch ?? ''}`);
    const lossChart = lineChart([
      { name: 'Train Loss', color: '#635bdf', values: history.map(item => item.train_loss) },
      { name: 'Validation Loss', color: '#159c8c', values: history.map(item => item.val_loss) }
    ], labels, { area: true, decimals: 2 });
    const accuracyChart = lineChart([
      { name: 'Train Accuracy', color: '#635bdf', values: history.map(item => item.train_acc) },
      { name: 'Validation Accuracy', color: '#159c8c', values: history.map(item => item.val_acc) }
    ], labels, { percent: true, decimals: 2 });
    const testF1 = metricValue(metrics, 'test', 'f1');
    const valF1 = metricValue(metrics, 'val', 'f1');
    const gap = (testF1 - valF1) * 100;
    const experimentId = currentExperimentId(task);
    const curveImage = type => experiment.curves && experiment.curves[type]
      ? `<img class="curve-image" data-curve="${type}" src="${escapeHTML(experiment.curves[type])}" alt="${type === 'loss' ? '损失' : '准确率'}曲线">`
      : type === 'loss' ? lossChart : accuracyChart;

    document.getElementById('results-content').innerHTML = `
      <div class="page-toolbar"><div class="page-toolbar-left"><div><h2>实验结果分析</h2><p>${escapeHTML(experimentId)} · 指标来源 results/runs/${escapeHTML(experimentId)}/metrics.json</p></div></div><div class="page-toolbar-right">${taskSelector()}<button class="button button-secondary button-small" data-action="refresh-experiment">${icon('refresh')}刷新</button></div></div>
      <div class="result-metrics">
        ${resultMetricCard('Test Accuracy', metricValue(metrics, 'test', 'accuracy'), metricValue(metrics, 'val', 'accuracy'), 'check')}
        ${resultMetricCard('Test Macro F1', testF1, valF1, 'results')}
        ${resultMetricCard('Test Weighted F1', metricValue(metrics, 'test', 'f1', 'weighted'), metricValue(metrics, 'val', 'f1', 'weighted'), 'results')}
        ${resultMetricCard('Test Macro Recall', metricValue(metrics, 'test', 'recall'), metricValue(metrics, 'val', 'recall'), 'monitor')}
      </div>
      <div class="results-grid"><section class="card chart-card"><div class="card-header"><div><h3>训练曲线</h3><p>损失与准确率变化趋势</p></div><div class="tab-group"><button class="tab-button active" data-chart-tab="loss">Loss</button><button class="tab-button" data-chart-tab="accuracy">Accuracy</button></div></div><div class="curve-panel active" data-chart-panel="loss">${curveImage('loss')}</div><div class="curve-panel" data-chart-panel="accuracy">${curveImage('accuracy')}</div><div class="file-path-note">${icon('paper')}<code>results/runs/${escapeHTML(experimentId)}/loss_curve.png · accuracy_curve.png</code></div></section>
      <section class="card"><div class="card-header"><div><h3>完整评估指标</h3><p>Macro / Weighted 多维度对照</p></div><span class="soft-badge">Val & Test</span></div><div class="table-wrap">${metricsTable(metrics)}</div><div class="file-path-note">${icon('database')}<code>metrics.json · 最近读取 ${formatDate(state.lastRefresh, true)}</code></div></section></div>
      <section class="card insight-card"><h3>结果速览</h3><div class="insight-list"><div class="insight-item"><span class="insight-number">01</span><div><strong>测试集综合表现</strong><p>测试集 Macro F1 为 ${formatPercent(testF1)}，Weighted F1 为 ${formatPercent(metricValue(metrics, 'test', 'f1', 'weighted'))}。</p></div></div><div class="insight-item"><span class="insight-number">02</span><div><strong>验证 / 测试差异</strong><p>Macro F1 相差 ${Math.abs(gap).toFixed(1)} 个百分点（测试集${gap >= 0 ? '更高' : '更低'}），${Math.abs(gap) > 5 ? '建议进一步检查数据分布与过拟合情况。' : '两组评估表现较为接近。'}</p></div></div><div class="insight-item"><span class="insight-number">03</span><div><strong>实验可追溯性</strong><p>配置、逐轮日志、指标文件和曲线均按实验 ID 归档，可回到实验监控交叉核验。</p></div></div></div></section>
    `;
    bindCurveImages();
  }

  function bindCurveImages() {
    document.querySelectorAll('img[data-curve]').forEach(image => {
      image.addEventListener('error', () => {
        const fallback = document.createElement('div');
        fallback.className = 'curve-missing';
        fallback.innerHTML = `<div>${icon('results')}未找到 ${escapeHTML(image.dataset.curve)}_curve.png<br>请确认实验曲线已生成，且文件根路径配置正确。</div>`;
        image.replaceWith(fallback);
      }, { once: true });
    });
  }

  async function loadPaper() {
    const task = currentTask();
    if (!task) return;
    const container = document.getElementById('paper-content');
    container.innerHTML = loadingState('正在读取论文内容与生成状态…');
    try {
      state.paper = await client.getPaper(task.id);
      state.paperTaskId = task.id;
      renderPaper();
    } catch (error) {
      state.paper = null;
      state.paperTaskId = task.id;
      container.innerHTML = `<div class="page-toolbar"><div class="page-toolbar-left">${taskSelector()}</div></div>${emptyState('paper', '论文尚未生成', error.message || '完成实验验证与结果分析后，Writing Agent 会在这里生成论文。', '<button class="button button-primary" data-go-route="monitor">查看任务进度</button>')}`;
    }
  }

  function renderPaper() {
    const task = currentTask();
    const paper = state.paper;
    if (!task || !paper) return;
    const sections = Array.isArray(paper.sections) ? paper.sections : [];
    const pdfUrl = paper.pdf_url || paper.pdfUrl || '';
    const preview = pdfUrl
      ? `<iframe class="pdf-frame" src="${escapeHTML(pdfUrl)}" title="论文 PDF 预览"></iframe>`
      : `<article class="paper-document"><div class="paper-venue">${escapeHTML(paper.venue || 'ICLR Short Paper')}</div><h1>${escapeHTML(paper.title || task.title)}</h1><div class="paper-authors">${escapeHTML(paper.authors || 'Jiuwen Research Team')}</div><section class="paper-abstract"><h2>Abstract</h2><p>${escapeHTML(paper.abstract || 'The abstract will appear here after paper generation.')}</p></section><div class="paper-keywords"><strong>Keywords:</strong> ${escapeHTML((paper.keywords || ['Research Agents', 'Self-Evolution', 'Scientific Discovery']).join(', '))}</div><div class="paper-sections">${sections.map(section => `<section class="paper-section"><h3>${escapeHTML(section.title || section.name)}</h3><p>${escapeHTML(section.summary || section.content || '')}</p></section>`).join('')}</div></article>`;
    const downloadButtons = pdfUrl
      ? `<a class="button button-primary" href="${escapeHTML(pdfUrl)}" target="_blank" rel="noopener">${icon('download')}下载论文 PDF</a><a class="button button-secondary" href="${escapeHTML(pdfUrl)}" target="_blank" rel="noopener">${icon('external')}在新窗口打开</a>`
      : `<button class="button button-primary" data-action="paper-unavailable">${icon('download')}下载论文 PDF</button><button class="button button-secondary" data-action="refresh-paper">${icon('refresh')}检查生成状态</button>`;

    document.getElementById('paper-content').innerHTML = `
      <div class="page-toolbar"><div class="page-toolbar-left"><div><h2>科研成果</h2><p>Writing Agent 生成内容 · 所有实验数字应可追溯至真实结果</p></div></div><div class="page-toolbar-right">${taskSelector()}${statusPill(paper.status || task.status)}</div></div>
      <div class="paper-layout"><section class="card paper-preview-card"><div class="paper-toolbar"><div class="paper-toolbar-left">${icon('paper')}<span>${pdfUrl ? 'PDF 原文预览' : '论文内容预览'}</span></div><span class="soft-badge">${escapeHTML(paper.language || 'English')}</span></div>${preview}</section>
      <aside class="paper-sidebar"><section class="card"><div class="card-header"><div><h3>论文信息</h3><p>生成产物与版本信息</p></div>${icon('paper')}</div><div class="paper-info-list"><div class="paper-info-item"><span>关联任务</span><strong>${escapeHTML(task.title)}</strong></div><div class="paper-info-item"><span>论文模板</span><strong>${escapeHTML(paper.venue || 'ICLR Short Paper')}</strong></div><div class="paper-info-item"><span>最近更新</span><strong>${formatDate(paper.updated_at || task.updated_at, true)}</strong></div><div class="paper-info-item"><span>实验 ID</span><strong>${escapeHTML(currentExperimentId(task))}</strong></div></div><div class="paper-actions">${downloadButtons}</div></section>
      <section class="card"><div class="card-header"><div><h3>交付检查</h3><p>论文生成质量核对</p></div></div><div class="paper-checklist"><div class="paper-check">${icon('check')}英文短论文结构</div><div class="paper-check">${icon('check')}ICLR 模板规范</div><div class="paper-check">${icon('check')}关联真实实验结果</div><div class="paper-check">${icon('check')}引用与参考文献待核验</div></div></section></aside></div>
    `;
  }

  function setupPolling() {
    clearInterval(state.pollTimer);
    const task = currentTask();
    if (state.route !== 'monitor' || !state.autoRefresh || !task || task.status !== 'running') return;
    state.pollTimer = setInterval(async () => {
      if (document.hidden || state.experimentLoading) return;
      await loadExperiment();
      setupPolling();
    }, 5000);
  }

  async function controlTask(action) {
    const task = currentTask();
    if (!task) return;
    if (action === 'stop' && !window.confirm(`确定停止任务“${task.title}”吗？已保存的日志与结果不会删除。`)) return;
    try {
      const updated = await client.controlTask(task.id, action);
      Object.assign(task, updated);
      task.updated_at = task.updated_at || new Date().toISOString();
      toast(({ pause: '任务已暂停', resume: '任务已继续', start: '任务已启动', stop: '任务已停止' })[action] || '任务状态已更新');
      await loadExperiment();
      renderTaskCenter();
      setupPolling();
    } catch (error) {
      toast(`任务操作失败：${error.message}`, 'error');
    }
  }

  function openModal(id) {
    const dialog = document.getElementById(id);
    document.getElementById('modal-overlay').hidden = false;
    document.body.style.overflow = 'hidden';
    if (!dialog.open) dialog.show();
    dialog.setAttribute('aria-modal', 'true');
    requestAnimationFrame(() => dialog.querySelector('input, select, textarea, button')?.focus());
  }

  function closeModal(id) {
    const dialog = typeof id === 'string' ? document.getElementById(id) : id;
    if (dialog?.open) dialog.close();
    if (!document.querySelector('.modal[open]')) {
      document.getElementById('modal-overlay').hidden = true;
      document.body.style.overflow = '';
    }
  }

  function openSettings() {
    const form = document.getElementById('settings-form');
    form.elements.mode.value = client.settings.mode;
    form.elements.apiBase.value = client.settings.apiBase || '';
    form.elements.assetBase.value = client.settings.assetBase || '';
    openModal('settings-dialog');
  }

  function navigate(route, taskId) {
    const next = `#/${route}${taskId ? `/${encodeURIComponent(taskId)}` : ''}`;
    if (location.hash === next) router();
    else location.hash = next;
  }

  function bindEvents() {
    window.addEventListener('hashchange', () => router().catch(error => toast(error.message, 'error')));
    document.getElementById('new-task-button').addEventListener('click', () => openModal('create-task-dialog'));
    document.getElementById('modal-overlay').addEventListener('click', () => document.querySelectorAll('.modal[open]').forEach(dialog => closeModal(dialog)));
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape') document.querySelectorAll('.modal[open]').forEach(dialog => closeModal(dialog));
    });
    document.getElementById('open-settings').addEventListener('click', openSettings);
    document.getElementById('data-source-button').addEventListener('click', openSettings);
    document.getElementById('mobile-menu').addEventListener('click', () => document.getElementById('sidebar').classList.toggle('open'));
    document.getElementById('task-search').addEventListener('input', event => { state.taskSearch = event.target.value; renderTaskCenter(); });
    document.getElementById('task-filter').addEventListener('change', event => { state.taskFilter = event.target.value; renderTaskCenter(); });
    document.getElementById('task-ratio').addEventListener('input', event => { document.getElementById('ratio-output').textContent = `${event.target.value}%`; });

    document.addEventListener('click', event => {
      const close = event.target.closest('[data-close]');
      if (close) closeModal(close.dataset.close);
      const taskButton = event.target.closest('[data-open-task]');
      if (taskButton) navigate('monitor', taskButton.dataset.openTask);
      const routeButton = event.target.closest('[data-go-route]');
      if (routeButton) navigate(routeButton.dataset.goRoute, state.selectedTaskId);
      const actionButton = event.target.closest('[data-action]');
      if (actionButton) {
        const action = actionButton.dataset.action;
        if (action === 'refresh-experiment') loadExperiment().then(setupPolling);
        if (action === 'retry-tasks') loadTasks().then(router);
        if (action === 'paper-unavailable') toast(client.mode === 'demo' ? '演示论文未附带真实 PDF 文件；切换真实数据源后可下载。' : '论文 PDF 尚未生成，请稍后重试。');
        if (action === 'refresh-paper') loadPaper();
      }
      const controlButton = event.target.closest('[data-task-action]');
      if (controlButton) controlTask(controlButton.dataset.taskAction);
      const chartTab = event.target.closest('[data-chart-tab]');
      if (chartTab) {
        const type = chartTab.dataset.chartTab;
        document.querySelectorAll('[data-chart-tab]').forEach(button => button.classList.toggle('active', button === chartTab));
        document.querySelectorAll('[data-chart-panel]').forEach(panel => panel.classList.toggle('active', panel.dataset.chartPanel === type));
      }
    });

    document.addEventListener('change', event => {
      if (event.target.matches('[data-experiment-selector]')) navigate(state.route, event.target.value);
      if (event.target.id === 'auto-refresh') { state.autoRefresh = event.target.checked; setupPolling(); }
      if (event.target.id === 'auto-scroll') state.autoScroll = event.target.checked;
    });

    document.getElementById('create-task-form').addEventListener('submit', async event => {
      event.preventDefault();
      const form = event.currentTarget;
      const submit = document.getElementById('create-task-submit');
      const data = new FormData(form);
      const payload = {
        title: String(data.get('title')).trim(),
        research_direction: data.get('research_direction'),
        description: String(data.get('description') || '').trim(),
        config: {
          model: data.get('model'), strategy: data.get('strategy'),
          epochs: Number(data.get('epochs')), data_ratio: Number(data.get('data_ratio')) / 100
        }
      };
      submit.disabled = true;
      submit.innerHTML = '<span class="spinner" style="width:14px;height:14px;margin:0;border-color:rgba(255,255,255,.35);border-top-color:#fff"></span>正在创建';
      try {
        const task = await client.createTask(payload);
        state.tasks.unshift(task);
        state.routeSelections.monitor = task.id;
        state.routeSelections.results = task.id;
        state.routeSelections.paper = task.id;
        closeModal('create-task-dialog');
        form.reset();
        document.getElementById('ratio-output').textContent = '80%';
        document.getElementById('nav-task-count').textContent = state.tasks.length;
        renderTaskCenter();
        toast('科研任务已创建，正在进入任务监控。');
        navigate('monitor', task.id);
      } catch (error) {
        toast(`创建失败：${error.message}`, 'error');
      } finally {
        submit.disabled = false;
        submit.innerHTML = `${icon('plus')}创建并启动`;
      }
    });

    document.getElementById('settings-form').addEventListener('submit', async event => {
      event.preventDefault();
      const data = new FormData(event.currentTarget);
      const settings = {
        mode: data.get('mode'), apiBase: String(data.get('apiBase') || '').trim().replace(/\/$/, ''),
        assetBase: String(data.get('assetBase') || '').trim().replace(/\/$/, '')
      };
      saveSettings(settings);
      client = new ApiClient(settings);
      state.routeSelections = { monitor: null, results: null, paper: null };
      state.selectedTaskId = null;
      state.experiment = null;
      state.paper = null;
      closeModal('settings-dialog');
      updateSourceUI();
      await loadTasks();
      await router();
      toast(settings.mode === 'demo' ? '已切换到演示工作区。' : '数据源设置已保存，已重新连接真实接口。');
    });
  }

  async function init() {
    bindEvents();
    updateSourceUI();
    await loadTasks();
    await router();
  }

  init().catch(error => {
    console.error(error);
    toast(`页面初始化失败：${error.message}`, 'error');
  });
})();
