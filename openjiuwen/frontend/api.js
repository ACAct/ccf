(function () {
  'use strict';

  const SETTINGS_KEY = 'jiuwen-research-settings';
  const wait = (value, ms = 120) => new Promise(resolve => setTimeout(() => resolve(value), ms));

  class ApiError extends Error {
    constructor(message, status = 0) {
      super(message);
      this.name = 'ApiError';
      this.status = status;
    }
  }

  function loadSettings() {
    try {
      return Object.assign({ mode: 'demo', apiBase: '', assetBase: '' }, JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}'));
    } catch (_) {
      return { mode: 'demo', apiBase: '', assetBase: '' };
    }
  }

  function saveSettings(settings) {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  }

  const demoStartedAt = Date.now();
  const demoTasks = [
    {
      id: 'task_memory_001', experiment_id: 'exp_memory_lora_01',
      title: '面向长程科研任务的分层记忆机制研究', research_direction: 'Agent记忆引擎设计',
      description: '验证情景记忆与语义记忆分层组织对长程科研任务一致性的提升。',
      status: 'running', stage: 'experiment', current_epoch: 14, epochs: 30,
      model: 'Qwen2.5-7B-Instruct', strategy: 'LoRA', data_ratio: 0.8,
      created_at: '2026-11-03T09:30:00+08:00', updated_at: '2026-11-03T11:42:00+08:00',
      resources: { gpu_percent: 78, gpu_memory_used: 18.6, gpu_memory_total: 24, cpu_percent: 43, ram_used: 31.2, ram_total: 64 }
    },
    {
      id: 'task_evolve_002', experiment_id: 'exp_evolve_qlora_02',
      title: '基于反思轨迹的科研 Agent 自演进方法', research_direction: 'Agent自演进',
      description: '利用评审反馈构造反思轨迹，研究自演进对论文实验质量的影响。',
      status: 'completed', stage: 'paper', current_epoch: 30, epochs: 30,
      model: 'Qwen2.5-7B-Instruct', strategy: 'QLoRA', data_ratio: 1,
      created_at: '2026-11-02T14:10:00+08:00', updated_at: '2026-11-03T10:18:00+08:00',
      resources: { gpu_percent: 0, gpu_memory_used: 0, gpu_memory_total: 24, cpu_percent: 8, ram_used: 12.4, ram_total: 64 }
    },
    {
      id: 'task_context_003', experiment_id: 'exp_context_lora_03',
      title: '面向多源文献的上下文压缩与路由机制', research_direction: 'Agent上下文工程设计',
      description: '研究文献证据的动态压缩与上下文路由，缓解多源信息冲突。',
      status: 'queued', stage: 'research', current_epoch: 0, epochs: 24,
      model: 'Llama-3.1-8B-Instruct', strategy: 'LoRA', data_ratio: 0.6,
      created_at: '2026-11-03T10:45:00+08:00', updated_at: '2026-11-03T10:45:00+08:00',
      resources: { gpu_percent: 0, gpu_memory_used: 0, gpu_memory_total: 24, cpu_percent: 3, ram_used: 8.1, ram_total: 64 }
    },
    {
      id: 'task_swarm_004', experiment_id: 'exp_swarm_full_04',
      title: '多智能体协同中的共享记忆一致性研究', research_direction: 'Agent记忆引擎设计',
      description: '分析多智能体读写共享记忆时的一致性冲突及恢复策略。',
      status: 'failed', stage: 'experiment', current_epoch: 8, epochs: 30,
      model: 'Qwen2.5-14B-Instruct', strategy: 'Full Fine-tuning', data_ratio: 0.8,
      created_at: '2026-11-02T16:20:00+08:00', updated_at: '2026-11-03T08:52:00+08:00',
      error: 'CUDA out of memory. 请降低 batch size 或改用 QLoRA 后重试。',
      resources: { gpu_percent: 0, gpu_memory_used: 23.8, gpu_memory_total: 24, cpu_percent: 12, ram_used: 48.2, ram_total: 64 }
    }
  ];

  const demoPaper = {
    task_id: 'task_evolve_002', status: 'completed',
    title: 'EvoReflect: Self-Evolving Research Agents through Structured Reflection Trajectories',
    authors: 'Jiuwen Research Team', venue: 'ICLR Short Paper', language: 'English',
    updated_at: '2026-11-03T10:18:00+08:00', pdf_url: '',
    abstract: 'Self-evolving agents offer a promising path toward autonomous scientific discovery, yet maintaining reliable improvement across iterative research cycles remains challenging. We present EvoReflect, a structured reflection framework that transforms reviewer feedback and experimental evidence into reusable agent trajectories. Across controlled research tasks, EvoReflect improves experimental consistency and Macro F1 while preserving a traceable research workflow.',
    sections: [
      { title: '1. Introduction', summary: 'Motivation, research gap, and the structured reflection hypothesis.' },
      { title: '2. Related Work', summary: 'Agent self-evolution, reflective learning, and automated scientific discovery.' },
      { title: '3. Method', summary: 'Reflection trajectory construction, evidence grounding, and iterative policy updates.' },
      { title: '4. Experiments', summary: 'Baselines, ablation studies, evaluation metrics, and reproducible experimental settings.' },
      { title: '5. Conclusion', summary: 'Empirical findings, limitations, and future research directions.' }
    ]
  };

  function demoMetrics(task) {
    const offset = task.id === 'task_evolve_002' ? 0.035 : task.id === 'task_swarm_004' ? -0.09 : 0;
    const make = (base) => ({
      accuracy: base + 0.012,
      precision: { macro: base - 0.006, weighted: base + 0.009 },
      recall: { macro: base - 0.012, weighted: base + 0.012 },
      f1: { macro: base, weighted: base + 0.014 }
    });
    return { val: make(0.864 + offset), test: make(0.847 + offset) };
  }

  function demoEpoch(task) {
    if (task.status === 'completed') return task.epochs;
    if (task.status !== 'running') return task.current_epoch || 0;
    return Math.min(task.epochs, Number(task.current_epoch || 0) + Math.floor((Date.now() - demoStartedAt) / 15000));
  }

  function demoLog(task) {
    const current = demoEpoch(task);
    const lines = [];
    for (let epoch = 1; epoch <= current; epoch += 1) {
      const progress = epoch / task.epochs;
      const wave = Math.sin(epoch * 1.71) * 0.008;
      lines.push(JSON.stringify({
        timestamp: new Date(Date.now() - (current - epoch) * 15000).toISOString(),
        epoch,
        train_loss: Number((1.62 * Math.exp(-2.7 * progress) + 0.18 + wave).toFixed(4)),
        train_acc: Number(Math.min(0.985, 0.51 + 0.46 * (1 - Math.exp(-3 * progress)) + wave).toFixed(4)),
        val_loss: Number((1.7 * Math.exp(-2.45 * progress) + 0.25 + Math.cos(epoch) * 0.009).toFixed(4)),
        val_acc: Number(Math.min(0.94, 0.48 + 0.41 * (1 - Math.exp(-2.7 * progress)) - wave).toFixed(4)),
        val_f1: Number(Math.min(0.93, 0.47 + 0.4 * (1 - Math.exp(-2.65 * progress)) - wave).toFixed(4)),
        lr: epoch > 20 ? 0.00005 : epoch > 10 ? 0.0001 : 0.0002,
        duration_seconds: 82 + (epoch % 4)
      }));
    }
    if (task.status === 'failed' && current > 0) lines.push('[ERROR] CUDA out of memory. Training process exited with code 1.');
    return lines.join('\n');
  }

  class DemoClient {
    async listTasks() {
      return wait(demoTasks.map(task => Object.assign({}, task, { metrics: task.status === 'completed' ? demoMetrics(task) : undefined })));
    }

    async getTask(id) {
      const task = demoTasks.find(item => item.id === id);
      if (!task) throw new ApiError('未找到该演示任务', 404);
      return wait(Object.assign({}, task));
    }

    async createTask(payload) {
      const id = `task_demo_${Date.now()}`;
      const task = {
        id, experiment_id: `exp_demo_${String(Date.now()).slice(-8)}`,
        title: payload.title, research_direction: payload.research_direction,
        description: payload.description || '', status: 'queued', stage: 'topic',
        current_epoch: 0, epochs: Number(payload.config.epochs),
        model: payload.config.model, strategy: payload.config.strategy,
        data_ratio: Number(payload.config.data_ratio), created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        resources: { gpu_percent: 0, gpu_memory_used: 0, gpu_memory_total: 24, cpu_percent: 2, ram_used: 6.4, ram_total: 64 }
      };
      demoTasks.unshift(task);
      return wait(task, 240);
    }

    async controlTask(id, action) {
      const task = demoTasks.find(item => item.id === id);
      if (!task) throw new ApiError('未找到该演示任务', 404);
      if (action === 'start' || action === 'resume') {
        task.status = 'running'; task.stage = 'experiment';
      } else if (action === 'pause') {
        task.status = 'paused';
      } else if (action === 'stop') {
        task.status = 'stopped';
      }
      task.updated_at = new Date().toISOString();
      return wait(Object.assign({}, task));
    }

    async getExperiment(task) {
      const source = demoTasks.find(item => item.id === task.id) || task;
      return wait({
        config: {
          experiment_id: source.experiment_id, model: source.model, strategy: source.strategy,
          data_ratio: source.data_ratio, epochs: source.epochs, batch_size: 16,
          learning_rate: 0.0002, seed: 42
        },
        metrics: source.status === 'completed' ? demoMetrics(source) : null,
        log: demoLog(source), resources: source.resources,
        curves: null
      });
    }

    async getPaper(taskId) {
      if (taskId !== 'task_evolve_002') throw new ApiError('该任务的论文尚未生成，完成实验后可在论文页面查看。', 404);
      return wait(Object.assign({}, demoPaper));
    }

    getCurveUrl() { return ''; }
  }

  class RealClient {
    constructor(settings) {
      this.apiBase = (settings.apiBase || '').replace(/\/$/, '');
      this.assetBase = (settings.assetBase || '').replace(/\/$/, '');
    }

    apiUrl(path) { return `${this.apiBase}${path}`; }
    assetUrl(path) { return `${this.assetBase}/${path.replace(/^\//, '')}`; }

    async request(path, options = {}) {
      const response = await fetch(this.apiUrl(path), {
        headers: Object.assign({ 'Content-Type': 'application/json' }, options.headers || {}),
        ...options
      });
      if (!response.ok) {
        let detail = '';
        try { detail = (await response.json()).detail || ''; } catch (_) { /* ignore non-json errors */ }
        throw new ApiError(detail || `接口请求失败（HTTP ${response.status}）`, response.status);
      }
      if (response.status === 204) return null;
      return response.json();
    }

    normalizeTask(task) {
      return Object.assign({}, task, {
        id: task.id || task.task_id,
        experiment_id: task.experiment_id || task.experimentId || task.id || task.task_id,
        research_direction: task.research_direction || task.direction || 'Agent 科研任务',
        updated_at: task.updated_at || task.update_time || task.created_at
      });
    }

    async listTasks() {
      const data = await this.request('/api/tasks');
      return (Array.isArray(data) ? data : data.tasks || data.items || []).map(task => this.normalizeTask(task));
    }

    async getTask(id) {
      const data = await this.request(`/api/tasks/${encodeURIComponent(id)}`);
      return this.normalizeTask(data.task || data);
    }

    async createTask(payload) {
      const data = await this.request('/api/tasks', { method: 'POST', body: JSON.stringify(payload) });
      return this.normalizeTask(data.task || data);
    }

    async controlTask(id, action) {
      const data = await this.request(`/api/tasks/${encodeURIComponent(id)}/actions`, { method: 'POST', body: JSON.stringify({ action }) });
      return this.normalizeTask(data.task || data);
    }

    async fetchJson(path) {
      const response = await fetch(this.assetUrl(path), { cache: 'no-store' });
      if (!response.ok) throw new ApiError(`${path} 读取失败（HTTP ${response.status}）`, response.status);
      return response.json();
    }

    async fetchText(path) {
      const response = await fetch(this.assetUrl(path), { cache: 'no-store' });
      if (!response.ok) throw new ApiError(`${path} 读取失败（HTTP ${response.status}）`, response.status);
      return response.text();
    }

    async firstJson(paths) {
      let lastError;
      for (const path of paths) {
        try { return await this.fetchJson(path); } catch (error) { lastError = error; }
      }
      throw lastError || new ApiError('配置文件不存在', 404);
    }

    async getExperiment(task) {
      const id = encodeURIComponent(task.experiment_id || task.id);
      const base = `results/runs/${id}`;
      const requests = await Promise.allSettled([
        this.firstJson([`${base}/config.json`, 'config.json']),
        this.firstJson([`${base}/metrics.json`, 'metrics.json']),
        this.fetchText(`logs/${id}.log`)
      ]);
      if (requests.every(result => result.status === 'rejected')) {
        throw new ApiError('未读取到实验数据，请检查实验文件根路径及文件命名。', 404);
      }
      const value = index => requests[index].status === 'fulfilled' ? requests[index].value : null;
      return {
        config: value(0) || task.config || task,
        metrics: value(1), log: value(2) || '', resources: task.resources || task.resource || null,
        curves: { loss: this.getCurveUrl(task.experiment_id || task.id, 'loss'), accuracy: this.getCurveUrl(task.experiment_id || task.id, 'accuracy') },
        warnings: requests.filter(result => result.status === 'rejected').map(result => result.reason.message)
      };
    }

    async getPaper(taskId) {
      const data = await this.request(`/api/tasks/${encodeURIComponent(taskId)}/paper`);
      return data.paper || data;
    }

    getCurveUrl(id, type) {
      return `${this.assetUrl(`results/runs/${encodeURIComponent(id)}/${type}_curve.png`)}?t=${Date.now()}`;
    }
  }

  class ApiClient {
    constructor(settings) {
      this.settings = settings || loadSettings();
      this.mode = this.settings.mode;
      this.client = this.mode === 'real' ? new RealClient(this.settings) : new DemoClient();
    }

    listTasks() { return this.client.listTasks(); }
    getTask(id) { return this.client.getTask(id); }
    createTask(payload) { return this.client.createTask(payload); }
    controlTask(id, action) { return this.client.controlTask(id, action); }
    getExperiment(task) { return this.client.getExperiment(task); }
    getPaper(id) { return this.client.getPaper(id); }
    getCurveUrl(id, type) { return this.client.getCurveUrl(id, type); }
  }

  window.JiuwenAPI = { ApiClient, ApiError, loadSettings, saveSettings };
})();
