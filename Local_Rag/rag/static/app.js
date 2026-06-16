
// ── Settings ──────────────────────────────────────────────────────────────────
let appSettings = {
  provider: 'ollama',
  baseUrl: 'http://localhost:11434',
  model: 'gemma3:12b'
};

function loadSettings() {
  const saved = localStorage.getItem('papers_rag_settings');
  if (saved) {
    try {
      appSettings = { ...appSettings, ...JSON.parse(saved) };
    } catch(e) {}
  }
}

function openSettingsModal() {
  document.getElementById('settings-provider').value = appSettings.provider;
  document.getElementById('settings-url').value = appSettings.baseUrl;
  document.getElementById('settings-model').value = appSettings.model;
  document.getElementById('settings-overlay').classList.add('open');
}

function closeSettingsModal(e) {
  if (e && e.target !== document.getElementById('settings-overlay')) return;
  document.getElementById('settings-overlay').classList.remove('open');
}

function toggleSettingsFields() {
  const provider = document.getElementById('settings-provider').value;
  const urlInput = document.getElementById('settings-url');
  if (provider === 'lmstudio' && urlInput.value.includes('11434')) {
      urlInput.value = 'http://localhost:1234';
  } else if (provider === 'ollama' && urlInput.value.includes('1234')) {
      urlInput.value = 'http://localhost:11434';
  }
}

function saveSettings() {
  appSettings.provider = document.getElementById('settings-provider').value;
  appSettings.baseUrl = document.getElementById('settings-url').value;
  appSettings.model = document.getElementById('settings-model').value;
  localStorage.setItem('papers_rag_settings', JSON.stringify(appSettings));
  closeSettingsModal();
}

// ── marked.js ─────────────────────────────────────────────────────────────────
let _markedRenderer;
try {
  _markedRenderer = new marked.Renderer();
  _markedRenderer.link = ({ href, title, text }) =>
    `<a href="${href}"${title ? ` title="${title}"` : ''} target="_blank" rel="noopener">${text}</a>`;
  marked.use({ renderer: _markedRenderer, gfm: true, breaks: true });
} catch(e) {
  console.warn("marked.js failed to load. Markdown rendering will be disabled.");
}

function renderMarkdown(md) {
  let html = md;
  try {
    if (typeof marked !== 'undefined') html = marked.parse(md);
  } catch(e) {}
  const badges = [
    ['Strong Evidence','verdict-strong'], ['Moderate Evidence','verdict-moderate'],
    ['Limited Evidence','verdict-limited'], ['Insufficient Evidence','verdict-insufficient'],
    ['Contradictory Evidence','verdict-contradictory'],
  ];
  badges.forEach(([label, cls]) => {
    html = String(html).replace(
      new RegExp(`<strong>${label}<\\/strong>`, 'g'),
      `<span class="verdict-badge ${cls}">${label}</span>`
    );
  });
  return html;
}

let _lastWebResults = [];
function applyWebLinks(html) {
  return html.replace(/\[web:(\d+)\]/g, (_, n) => {
    const r = _lastWebResults.find(x => x.index == n);
    if (r) return `<a href="${escAttr(r.url)}" target="_blank" rel="noopener"
      style="color:var(--green);font-size:11px;text-decoration:none;border:1px solid #1e3a22;border-radius:3px;padding:0 4px">[web:${n}]</a>`;
    return `<span style="color:var(--green);font-size:11px">[web:${n}]</span>`;
  });
}

// ── State ─────────────────────────────────────────────────────────────────────
let isGenerating   = false;
let activeAgent    = 'chat';
let activeView     = 'chat';
let AGENTS         = {};
let _rpCollapsed   = false;
let _rpTab         = 'history';
let _currentSessionId = null;
let _currentNoteId = null;
let _pendingUploads = [];
let _lastUsedMemoryIds = new Set();

// Search state
let _lastSearchQuery = '';
let _searchTopK      = 12;
let _searchShownIds  = new Set();

// Memory search debounce
let _memSearchTimer = null;

// ── Bootstrap ─────────────────────────────────────────────────────────────────
async function init() {
  loadSettings();
  try {
    const r = await fetch('/api/agents');
    AGENTS = await r.json();
  } catch(e) {
    AGENTS = {
      chat:{name:'Academic Chat',icon:'💬',description:'Ask questions about your papers'},
      gaps:{name:'Research Gaps',icon:'🔭',description:'Find unexplored areas'},
      brainstorm:{name:'Brainstorm Questions',icon:'💡',description:'Generate novel research questions'},
      evidence:{name:'Claim Evidence',icon:'⚖️',description:'Find supporting / contradicting evidence'},
      overview:{name:'Literature Overview',icon:'📚',description:'Structured overview of a research area'},
      citations:{name:'Find Citations',icon:'🗂️',description:'Find papers that support a statement'},
      verdict:{name:'Research Verdict',icon:'🏛️',description:'Evaluate evidence for a hypothesis'},
      peerreview:{name:'Mock Peer Review',icon:'📝',description:'Peer review on your idea'},
      hallucination:{name:'Hallucination Check',icon:'🔍',description:'Verify if a claim is in the papers'},
    };
  }
  buildSidebar();
  selectAgent('chat');
  checkStatus();
  setInterval(checkStatus, 15000);
  loadHistory();
  loadMemory();
}

// ── Left sidebar ──────────────────────────────────────────────────────────────
function buildSidebar() {
  const list = document.getElementById('agent-list');
  list.innerHTML = '';
  
  // Group agents by category
  const categories = {};
  Object.keys(AGENTS).forEach(id => {
    const a = AGENTS[id];
    const cat = a.category || 'Other';
    if (!categories[cat]) categories[cat] = [];
    categories[cat].push({ id, ...a });
  });

  // Render categories
  for (const [cat, agents] of Object.entries(categories)) {
    const sectionLabel = document.createElement('div');
    sectionLabel.className = 'sidebar-section-label';
    sectionLabel.textContent = cat;
    list.appendChild(sectionLabel);

    agents.forEach(a => {
      const el = document.createElement('div');
      el.className = 'agent-item';
      el.id = 'agent-item-' + a.id;
      el.innerHTML = `<span class="agent-icon">${a.icon}</span><span class="agent-label">${a.name}</span>`;
      el.onclick = () => selectAgent(a.id);
      list.appendChild(el);
    });
  }

  const div1 = document.createElement('div'); div1.className = 'sidebar-divider'; list.appendChild(div1);
  const notesEl = document.createElement('div');
  notesEl.className = 'agent-item'; notesEl.id = 'agent-item-notes';
  notesEl.innerHTML = `<span class="agent-icon">📓</span><span class="agent-label">My Notes</span>`;
  notesEl.onclick = () => selectAgent('notes'); list.appendChild(notesEl);
  const div2 = document.createElement('div'); div2.className = 'sidebar-divider'; list.appendChild(div2);
  const searchEl = document.createElement('div');
  searchEl.className = 'agent-item'; searchEl.id = 'agent-item-search';
  searchEl.innerHTML = `<span class="agent-icon">🔍</span><span class="agent-label">Find Papers</span>`;
  searchEl.onclick = () => selectAgent('search'); list.appendChild(searchEl);
  const div3 = document.createElement('div'); div3.className = 'sidebar-divider'; list.appendChild(div3);
  const graphEl = document.createElement('div');
  graphEl.className = 'agent-item'; graphEl.id = 'agent-item-graph';
  graphEl.innerHTML = `<span class="agent-icon">🕸️</span><span class="agent-label">Knowledge Graph</span>`;
  graphEl.onclick = () => selectAgent('graph'); list.appendChild(graphEl);
}

function showView(name) {
  ['chat','search','notes','graph'].forEach(v => {
    const el = document.getElementById('view-'+v);
    if (el) el.classList.toggle('active', v === name);
  });
  activeView = name;
}

function selectAgent(id) {
  const prevAgent = activeAgent;
  document.querySelectorAll('.agent-item').forEach(el => el.classList.remove('active'));
  const el = document.getElementById('agent-item-'+id);
  if (el) el.classList.add('active');
  activeAgent = id;
  if (id === 'search') {
    showView('search');
    document.getElementById('ah-icon').textContent = '🔍';
    document.getElementById('ah-name').textContent = 'Find Papers';
    document.getElementById('ah-desc').textContent = 'Semantic + keyword search across all indexed chunks';
    return;
  }
  if (id === 'notes') {
    showView('notes');
    document.getElementById('ah-icon').textContent = '📓';
    document.getElementById('ah-name').textContent = 'My Notes';
    document.getElementById('ah-desc').textContent = 'Saved answers and research notes';
    loadNotes();
    return;
  }
  if (id === 'graph') {
    showView('graph');
    document.getElementById('ah-icon').textContent = '🕸️';
    document.getElementById('ah-name').textContent = 'Knowledge Graph';
    document.getElementById('ah-desc').textContent = 'Entity & relation graph built from your papers';
    initGraphView();
    return;
  }
  showView('chat');
  const a = AGENTS[id] || {};
  document.getElementById('ah-icon').textContent = a.icon || '💬';
  document.getElementById('ah-name').textContent = a.name || id;
  document.getElementById('ah-desc').textContent = a.description || '';
  document.getElementById('ask-input').placeholder =
    (a.placeholder || 'Ask a question…') + ' (Enter to send)';
  refreshEmptyState(id);
  // Start a fresh chat whenever switching to a different chat agent
  if (prevAgent !== id) newChat();
}

function refreshEmptyState(id) {
  const a = AGENTS[id] || {};
  const emptyEl = document.getElementById('empty-chat');
  if (!emptyEl) return;
  document.getElementById('empty-icon').textContent  = a.icon || '🧪';
  document.getElementById('empty-title').textContent = a.name || 'Research Assistant';
  document.getElementById('empty-sub').textContent   = a.description || '';
  const exQ = document.getElementById('example-queries');
  exQ.innerHTML = '';
  (a.examples || []).forEach(q => {
    const d = document.createElement('div');
    d.className = 'example-q'; d.textContent = q;
    d.onclick = () => setQuery(q); exQ.appendChild(d);
  });
}

// ── Right panel ───────────────────────────────────────────────────────────────
function toggleRightPanel() {
  _rpCollapsed = !_rpCollapsed;
  document.getElementById('right-panel').classList.toggle('collapsed', _rpCollapsed);
  const btn = document.getElementById('panel-toggle-btn');
  btn.classList.toggle('active', !_rpCollapsed);
  btn.textContent = _rpCollapsed ? '💬 History' : '💬 History';
}

function switchRpTab(tab) {
  _rpTab = tab;
  ['history','memory'].forEach(t => {
    document.getElementById('rp-tab-'+t).classList.toggle('active', t === tab);
    document.getElementById('rp-'+t).classList.toggle('active', t === tab);
  });
  if (tab === 'history') loadHistory();
  else loadMemory();
}

// ── Status polling ─────────────────────────────────────────────────────────────
async function checkStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    document.getElementById('dot-ollama').className = 'status-dot '+(d.ollama     ?'ok':'err');
    document.getElementById('dot-embed').className  = 'status-dot '+(d.embeddings ?'ok':'err');
    document.getElementById('dot-index').className  = 'status-dot '+(d.index      ?'ok':'err');
    const cnt = d.papers || 0;
    document.getElementById('paper-count').textContent = cnt+' papers';
    document.getElementById('sidebar-stats').textContent = cnt+' papers indexed';
    const ing = d.ingest || {};
    const indEl = document.getElementById('ingest-indicator');
    if (ing.running || ing.new_count > 0) {
      indEl.style.display = 'flex';
      document.getElementById('ingest-msg').textContent = ing.message || 'Checking…';
    } else { indEl.style.display = 'none'; }
  } catch(e) {
    ['dot-ollama','dot-embed','dot-index'].forEach(id =>
      document.getElementById(id).className = 'status-dot err');
  }
}

// ── Session / History ─────────────────────────────────────────────────────────
async function loadHistory() {
  const el = document.getElementById('rp-sessions');
  try {
    const resp = await fetch('/api/sessions');
    const sessions = await resp.json();
    if (!sessions.length) {
      el.innerHTML = '<div class="rp-empty">No chat history yet.<br>Start a conversation to save it here.</div>';
      return;
    }
    // Group by date
    const groups = {};
    const now = Date.now() / 1000;
    sessions.forEach(s => {
      let g = 'Older';
      const diff = now - s.updated_at;
      if (diff < 86400)   g = 'Today';
      else if (diff < 172800) g = 'Yesterday';
      else if (diff < 604800) g = 'This week';
      (groups[g] = groups[g] || []).push(s);
    });
    el.innerHTML = '';
    ['Today','Yesterday','This week','Older'].forEach(grp => {
      if (!groups[grp]) return;
      const lbl = document.createElement('div');
      lbl.className = 'session-group-label'; lbl.textContent = grp;
      el.appendChild(lbl);
      groups[grp].forEach(s => el.appendChild(_makeSessionCard(s)));
    });
  } catch(e) {
    el.innerHTML = `<div class="rp-empty" style="color:var(--red)">Error loading history</div>`;
  }
}

function _makeSessionCard(s) {
  const card = document.createElement('div');
  card.className = 'session-card' + (s.session_id === _currentSessionId ? ' active' : '');
  card.dataset.id = s.session_id;

  const agentIcon = (AGENTS[s.agent_id] || {}).icon || '💬';
  const timeAgo = _timeAgo(s.updated_at);

  card.innerHTML = `
    <div class="session-card-body">
      <div class="session-card-title">${escHtml(s.title)}</div>
      <div class="session-card-meta">${agentIcon} ${s.turn_count} turns · ${timeAgo}</div>
    </div>
    <button class="session-delete-btn" title="Delete" onclick="event.stopPropagation();deleteSession('${s.session_id}')">✕</button>`;
  card.onclick = () => loadSession(s.session_id);
  return card;
}

async function loadSession(sessionId) {
  // Highlight in list
  document.querySelectorAll('.session-card').forEach(c =>
    c.classList.toggle('active', c.dataset.id === sessionId));

  _currentSessionId = sessionId;

  // Switch to chat view
  selectAgent(activeAgent === 'search' || activeAgent === 'notes' ? 'chat' : activeAgent);

  const area = document.getElementById('chat-area');
  area.innerHTML = `<div style="padding:20px;color:var(--text-dim);display:flex;align-items:center;gap:10px"><span class="spinner"></span> Loading session…</div>`;

  try {
    const resp = await fetch('/api/sessions/' + sessionId);
    const data = await resp.json();
    if (data.error) throw new Error(data.error);

    area.innerHTML = '';
    area.appendChild(_makeSessionSeparator(`📋 Restored: ${data.title}`));

    // Render historical turns in pairs
    const turns = data.turns || [];
    for (let i = 0; i < turns.length; i++) {
      const t = turns[i];
      if (t.role === 'user') {
        const div = document.createElement('div');
        div.className = 'msg';
        div.innerHTML = `<div class="msg-user">${escHtml(t.content)}</div>`;
        area.appendChild(div);
      } else {
        const agentInfo = AGENTS[t.agent_id || data.agent_id] || {};
        const sys = document.createElement('div');
        sys.className = 'msg msg-system';
        const lbl = document.createElement('div');
        lbl.className = 'msg-agent-label';
        lbl.innerHTML = `<span>${agentInfo.icon || '🤖'}</span> <strong>${agentInfo.name || t.agent_id || 'Assistant'}</strong>`;
        const bubble = document.createElement('div');
        bubble.className = 'answer-bubble';
        bubble.innerHTML = applyWebLinks(renderMarkdown(t.content));
        sys.appendChild(lbl); sys.appendChild(bubble);
        area.appendChild(sys);
      }
    }

    // "Continue conversation" separator
    area.appendChild(_makeSessionSeparator('Continue this conversation below ↓'));
    area.scrollTop = area.scrollHeight;

  } catch(e) {
    area.innerHTML = `<div style="padding:20px;color:var(--red)">Error loading session: ${escHtml(e.message)}</div>`;
  }
}

function _makeSessionSeparator(text) {
  const sep = document.createElement('div');
  sep.className = 'session-separator';
  sep.textContent = text;
  return sep;
}

async function deleteSession(sessionId) {
  await fetch('/api/sessions/'+sessionId, { method: 'DELETE' });
  if (_currentSessionId === sessionId) { _currentSessionId = null; }
  loadHistory();
}

function newChat() {
  _currentSessionId = null;
  _lastWebResults = [];
  const area = document.getElementById('chat-area');
  area.innerHTML = `
    <div class="empty-state" id="empty-chat">
      <div class="empty-icon" id="empty-icon"></div>
      <div class="empty-title" id="empty-title"></div>
      <div class="empty-sub" id="empty-sub"></div>
      <div class="example-queries" id="example-queries"></div>
    </div>`;
  refreshEmptyState(activeAgent);
  // Unhighlight sessions
  document.querySelectorAll('.session-card').forEach(c => c.classList.remove('active'));
  document.getElementById('ask-input').focus();
}

// ── Memory ────────────────────────────────────────────────────────────────────
async function loadMemory(highlightIds) {
  const el = document.getElementById('rp-memories');
  try {
    const resp = await fetch('/api/memory');
    const memories = await resp.json();
    _renderMemories(el, memories, highlightIds);
  } catch(e) {
    el.innerHTML = `<div class="rp-empty" style="color:var(--red)">Error loading memory</div>`;
  }
}

function _renderMemories(el, memories, highlightIds) {
  if (!memories.length) {
    el.innerHTML = '<div class="rp-empty">No memories yet.<br>The AI automatically extracts key findings from your conversations.</div>';
    return;
  }
  el.innerHTML = '';
  const hiSet = new Set(highlightIds || []);
  memories.forEach(m => {
    const card = document.createElement('div');
    card.className = 'memory-card' + (hiSet.has(m.memory_id) ? ' highlighted' : '');
    card.dataset.id = m.memory_id;
    const timeAgo = _timeAgo(m.created_at);
    card.innerHTML = `
      <div class="memory-content">${escHtml(m.content)}</div>
      <div class="memory-meta">
        <span class="memory-type-badge">${escHtml(m.memory_type)}</span>
        <span>${timeAgo}</span>
        ${m.source_query ? `<span title="${escAttr(m.source_query)}" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:70px">↩ ${escHtml(m.source_query.slice(0,30))}…</span>` : ''}
        <button class="memory-delete-btn" title="Delete" onclick="deleteMemoryCard('${m.memory_id}', this.closest('.memory-card'))">✕</button>
      </div>`;
    el.appendChild(card);
  });
}

async function deleteMemoryCard(memoryId, cardEl) {
  await fetch('/api/memory/'+memoryId, { method: 'DELETE' });
  cardEl?.remove();
  const el = document.getElementById('rp-memories');
  if (!el.querySelector('.memory-card')) {
    el.innerHTML = '<div class="rp-empty">No memories yet.</div>';
  }
}

function debounceMemorySearch(val) {
  clearTimeout(_memSearchTimer);
  if (!val.trim()) { loadMemory(); return; }
  _memSearchTimer = setTimeout(() => searchMemory(val), 400);
}

async function searchMemory(query) {
  if (!query.trim()) { loadMemory(); return; }
  const el = document.getElementById('rp-memories');
  el.innerHTML = `<div style="padding:12px;color:var(--text-dim);font-size:12px;display:flex;gap:8px;align-items:center"><span class="spinner"></span>Searching…</div>`;
  try {
    const resp = await fetch('/api/memory/search', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({query})
    });
    const results = await resp.json();
    _renderMemories(el, results);
  } catch(e) {
    el.innerHTML = `<div class="rp-empty" style="color:var(--red)">Search failed</div>`;
  }
}

// ── File upload ───────────────────────────────────────────────────────────────
async function handleFileSelect(e) {
  const files = Array.from(e.target.files || []);
  e.target.value = '';
  for (const file of files) {
    const fd = new FormData(); fd.append('file', file);
    try {
      const resp = await fetch('/api/upload', { method:'POST', body:fd });
      const data = await resp.json();
      if (data.error) { alert('Upload error: '+data.error); continue; }
      _pendingUploads.push(data);
      addUploadChip(data);
    } catch(err) { alert('Upload failed: '+err.message); }
  }
}
function addUploadChip(upload) {
  const strip = document.getElementById('upload-strip');
  strip.style.display = 'flex';
  const chip = document.createElement('div'); chip.className = 'upload-chip'; chip.dataset.filename = upload.filename;
  if (upload.type === 'image') {
    chip.innerHTML = `<img src="data:${upload.mime};base64,${upload.data}" alt=""><span>${escHtml(upload.filename)}</span><span class="upload-chip-remove" onclick="removeUpload('${escAttr(upload.filename)}',this.closest('.upload-chip'))">✕</span>`;
  } else {
    chip.innerHTML = `<span>📄</span><span>${escHtml(upload.filename)}</span><span class="upload-chip-remove" onclick="removeUpload('${escAttr(upload.filename)}',this.closest('.upload-chip'))">✕</span>`;
  }
  strip.appendChild(chip);
}
function removeUpload(filename, el) {
  _pendingUploads = _pendingUploads.filter(u => u.filename !== filename);
  el?.remove();
  if (!document.querySelectorAll('#upload-strip .upload-chip').length) {
    document.getElementById('upload-strip').style.display = 'none';
  }
}

// ── Textarea helpers ──────────────────────────────────────────────────────────
function autoResize(el) { el.style.height='auto'; el.style.height=Math.min(el.scrollHeight,140)+'px'; }
function handleKey(e) { if (e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendQuery();} }
function setQuery(q) { const el=document.getElementById('ask-input'); el.value=q; autoResize(el); el.focus(); }

// ── Send query ─────────────────────────────────────────────────────────────────
async function sendQuery() {
  if (isGenerating) return;
  const input = document.getElementById('ask-input');
  const query = input.value.trim();
  if (!query && !_pendingUploads.length) return;

  isGenerating = true;
  const btn = document.getElementById('ask-btn');
  btn.disabled = true; btn.textContent = '…';
  input.value = ''; autoResize(input);

  const images    = _pendingUploads.filter(u=>u.type==='image').map(u=>u.data);
  const docParts  = _pendingUploads.filter(u=>u.type==='document').map(u=>`[${u.filename}]\n${u.text}`);
  const docContext= docParts.join('\n\n') || null;
  const uploadSnap= [..._pendingUploads];
  _pendingUploads = [];
  const strip = document.getElementById('upload-strip');
  strip.innerHTML = ''; strip.style.display = 'none';

  document.getElementById('empty-chat')?.remove();
  const area  = document.getElementById('chat-area');
  const agent = AGENTS[activeAgent] || {};

  // User bubble
  const userMsg = document.createElement('div');
  userMsg.className = 'msg';
  let userHtml = `<div class="msg-user">${escHtml(query||'(attached files)')}</div>`;
  if (uploadSnap.length) {
    let prev = '<div class="msg-upload-preview">';
    uploadSnap.forEach(u => {
      if (u.type==='image') prev += `<img src="data:${u.mime};base64,${u.data}" alt="${escAttr(u.filename)}">`;
      else prev += `<div class="doc-tag">📄 ${escHtml(u.filename)}</div>`;
    });
    userHtml += prev + '</div>';
  }
  userMsg.innerHTML = userHtml;
  area.appendChild(userMsg);

  // System message
  const sysMsg = document.createElement('div'); sysMsg.className='msg msg-system';
  const agentLabel = document.createElement('div'); agentLabel.className='msg-agent-label';
  agentLabel.innerHTML = `<span>${agent.icon||'🤖'}</span> <strong>${agent.name||activeAgent}</strong>`;

  const papersStrip = document.createElement('div'); papersStrip.className='papers-strip';
  papersStrip.innerHTML=`<div style="padding:7px;color:var(--text-dim);font-size:12px;display:flex;align-items:center;gap:8px"><span class="spinner"></span> Retrieving papers…</div>`;

  const webStripWrap = document.createElement('div'); webStripWrap.style.display='none';
  const webStripLabel = document.createElement('div'); webStripLabel.className='web-strip-label'; webStripLabel.innerHTML='🌐 Web results';
  const webStrip = document.createElement('div'); webStrip.className='papers-strip';
  webStripWrap.appendChild(webStripLabel); webStripWrap.appendChild(webStrip);

  const answerBubble = document.createElement('div'); answerBubble.className='answer-bubble';
  answerBubble.innerHTML='<span class="typing-cursor"></span>';

  sysMsg.appendChild(agentLabel);
  sysMsg.appendChild(papersStrip);
  sysMsg.appendChild(webStripWrap);
  sysMsg.appendChild(answerBubble);
  area.appendChild(sysMsg);
  area.scrollTop = area.scrollHeight;

  let answerText = '';
  const savedQuery = query || uploadSnap.map(u=>u.filename).join(', ');

  try {
    const body = { 
      query: query||'Please analyse the attached content.', 
      agent: activeAgent,
      provider: appSettings.provider,
      base_url: appSettings.baseUrl,
      gen_model: appSettings.model
    };
    if (_currentSessionId) body.session_id = _currentSessionId;
    if (images.length)     body.images = images;
    if (docContext)        body.doc_context = docContext;
    if (_selectedPaperIds.size > 0) body.selected_paper_ids = Array.from(_selectedPaperIds);

    await streamSSE('/api/query', body, {
      session: (data) => {
        _currentSessionId = data.session_id;
        // Refresh history after a moment
        setTimeout(() => { if (_rpTab==='history') loadHistory(); else loadHistory(); }, 800);
      },
      papers: (data) => {
        const papers = data.papers || data;
        papersStrip.innerHTML = '';
        const shownIds = new Set();
        if (!papers.length) {
          papersStrip.innerHTML = `<span style="font-size:12px;color:var(--text-dim)">No local papers found.</span>`;
        } else {
          papers.forEach(p => { shownIds.add(p.paper_id); papersStrip.appendChild(_makePaperChip(p)); });
        }
        const moreBtn = document.createElement('button');
        moreBtn.className='more-sources-btn'; moreBtn.textContent='＋ More sources';
        moreBtn.onclick = () => loadMoreSources(savedQuery, shownIds, papersStrip, moreBtn);
        papersStrip.appendChild(moreBtn);
      },
      web_results: (data) => {
        if (!data||!data.length) return;
        _lastWebResults = data;
        webStripWrap.style.display = 'block';
        webStrip.innerHTML = '';
        data.forEach(r => {
          const chip = document.createElement('a');
          chip.className='web-chip'; chip.href=r.url; chip.target='_blank'; chip.rel='noopener noreferrer';
          const domain = (() => { try{return new URL(r.url).hostname.replace('www.','')}catch(e){return r.url}})();
          chip.innerHTML=`<div class="web-chip-title">${escHtml(r.title||r.url)}</div><div class="web-chip-meta"><span class="web-badge">[web:${r.index}]</span><span style="margin-left:6px">${escHtml(domain)}</span></div>`;
          webStrip.appendChild(chip);
        });
      },
      memories: (data) => {
        if (!data||!data.length) return;
        const badge = document.createElement('span');
        badge.className = 'memory-used-badge';
        badge.textContent = `🧠 ${data.length} memor${data.length===1?'y':'ies'} recalled`;
        agentLabel.appendChild(badge);
      },
      graph_entities: (data) => {
        if (!data||!data.length) return;
        const badge = document.createElement('span');
        badge.className = 'memory-used-badge';
        badge.style.cssText = 'background:rgba(139,92,246,.15);color:#8b5cf6;border-color:rgba(139,92,246,.3)';
        badge.textContent = `🕸️ ${data.length} graph entit${data.length===1?'y':'ies'}`;
        agentLabel.appendChild(badge);
      },
      token: (data) => {
        answerText += data.text;
        answerBubble.innerHTML = applyWebLinks(renderMarkdown(answerText))+'<span class="typing-cursor"></span>';
        area.scrollTop = area.scrollHeight;
      },
      done: () => {
        answerBubble.innerHTML = applyWebLinks(renderMarkdown(answerText));
        // Save-to-notes button
        const saveBtn = document.createElement('button');
        saveBtn.className='btn btn-ghost save-note-btn'; saveBtn.textContent='📓 Save to Notes';
        saveBtn.onclick = () => saveToNotes(savedQuery, answerText, activeAgent, saveBtn);
        answerBubble.appendChild(saveBtn);
        area.scrollTop = area.scrollHeight;
        // Refresh memory tab after extraction (delayed ~10s for extraction to complete)
        setTimeout(() => loadMemory(), 12000);
      },
      error: (data) => {
        answerBubble.innerHTML=`<span style="color:var(--red)">Error: ${escHtml(data.message)}</span>`;
      }
    });
  } catch(e) {
    answerBubble.innerHTML=`<span style="color:var(--red)">Connection error: ${escHtml(e.message)}</span>`;
  }

  isGenerating = false;
  btn.disabled = false; btn.textContent = 'Send';
}

// ── SSE ───────────────────────────────────────────────────────────────────────
async function streamSSE(url, body, handlers) {
  const resp = await fetch(url, {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)
  });
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf='', curEvent=null;
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    buf += dec.decode(value, {stream:true});
    const lines = buf.split('\n'); buf = lines.pop()??'';
    for (const line of lines) {
      if (line.startsWith('event: '))     curEvent = line.slice(7).trim();
      else if (line.startsWith('data: ')&&curEvent) {
        try { if (handlers[curEvent]) handlers[curEvent](JSON.parse(line.slice(6))); } catch(e) {}
        curEvent = null;
      }
    }
  }
}

// ── Search ────────────────────────────────────────────────────────────────────
async function doSearch(append=false) {
  const inputEl = document.getElementById('search-input');
  const resultsEl = document.getElementById('search-results');
  if (!append) {
    const query = inputEl.value.trim(); if (!query) return;
    _lastSearchQuery=query; _searchTopK=12; _searchShownIds=new Set();
    resultsEl.innerHTML=`<div style="padding:20px;color:var(--text-dim);display:flex;align-items:center;gap:10px"><span class="spinner"></span> Searching…</div>`;
  } else {
    document.getElementById('load-more-bar')?.remove();
    const loading=document.createElement('div'); loading.id='search-loading';
    loading.style.cssText='padding:12px;color:var(--text-dim);display:flex;align-items:center;gap:8px;font-size:13px';
    loading.innerHTML=`<span class="spinner"></span> Loading more…`;
    resultsEl.appendChild(loading);
  }
  try {
    const resp=await fetch('/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:_lastSearchQuery,top_k:_searchTopK})});
    const data=await resp.json();
    const allResults=data.results||[];
    const newResults=allResults.filter(p=>!_searchShownIds.has(p.paper_id));
    if (!append){resultsEl.innerHTML='';if(!allResults.length){resultsEl.innerHTML=`<div class="empty-state"><div class="empty-icon">🔍</div><div class="empty-title">No results</div></div>`;return;}}
    else document.getElementById('search-loading')?.remove();
    newResults.forEach(p => {
      _searchShownIds.add(p.paper_id);
      const card=document.createElement('div'); card.className='result-card';
      card.style.position = 'relative';
      const isPinned = _selectedPaperIds.has(p.paper_id);
      const pinClass = isPinned ? 'pinned' : '';
      const pinText = isPinned ? '📌 Pinned' : '📌 Pin';
      card.innerHTML=`<div class="result-title">${escHtml(p.title||p.paper_id)}</div><div class="result-meta"><span class="section-badge">${escHtml(p.section_name||'Body')}</span>${p.year?`<span>${p.year}</span>`:''}<span>p.${p.page_start}–${p.page_end}</span></div><div class="result-snippet">${escHtml(p.text||'')}</div><button class="btn-ghost pin-btn ${pinClass}" style="position:absolute;top:10px;right:10px;font-size:11px;padding:4px 8px;border-radius:6px;cursor:pointer;">${pinText}</button>`;
      card.onclick=(e)=>{
        if(e.target.classList.contains('pin-btn')) {
          e.stopPropagation();
          togglePaperSelection(p.paper_id, p.title);
          e.target.classList.toggle('pinned');
          e.target.textContent = _selectedPaperIds.has(p.paper_id) ? '📌 Pinned' : '📌 Pin';
        } else {
          openPaper(p.paper_id,p.title);
        }
      };
      resultsEl.appendChild(card);
    });
    if (!append) allResults.forEach(p=>_searchShownIds.add(p.paper_id));
    if (_searchTopK<40) {
      const bar=document.createElement('div'); bar.id='load-more-bar'; bar.className='load-more-bar';
      bar.innerHTML=`<button id="load-more-btn" class="btn btn-ghost" onclick="_doLoadMore()">＋ Load more results</button><span class="results-count">Showing ${_searchShownIds.size} chunks</span>`;
      resultsEl.appendChild(bar);
    }
  } catch(e) {
    if (!append) resultsEl.innerHTML=`<div style="padding:20px;color:var(--red)">Error: ${escHtml(e.message)}</div>`;
  }
}
function _doLoadMore(){_searchTopK=Math.min(_searchTopK+14,40);doSearch(true);}

// ── Paper chip + more sources ─────────────────────────────────────────────────
function _makePaperChip(p) {
  const chip=document.createElement('div'); chip.className='paper-chip';
  chip.innerHTML=`<div class="paper-chip-title">${escHtml(p.title||p.paper_id)}</div><div class="paper-chip-meta"><span class="section-badge">${escHtml(p.section_name||'Body')}</span>${p.year?`<span>${p.year}</span>`:''}<span style="margin-left:auto;font-size:9px">p.${p.page_start}–${p.page_end}</span></div>`;
  chip.onclick=()=>openPaper(p.paper_id,p.title); return chip;
}
async function loadMoreSources(query, shownIds, strip, btn) {
  if (!query){btn.textContent='No query';btn.disabled=true;return;}
  btn.disabled=true; btn.textContent='…';
  const nextTopK=shownIds.size<=12?25:shownIds.size<=20?35:40;
  try {
    const resp=await fetch('/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query,top_k:nextTopK})});
    const data=await resp.json();
    const newPapers=(data.results||[]).filter(p=>!shownIds.has(p.paper_id));
    if (!newPapers.length){btn.textContent='No more found';return;}
    newPapers.forEach(p=>{shownIds.add(p.paper_id);strip.insertBefore(_makePaperChip(p),btn);});
    if (nextTopK>=40||newPapers.length<3){btn.textContent=`All ${shownIds.size} found`;}
    else {btn.textContent='＋ More sources';btn.disabled=false;}
  } catch(e){btn.textContent='Failed — retry';btn.disabled=false;}
}

// ── Paper modal ───────────────────────────────────────────────────────────────
async function openPaper(paperId, title) {
  document.getElementById('modal-title').textContent = title||paperId;
  document.getElementById('modal-body').innerHTML=`<div style="display:flex;align-items:center;gap:10px;color:var(--text-dim)"><span class="spinner"></span> Loading…</div>`;
  document.getElementById('modal-overlay').classList.add('open');
  try {
    const resp=await fetch('/api/paper?id='+encodeURIComponent(paperId));
    const data=await resp.json();
    if (data.error){document.getElementById('modal-body').innerHTML=`<p style="color:var(--text-dim)">Details not available.</p>`;return;}
    let html=`<div class="modal-meta">`;
    if (data.authors) html+=`<span>👤 ${escHtml(data.authors)}</span>`;
    if (data.year)    html+=`<span>📅 ${data.year}</span>`;
    if (data.pdf_available) html+=`<a href="/api/pdf?id=${encodeURIComponent(paperId)}" target="_blank" class="btn" style="font-size:11px;padding:5px 12px;margin-left:auto;text-decoration:none">📄 Open PDF</a>`;
    html+=`</div>`;
    if (data.sections&&data.sections.length) {
      html+=`<p style="font-size:12px;color:var(--text-dim);margin-bottom:8px">Sections: `+data.sections.map(s=>`<span style="background:var(--surface2);border-radius:4px;padding:1px 6px;margin-right:4px;font-size:11px">${escHtml(s.name)}</span>`).join('')+`</p>`;
    }
    html += data.summary ? renderMarkdown(data.summary) : `<p style="color:var(--text-dim)">No summary available.</p>`;
    document.getElementById('modal-body').innerHTML = html;
  } catch(e) {
    document.getElementById('modal-body').innerHTML=`<p style="color:var(--red)">Error loading paper.</p>`;
  }
}
function closeModal(e) {
  if (e.target===document.getElementById('modal-overlay'))
    document.getElementById('modal-overlay').classList.remove('open');
}

// ── Notes ─────────────────────────────────────────────────────────────────────
async function loadNotes() {
  const listEl=document.getElementById('notes-list-items');
  listEl.innerHTML=`<div style="padding:12px;color:var(--text-dim);font-size:12px;display:flex;align-items:center;gap:8px"><span class="spinner"></span></div>`;
  try {
    const resp=await fetch('/api/notes'); const notes=await resp.json();
    listEl.innerHTML='';
    if (!notes.length){listEl.innerHTML=`<div style="padding:20px;color:var(--text-dim);font-size:12px;text-align:center">No notes yet.</div>`;return;}
    notes.forEach(note=>{
      const el=document.createElement('div');
      el.className='note-list-item'+(note.id===_currentNoteId?' active':''); el.dataset.id=note.id;
      const date=new Date(note.modified*1000).toLocaleDateString('en-US',{month:'short',day:'numeric',year:'numeric'});
      el.innerHTML=`<div class="note-list-item-title">${escHtml(note.title)}</div><div class="note-list-item-date">${date}</div>`;
      el.onclick=()=>openNote(note.id); listEl.appendChild(el);
    });
    if (_currentNoteId) openNote(_currentNoteId);
  } catch(e) { listEl.innerHTML=`<div style="color:var(--red);padding:12px;font-size:12px">Error</div>`; }
}
async function openNote(noteId) {
  _currentNoteId=noteId;
  document.querySelectorAll('.note-list-item').forEach(el=>el.classList.toggle('active',el.dataset.id===noteId));
  const titleEl=document.getElementById('notes-detail-title');
  const viewerEl=document.getElementById('notes-viewer-content');
  const emptyMsg=document.getElementById('notes-empty-msg');
  const deleteBtn=document.getElementById('notes-delete-btn');
  const exportLink=document.getElementById('notes-export-link');
  emptyMsg.style.display='none'; viewerEl.style.display='block';
  deleteBtn.style.display='inline-flex'; exportLink.style.display='inline-flex';
  viewerEl.innerHTML=`<div style="color:var(--text-dim);font-size:12px;display:flex;align-items:center;gap:8px"><span class="spinner"></span></div>`;
  try {
    const resp=await fetch('/api/notes/'+noteId); const data=await resp.json();
    if (data.error) throw new Error(data.error);
    let title=noteId, content=data.content;
    const fmMatch=content.match(/^---\n([\s\S]*?)\n---\n\n?/);
    if (fmMatch){const fm=fmMatch[1];const tm=fm.match(/^title:\s*(.+)$/m);if(tm)title=tm[1].trim();content=content.slice(fmMatch[0].length);}
    titleEl.textContent=title; viewerEl.innerHTML=renderMarkdown(content);
    const blob=new Blob([data.content],{type:'text/markdown'});
    exportLink.href=URL.createObjectURL(blob); exportLink.download=noteId+'.md';
    deleteBtn.onclick=()=>deleteNote(noteId);
  } catch(e){viewerEl.innerHTML=`<div style="color:var(--red)">Error: ${escHtml(e.message)}</div>`;}
}
async function deleteNote(noteId) {
  if (!confirm('Delete this note?')) return;
  await fetch('/api/notes/'+noteId,{method:'DELETE'});
  _currentNoteId=null;
  document.getElementById('notes-detail-title').textContent='Select a note';
  document.getElementById('notes-viewer-content').style.display='none';
  document.getElementById('notes-empty-msg').style.display='flex';
  document.getElementById('notes-delete-btn').style.display='none';
  document.getElementById('notes-export-link').style.display='none';
  loadNotes();
}
async function saveToNotes(query, content, agentId, btn) {
  btn.disabled=true; btn.textContent='Saving…';
  try {
    const title=query.length>70?query.slice(0,70)+'…':query;
    const resp=await fetch('/api/notes',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({title,content,agent:agentId})});
    const data=await resp.json();
    if (data.error) throw new Error(data.error);
    btn.textContent='✓ Saved'; btn.style.color='var(--green)';
    setTimeout(()=>{btn.textContent='📓 Save to Notes';btn.style.color='';btn.disabled=false;},2500);
    if (activeView==='notes') loadNotes();
  } catch(e){btn.textContent='❌ Failed';btn.disabled=false;setTimeout(()=>{btn.textContent='📓 Save to Notes';},2000);}
}

// ── Utility ───────────────────────────────────────────────────────────────────
function escHtml(s){return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
function escAttr(s){return String(s||'').replace(/"/g,'&quot;').replace(/'/g,'&#39;');}

function _timeAgo(ts) {
  const diff = (Date.now()/1000) - ts;
  if (diff < 60)    return 'just now';
  if (diff < 3600)  return Math.floor(diff/60)+'m ago';
  if (diff < 86400) return Math.floor(diff/3600)+'h ago';
  if (diff < 604800)return Math.floor(diff/86400)+'d ago';
  return new Date(ts*1000).toLocaleDateString('en-US',{month:'short',day:'numeric'});
}

// ── Boot ──────────────────────────────────────────────────────────────────────
init();

// ── Knowledge Graph ───────────────────────────────────────────────────────────
let _graphData = { nodes: [], edges: [] };
let _graphSim  = null;
let _graphSvg  = null;
let _graphSearchTimer = null;
let _graphBuildPoller = null;

let _activeGraphNode = null;
let _selectedPaperIds = new Set();

async function initGraphView() {
  await loadGraphStats();
  if (_graphData.nodes.length === 0) {
    await loadGraphData();
  }
}

async function loadGraphStats() {
  try {
    const resp = await fetch('/api/graph/stats');
    const s = await resp.json();
    const el = document.getElementById('graph-stats-bar');
    if (s.entities === 0) {
      el.innerHTML = '<span style="color:var(--amber)">⚠ Graph not built yet — click ⚡ Build Graph to extract entities</span>';
    } else {
      el.textContent =
        `🕸 ${s.entities.toLocaleString()} entities · ${s.relations.toLocaleString()} relations · ` +
        `${s.chunks_with_entities?.toLocaleString()||0} chunks indexed · ` +
        `${s.papers_indexed||0} papers`;
    }
    if (s.running) {
      document.getElementById('graph-progress').style.display = 'flex';
      document.getElementById('graph-progress-msg').textContent = s.message || 'Building…';
      const pct = s.total > 0 ? Math.round((s.progress / s.total) * 100) : 0;
      document.getElementById('graph-progress-bar').style.width = pct + '%';
      document.getElementById('graph-progress-sub').textContent =
        `${s.entities||0} entities · ${s.relations||0} relations`;
    } else {
      document.getElementById('graph-progress').style.display = 'none';
    }
  } catch(e) {
    document.getElementById('graph-stats-bar').textContent = 'Error loading stats';
  }
}

async function loadGraphData(filterName) {
  const svg = document.getElementById('graph-svg');
  svg.innerHTML = '<text x="50%" y="50%" fill="var(--text-dim)" text-anchor="middle" font-size="13" font-family="Quicksand">Loading graph…</text>';
  try {
    const resp = await fetch('/api/graph/data?limit=300');
    _graphData = await resp.json();
  } catch(e) {
    svg.innerHTML = '<text x="50%" y="50%" fill="var(--red)" text-anchor="middle" font-size="13" font-family="Quicksand">Error loading graph</text>';
    return;
  }
  renderGraph(_graphData);
}

function applyGraphFilter() {
  const type = document.getElementById('graph-type-filter').value;
  if (!type) { renderGraph(_graphData); return; }
  const filtered = {
    nodes: _graphData.nodes.filter(n => n.type === type),
    edges: _graphData.edges,
  };
  const nodeIds = new Set(filtered.nodes.map(n => n.id));
  filtered.edges = _graphData.edges.filter(e => nodeIds.has(e.source.id||e.source) && nodeIds.has(e.target.id||e.target));
  renderGraph(filtered);
}

function debounceGraphSearch(val) {
  clearTimeout(_graphSearchTimer);
  if (!val.trim()) { renderGraph(_graphData); return; }
  _graphSearchTimer = setTimeout(async () => {
    const resp = await fetch(`/api/graph/search?q=${encodeURIComponent(val)}`);
    const entities = await resp.json();
    if (!entities.length) return;
    const ids = new Set(entities.map(e => e.entity_id));
    const filtered = {
      nodes: _graphData.nodes.filter(n => ids.has(n.id)),
      edges: _graphData.edges,
    };
    const nodeIds = new Set(filtered.nodes.map(n => n.id));
    filtered.edges = _graphData.edges.filter(e => nodeIds.has(e.source.id||e.source) && nodeIds.has(e.target.id||e.target));
    renderGraph(filtered);
  }, 350);
}

function renderGraph(data) {
  const container = document.getElementById('graph-svg');
  const W = container.clientWidth  || 900;
  const H = container.clientHeight || 600;

  // Clear old
  d3.select('#graph-svg').selectAll('*').remove();

  if (!data.nodes.length) {
    d3.select('#graph-svg').append('text')
      .attr('x', W/2).attr('y', H/2)
      .attr('text-anchor','middle').attr('fill','var(--text-dim)')
      .attr('font-size', 13).attr('font-family', 'Quicksand')
      .text('No entities yet. Build the graph first.');
    return;
  }

  const svg = d3.select('#graph-svg')
    .attr('width', W).attr('height', H);

  // Zoom
  const g = svg.append('g');
  svg.call(d3.zoom().scaleExtent([0.1, 8]).on('zoom', e => g.attr('transform', e.transform)));

  // Prep data — clone for simulation
  const nodes = data.nodes.map(d => ({...d}));
  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  const links = data.edges
    .filter(e => nodeMap.has(e.source.id||e.source) && nodeMap.has(e.target.id||e.target))
    .map(e => ({...e, source: e.source.id||e.source, target: e.target.id||e.target}));

  // Force simulation
  const sim = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(80))
    .force('charge', d3.forceManyBody().strength(-120))
    .force('center', d3.forceCenter(W/2, H/2))
    .force('collision', d3.forceCollide(d => nodeRadius(d) + 4));
  _graphSim = sim;

  // Edges
  const link = g.append('g').selectAll('line')
    .data(links).join('line')
    .attr('class', 'graph-link')
    .attr('stroke-opacity', 0.4);

  // Nodes
  const node = g.append('g').selectAll('g')
    .data(nodes).join('g')
    .attr('class', 'graph-node')
    .call(d3.drag()
      .on('start', (e,d) => { if(!e.active) sim.alphaTarget(.3).restart(); d.fx=d.x; d.fy=d.y; })
      .on('drag',  (e,d) => { d.fx=e.x; d.fy=e.y; })
      .on('end',   (e,d) => { if(!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; })
    )
    .on('click', (e,d) => { e.stopPropagation(); showNodeDetail(d, links, nodeMap); });

  node.append('circle')
    .attr('r', d => nodeRadius(d))
    .attr('fill', d => d.color || '#6b7280');

  node.append('text')
    .attr('dy', d => nodeRadius(d) + 10)
    .attr('text-anchor', 'middle')
    .text(d => d.name.length > 18 ? d.name.slice(0,16)+'…' : d.name);

  // Click background to deselect
  svg.on('click', () => closeGraphDetail());

  sim.on('tick', () => {
    link
      .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    node.attr('transform', d => `translate(${d.x},${d.y})`);
  });
}

function nodeRadius(d) {
  return Math.max(5, Math.min(18, 4 + Math.sqrt(d.weight || 1) * 2));
}

function showNodeDetail(node, links, nodeMap) {
  const panel = document.getElementById('graph-detail');
  document.getElementById('graph-detail-name').textContent = node.name;
  const badge = document.getElementById('graph-detail-type-badge');
  badge.textContent = node.type;
  badge.style.background = node.color + '33';
  badge.style.color = node.color;
  document.getElementById('graph-detail-desc').textContent = node.description || 'No description.';

  // Relations
  const relContainer = document.getElementById('graph-detail-relations');
  const nodeLinks = links.filter(l =>
    (l.source.id||l.source) === node.id || (l.target.id||l.target) === node.id
  ).slice(0, 12);

  if (nodeLinks.length) {
    relContainer.innerHTML = '<div style="font-size:11px;font-weight:600;color:var(--text-dim);margin-bottom:4px">RELATIONS</div>';
    nodeLinks.forEach(l => {
      const isSource = (l.source.id||l.source) === node.id;
      const other = nodeMap.get(isSource ? (l.target.id||l.target) : (l.source.id||l.source));
      const otherName = other ? other.name : '?';
      const div = document.createElement('div');
      div.className = 'graph-relation-item';
      div.innerHTML = isSource
        ? `<span style="color:var(--text)">${escHtml(node.name)}</span><span class="graph-relation-arrow">→</span><span class="graph-relation-label">${escHtml(l.relation)}</span><span class="graph-relation-arrow">→</span><span style="color:var(--accent)">${escHtml(otherName)}</span>`
        : `<span style="color:var(--accent)">${escHtml(otherName)}</span><span class="graph-relation-arrow">→</span><span class="graph-relation-label">${escHtml(l.relation)}</span><span class="graph-relation-arrow">→</span><span style="color:var(--text)">${escHtml(node.name)}</span>`;
      relContainer.appendChild(div);
    });
  } else {
    relContainer.innerHTML = '<div style="font-size:11px;color:var(--text-dim)">No relations in current view.</div>';
  }

  // Papers
  const paperIds = JSON.parse(node.paper_ids || '[]');
  const paperEl = document.getElementById('graph-detail-papers');
  if (paperIds.length) {
    paperEl.innerHTML = `<div style="font-size:11px;font-weight:600;color:var(--text-dim);margin-bottom:4px">IN ${paperIds.length} PAPER(S)</div>`;
    paperIds.slice(0,4).forEach(pid => {
      const chip = document.createElement('div');
      chip.style.cssText = 'font-size:10px;color:var(--text-dim);padding:2px 0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap';
      chip.textContent = '📄 ' + pid.replace(/_/g,' ');
      chip.title = pid;
      chip.style.cursor = 'pointer';
      chip.onclick = () => openPaper(pid, pid);
      paperEl.appendChild(chip);
    });
  }

  panel.style.display = 'block';
}

function closeGraphDetail() {
  document.getElementById('graph-detail').style.display = 'none';
}

async function buildGraph() {
  const btn = document.getElementById('graph-build-btn');
  btn.disabled = true;
  btn.textContent = '⚙ Building…';
  document.getElementById('graph-progress').style.display = 'flex';
  document.getElementById('graph-progress-msg').textContent = 'Starting extraction…';
  document.getElementById('graph-progress-bar').style.width = '0%';
  document.getElementById('graph-progress-sub').textContent = '';

  try {
    const resp = await fetch('/api/graph/build', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'});
    const data = await resp.json();
    if (!resp.ok && resp.status !== 409) throw new Error(data.error);
  } catch(e) {
    alert('Failed to start: ' + e.message);
    document.getElementById('graph-progress').style.display = 'none';
    btn.disabled = false; btn.textContent = '⚡ Build Graph';
    return;
  }

  // Poll for progress
  clearInterval(_graphBuildPoller);
  _graphBuildPoller = setInterval(async () => {
    await loadGraphStats();
    const resp2 = await fetch('/api/graph/stats');
    const s = await resp2.json();
    if (!s.running) {
      clearInterval(_graphBuildPoller);
      btn.disabled = false; btn.textContent = '⚡ Build Graph';
      document.getElementById('graph-progress').style.display = 'none';
      await loadGraphData();  // Reload the graph
      await loadGraphStats();
    } else {
      const pct = s.total > 0 ? Math.round((s.progress / s.total) * 100) : 0;
      document.getElementById('graph-progress-bar').style.width = pct + '%';
      document.getElementById('graph-progress-msg').textContent = s.message || 'Processing…';
      document.getElementById('graph-progress-sub').textContent =
        `${s.entities||0} entities · ${s.relations||0} relations`;
    }
  }, 3000);
}

function togglePaperSelection(paperId, title) {
  if (_selectedPaperIds.has(paperId)) {
    _selectedPaperIds.delete(paperId);
  } else {
    _selectedPaperIds.add(paperId);
  }
  updatePinnedStrip();
}

function updatePinnedStrip() {
  let strip = document.getElementById('pinned-strip');
  if (!strip) {
    strip = document.createElement('div');
    strip.id = 'pinned-strip';
    strip.style.cssText = 'display:flex;gap:8px;padding:0 24px 12px;flex-wrap:wrap;background:var(--surface);';
    const inputBar = document.querySelector('.input-bar');
    inputBar.parentNode.insertBefore(strip, inputBar);
  }
  strip.innerHTML = '';
  if (_selectedPaperIds.size === 0) {
    strip.style.display = 'none';
    return;
  }
  strip.style.display = 'flex';
  const label = document.createElement('div');
  label.style.cssText = 'font-size:11px;font-weight:700;color:var(--text-dim);width:100%;text-transform:uppercase;';
  label.textContent = 'Targeted Papers:';
  strip.appendChild(label);

  _selectedPaperIds.forEach(id => {
    const chip = document.createElement('div');
    chip.className = 'upload-chip';
    chip.innerHTML = `📌 ${escHtml(id)} <span class="upload-chip-remove" onclick="togglePaperSelection('${id.replace(/'/g, "\\'")}')">×</span>`;
    strip.appendChild(chip);
  });
}