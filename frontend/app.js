/* ═══════════════════════════════════════════════════════════════════
   MusicGenAI – Frontend Application Logic
══════════════════════════════════════════════════════════════════════ */

'use strict';

// ── State ─────────────────────────────────────────────────────────────
const state = {
  currentSessionId: null,
  isGenerating: false,
  sessions: [],        // [{id, title, updated_at, message_count}]
  searchQuery: '',
};

// ── DOM refs ──────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const dom = {
  sidebar:        $('sidebar'),
  sidebarOverlay: $('sidebarOverlay'),
  historyList:    $('historyList'),
  searchInput:    $('searchInput'),
  welcomeScreen:  $('welcomeScreen'),
  messages:       $('messages'),
  chatWrapper:    $('chatWrapper'),
  promptInput:    $('promptInput'),
  sendBtn:        $('sendBtn'),
  charCount:      $('charCount'),
  chatTitle:      $('chatTitle'),
  modalOverlay:   $('modalOverlay'),
  modalTitle:     $('modalTitle'),
  modalMsg:       $('modalMsg'),
  modalCancel:    $('modalCancel'),
  modalConfirm:   $('modalConfirm'),
  toast:          $('toast'),
};

let modalResolve = null;

// ═══════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  loadHistory();
  bindEvents();
  adjustTextarea();
});

// ═══════════════════════════════════════════════════════════════════
// EVENT BINDING
// ═══════════════════════════════════════════════════════════════════
function bindEvents() {
  // Send on Enter (Shift+Enter = new line)
  dom.promptInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });
  dom.promptInput.addEventListener('input', () => {
    adjustTextarea();
    updateCharCount();
  });

  dom.sendBtn.addEventListener('click', handleSend);

  // New chat buttons
  $('newChatBtn').addEventListener('click', startNewChat);
  $('newChatTopBtn').addEventListener('click', startNewChat);

  // Clear all history
  $('clearAllBtn').addEventListener('click', async () => {
    const ok = await confirm('Xóa toàn bộ lịch sử', 'Tất cả hội thoại sẽ bị xóa vĩnh viễn. Bạn có chắc không?');
    if (!ok) return;
    await fetch('/api/history', { method: 'DELETE' });
    state.sessions = [];
    state.currentSessionId = null;
    renderHistory();
    startNewChat();
    showToast('Đã xóa toàn bộ lịch sử', 'success');
  });

  // Sidebar toggle (mobile)
  $('menuToggle').addEventListener('click', toggleSidebar);
  dom.sidebarOverlay.addEventListener('click', closeSidebar);

  // Modal
  dom.modalCancel.addEventListener('click', () => modalResolve && modalResolve(false));
  dom.modalOverlay.addEventListener('click', e => {
    if (e.target === dom.modalOverlay) modalResolve && modalResolve(false);
  });

  // Example chips
  document.querySelectorAll('.example-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      dom.promptInput.value = chip.dataset.prompt;
      adjustTextarea();
      updateCharCount();
      dom.promptInput.focus();
    });
  });

  // Search
  dom.searchInput.addEventListener('input', () => {
    state.searchQuery = dom.searchInput.value.trim().toLowerCase();
    renderHistory();
  });
}

// ═══════════════════════════════════════════════════════════════════
// SEND / GENERATE
// ═══════════════════════════════════════════════════════════════════
async function handleSend() {
  const prompt = dom.promptInput.value.trim();
  if (!prompt || state.isGenerating) return;

  // Hide welcome screen, show messages
  dom.welcomeScreen.style.display = 'none';
  dom.messages.style.display = 'flex';

  state.isGenerating = true;
  dom.sendBtn.disabled = true;
  dom.promptInput.disabled = true;

  // Add user message
  appendUserMessage(prompt);
  dom.promptInput.value = '';
  adjustTextarea();
  updateCharCount();
  scrollToBottom();

  // Show loading
  const loadingEl = appendLoading();

  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, session_id: state.currentSessionId }),
    });
    const data = await res.json();

    loadingEl.remove();

    if (!res.ok || data.status === 'error') {
      appendError(data.error || data.detail || 'Có lỗi xảy ra khi generate nhạc');
    } else {
      state.currentSessionId = data.session_id;
      appendAssistantMessage(data);
      dom.chatTitle.textContent = prompt.split(' ').slice(0, 5).join(' ') + (prompt.split(' ').length > 5 ? '...' : '');
      await loadHistory(); // refresh sidebar
      setActiveSession(state.currentSessionId);
    }
  } catch (err) {
    loadingEl.remove();
    appendError('Không thể kết nối tới server. Hãy kiểm tra backend đang chạy.');
  } finally {
    state.isGenerating = false;
    dom.sendBtn.disabled = false;
    dom.promptInput.disabled = false;
    dom.promptInput.focus();
    scrollToBottom();
  }
}

// ═══════════════════════════════════════════════════════════════════
// MESSAGE RENDERERS
// ═══════════════════════════════════════════════════════════════════
function appendUserMessage(text) {
  const div = document.createElement('div');
  div.className = 'msg-user';
  div.innerHTML = `<div class="msg-user-bubble">${escapeHtml(text)}</div>`;
  dom.messages.appendChild(div);
}

function appendAssistantMessage(data) {
  const isDemo = data.metadata?.mode === 'demo';
  const duration = data.duration ? formatDuration(data.duration) : '—';
  const timeStr = new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });

  const div = document.createElement('div');
  div.className = 'msg-assistant';
  div.innerHTML = `
    <div class="msg-avatar">🎵</div>
    <div class="msg-content">
      <div class="audio-card">
        <div class="audio-card-header">
          <div class="audio-card-icon">🎼</div>
          <div class="audio-card-info">
            <div class="audio-card-title">Bản nhạc vừa tạo</div>
            <div class="audio-card-meta">Thời lượng: ${duration}</div>
          </div>
        </div>
        ${isDemo
          ? `<div class="demo-notice">
               ⚠️ Chế độ demo – Tích hợp model thật để nghe nhạc
             </div>`
          : `<audio controls preload="none">
               <source src="${escapeHtml(data.audio_url)}" type="audio/wav">
               Trình duyệt không hỗ trợ audio.
             </audio>`
        }
      </div>
      <div class="msg-timestamp">${timeStr}</div>
    </div>`;
  dom.messages.appendChild(div);
}

function appendLoading() {
  const div = document.createElement('div');
  div.className = 'loading-indicator';
  div.innerHTML = `
    <div class="msg-avatar">🎵</div>
    <div class="loading-bubble">
      <div class="dots"><span></span><span></span><span></span></div>
      <span class="loading-text">Đang tạo nhạc, vui lòng chờ...</span>
    </div>`;
  dom.messages.appendChild(div);
  scrollToBottom();
  return div;
}

function appendError(msg) {
  const div = document.createElement('div');
  div.className = 'msg-error';
  div.innerHTML = `
    <div class="msg-avatar">🎵</div>
    <div class="error-bubble">⚠️ ${escapeHtml(msg)}</div>`;
  dom.messages.appendChild(div);
}

// ═══════════════════════════════════════════════════════════════════
// HISTORY
// ═══════════════════════════════════════════════════════════════════
async function loadHistory() {
  try {
    const res = await fetch('/api/history');
    state.sessions = await res.json();
    renderHistory();
  } catch {
    // server not ready yet – ignore
  }
}

function renderHistory() {
  const query = state.searchQuery;
  const filtered = query
    ? state.sessions.filter(s => s.title.toLowerCase().includes(query))
    : state.sessions;

  if (filtered.length === 0) {
    dom.historyList.innerHTML = `<p class="history-empty">${query ? 'Không tìm thấy kết quả' : 'Chưa có lịch sử nào'}</p>`;
    return;
  }

  // Group by date
  const groups = {};
  filtered.forEach(s => {
    const label = dateLabel(s.updated_at);
    if (!groups[label]) groups[label] = [];
    groups[label].push(s);
  });

  let html = '';
  for (const [label, items] of Object.entries(groups)) {
    html += `<div class="history-group-label">${label}</div>`;
    items.forEach(s => {
      const isActive = s.id === state.currentSessionId;
      html += `
        <div class="history-item ${isActive ? 'active' : ''}" data-id="${s.id}">
          <div class="history-item-content">
            <div class="history-item-title">${escapeHtml(s.title)}</div>
            <div class="history-item-meta">${s.message_count} bản nhạc</div>
          </div>
          <button class="history-item-delete" data-id="${s.id}" title="Xóa">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="3 6 5 6 21 6"/>
              <path d="M19 6l-1 14H6L5 6"/>
            </svg>
          </button>
        </div>`;
    });
  }
  dom.historyList.innerHTML = html;

  // Bind click events
  dom.historyList.querySelectorAll('.history-item').forEach(el => {
    el.addEventListener('click', e => {
      if (e.target.closest('.history-item-delete')) return;
      loadSession(el.dataset.id);
    });
  });
  dom.historyList.querySelectorAll('.history-item-delete').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      deleteSession(btn.dataset.id);
    });
  });
}

async function loadSession(sessionId) {
  try {
    const res = await fetch(`/api/history/${sessionId}`);
    const session = await res.json();

    state.currentSessionId = sessionId;
    dom.chatTitle.textContent = session.title;

    // Clear and rebuild messages
    dom.messages.innerHTML = '';
    dom.welcomeScreen.style.display = 'none';
    dom.messages.style.display = 'flex';

    const msgs = session.messages || [];
    for (let i = 0; i < msgs.length; i++) {
      const m = msgs[i];
      if (m.role === 'user') {
        appendUserMessage(m.content);
      } else {
        appendAssistantMessage({
          audio_url: m.audio_url,
          duration: m.duration,
          metadata: m.metadata,
        });
      }
    }
    setActiveSession(sessionId);
    scrollToBottom();
    closeSidebar();
  } catch {
    showToast('Không thể tải hội thoại', 'error');
  }
}

async function deleteSession(sessionId) {
  const s = state.sessions.find(x => x.id === sessionId);
  const ok = await confirm('Xóa hội thoại', `Xóa "${s?.title || 'hội thoại này'}"?`);
  if (!ok) return;

  await fetch(`/api/history/${sessionId}`, { method: 'DELETE' });
  state.sessions = state.sessions.filter(x => x.id !== sessionId);

  if (state.currentSessionId === sessionId) {
    startNewChat();
  }
  renderHistory();
  showToast('Đã xóa hội thoại', 'success');
}

function setActiveSession(id) {
  dom.historyList.querySelectorAll('.history-item').forEach(el => {
    el.classList.toggle('active', el.dataset.id === id);
  });
}

// ═══════════════════════════════════════════════════════════════════
// NEW CHAT
// ═══════════════════════════════════════════════════════════════════
function startNewChat() {
  state.currentSessionId = null;
  dom.messages.innerHTML = '';
  dom.messages.style.display = 'none';
  dom.welcomeScreen.style.display = '';
  dom.chatTitle.textContent = 'MusicGenAI';
  setActiveSession(null);
  dom.promptInput.focus();
  closeSidebar();
}

// ═══════════════════════════════════════════════════════════════════
// SIDEBAR
// ═══════════════════════════════════════════════════════════════════
function toggleSidebar() {
  dom.sidebar.classList.toggle('open');
  dom.sidebarOverlay.classList.toggle('show');
}
function closeSidebar() {
  dom.sidebar.classList.remove('open');
  dom.sidebarOverlay.classList.remove('show');
}

// ═══════════════════════════════════════════════════════════════════
// MODAL (confirm dialog)
// ═══════════════════════════════════════════════════════════════════
function confirm(title, msg) {
  dom.modalTitle.textContent = title;
  dom.modalMsg.textContent = msg;
  dom.modalOverlay.classList.add('show');

  return new Promise(resolve => {
    modalResolve = (val) => {
      dom.modalOverlay.classList.remove('show');
      modalResolve = null;
      resolve(val);
    };
    dom.modalConfirm.onclick = () => modalResolve(true);
    dom.modalCancel.onclick   = () => modalResolve(false);
  });
}

// ═══════════════════════════════════════════════════════════════════
// TOAST
// ═══════════════════════════════════════════════════════════════════
let toastTimer;
function showToast(msg, type = '') {
  dom.toast.textContent = msg;
  dom.toast.className = `toast ${type} show`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => dom.toast.classList.remove('show'), 3000);
}

// ═══════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════
function adjustTextarea() {
  const t = dom.promptInput;
  t.style.height = 'auto';
  t.style.height = Math.min(t.scrollHeight, 180) + 'px';
}

function updateCharCount() {
  const len = dom.promptInput.value.length;
  dom.charCount.textContent = `${len}/500`;
  dom.charCount.style.color = len > 450 ? '#f59e0b' : '';
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    dom.chatWrapper.scrollTop = dom.chatWrapper.scrollHeight;
  });
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatDuration(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

function dateLabel(isoStr) {
  if (!isoStr) return 'Khác';
  const date = new Date(isoStr);
  const today = new Date();
  const diffDays = Math.floor((today - date) / 86400000);
  if (diffDays === 0) return 'Hôm nay';
  if (diffDays === 1) return 'Hôm qua';
  if (diffDays < 7)  return '7 ngày qua';
  return '30 ngày qua';
}
