/* ═══════════════════════════════════════════════════════════
   ESG Insight Pro — Chat Logic (Bhagat Labs)
   ═══════════════════════════════════════════════════════════ */

const API_BASE = '';
let currentThreadId = 'session-' + Math.random().toString(36).substring(2, 10);
let detailsVisible = false;

// ── Core Functions ──

function fillQuery(text) {
    document.getElementById('queryInput').value = text;
    document.getElementById('queryInput').focus();
}

function newChat() {
    currentThreadId = 'session-' + Math.random().toString(36).substring(2, 10);
    const container = document.getElementById('messagesContainer');
    container.innerHTML = document.getElementById('welcomeScreen') ?
        '' : '';
    // Re-add welcome screen
    container.innerHTML = `
        <div class="welcome-screen" id="welcomeScreen">
            <div class="welcome-icon">
                <svg viewBox="0 0 48 48" fill="none"><rect width="48" height="48" rx="12" fill="url(#wG)"/><path d="M12 24C12 17.37 17.37 12 24 12C30.63 12 36 17.37 36 24" stroke="white" stroke-width="3" stroke-linecap="round"/><path d="M18 30C18 26.69 20.69 24 24 24C27.31 24 30 26.69 30 30" stroke="white" stroke-width="3" stroke-linecap="round"/><circle cx="24" cy="36" r="2" fill="white"/><defs><linearGradient id="wG" x1="0" y1="0" x2="48" y2="48"><stop stop-color="#10b981"/><stop offset="1" stop-color="#3b82f6"/></linearGradient></defs></svg>
            </div>
            <h2>ESG Insight Pro</h2>
            <p>Your AI-powered ESG report analyst. Ask questions about sustainability reports, carbon emissions, ESG metrics, and more.</p>
        </div>`;
}

function toggleDetails() {
    const panel = document.getElementById('detailsPanel');
    detailsVisible = !detailsVisible;
    panel.style.display = detailsVisible ? 'flex' : 'none';
}

async function submitQuery() {
    const input = document.getElementById('queryInput');
    const query = input.value.trim();
    if (!query) return;

    const btn = document.getElementById('submitBtn');
    btn.disabled = true;
    setStatus('Processing...', 'busy');

    // Hide welcome screen
    const welcome = document.getElementById('welcomeScreen');
    if (welcome) welcome.remove();

    // Add user message
    addMessage('user', query);
    input.value = '';
    autoResize(input);

    // Add loading indicator
    const loadingId = addLoadingMessage();

    try {
        const response = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                thread_id: currentThreadId,
                user_id: 'default',
            }),
        });

        removeMessage(loadingId);

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Query failed');
        }

        const data = await response.json();
        addAIMessage(data);
        updateDetailsPanel(data);

    } catch (error) {
        removeMessage(loadingId);
        addMessage('ai', `**Error:** ${error.message}`, true);
    } finally {
        btn.disabled = false;
        setStatus('System Ready', 'online');
    }
}

// ── Message Rendering ──

function addMessage(role, text, isError = false) {
    const container = document.getElementById('messagesContainer');
    const id = 'msg-' + Date.now();
    const avatarClass = role === 'user' ? 'user-avatar' : 'ai-avatar';
    const avatarText = role === 'user' ? 'U' : 'B';
    const label = role === 'user' ? 'You' : 'ESG Insight Pro';
    const rowClass = role === 'user' ? 'user' : 'ai';

    const html = `
        <div class="message-row ${rowClass}" id="${id}">
            <div class="message-inner">
                <div class="message-avatar ${avatarClass}">${avatarText}</div>
                <div class="message-content">
                    <div class="message-label">${label}</div>
                    <div class="message-text">${isError ? `<p style="color:var(--red)">${esc(text)}</p>` : formatText(text)}</div>
                </div>
            </div>
        </div>`;

    container.insertAdjacentHTML('beforeend', html);
    container.scrollTop = container.scrollHeight;
    return id;
}

function addAIMessage(data) {
    const container = document.getElementById('messagesContainer');
    const id = 'msg-' + Date.now();

    const conf = data.confidence_score || 0;
    const confClass = conf >= 0.7 ? 'conf-high' : conf >= 0.4 ? 'conf-med' : 'conf-low';
    const confText = `${(conf * 100).toFixed(0)}% confidence`;

    let gapsHtml = '';
    if (data.information_gaps && data.information_gaps.length > 0) {
        gapsHtml = `<div class="info-gaps-inline"><strong>Info gaps:</strong> ${data.information_gaps.map(g => esc(g)).join('; ')}</div>`;
    }

    const html = `
        <div class="message-row ai" id="${id}">
            <div class="message-inner">
                <div class="message-avatar ai-avatar">B</div>
                <div class="message-content">
                    <div class="message-label">ESG Insight Pro</div>
                    <div class="message-text">${formatAnswer(data.answer)}</div>
                    <span class="confidence-bar ${confClass}">${confText}</span>
                    ${gapsHtml}
                </div>
            </div>
        </div>`;

    container.insertAdjacentHTML('beforeend', html);
    container.scrollTop = container.scrollHeight;
}

function addLoadingMessage() {
    const container = document.getElementById('messagesContainer');
    const id = 'loading-' + Date.now();

    const html = `
        <div class="message-row ai" id="${id}">
            <div class="message-inner">
                <div class="message-avatar ai-avatar">B</div>
                <div class="message-content">
                    <div class="message-label">ESG Insight Pro</div>
                    <div class="typing-indicator"><span></span><span></span><span></span></div>
                </div>
            </div>
        </div>`;

    container.insertAdjacentHTML('beforeend', html);
    container.scrollTop = container.scrollHeight;
    return id;
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// ── Details Panel ──

function updateDetailsPanel(data) {
    // Sources
    const sourcesList = document.getElementById('sourcesList');
    const citations = data.citations || [];
    if (citations.length > 0) {
        sourcesList.innerHTML = citations.map(c =>
            `<div class="source-chip">
                <span class="doc">${esc(c.document || 'Unknown')}</span>
                <span class="page">Page ${c.page || '?'}</span>
            </div>`
        ).join('');
    } else {
        sourcesList.innerHTML = '<p class="empty-state">No citations in this response</p>';
    }

    // SQL
    const sqlSection = document.getElementById('detailSQL');
    if (data.generated_sql) {
        sqlSection.style.display = 'block';
        document.getElementById('sqlDisplay').textContent = data.generated_sql;
    } else {
        sqlSection.style.display = 'none';
    }

    // Trace
    const traceList = document.getElementById('traceList');
    const trace = data.execution_trace || [];
    if (trace.length > 0) {
        traceList.innerHTML = trace.map(s =>
            `<div class="trace-step-row">
                <span class="trace-agent-name">${esc(s.agent || '?')}</span>
                <span class="trace-summary">${esc(s.output_summary || s.action || '')}</span>
                <span class="trace-time">${s.duration_ms ? s.duration_ms.toFixed(0) + 'ms' : ''}</span>
            </div>`
        ).join('');
    }

    // Meta
    const metaGrid = document.getElementById('metaGrid');
    metaGrid.innerHTML = [
        { label: 'Type', value: data.query_type },
        { label: 'Scope', value: data.query_scope },
        { label: 'Strategy', value: data.retrieval_strategy },
        { label: 'Duration', value: data.duration_ms ? data.duration_ms.toFixed(0) + 'ms' : '-' },
    ].map(m =>
        `<div class="meta-item"><div class="label">${m.label}</div><div class="value">${esc(m.value || '-')}</div></div>`
    ).join('');

    // Auto-show panel if not visible
    if (!detailsVisible) toggleDetails();
}

// ── Formatting ──

function formatAnswer(text) {
    if (!text) return '<p class="empty-state">No answer generated</p>';
    return text
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\[(.*?),\s*Page\s*(\d+)\]/g, '<span class="citation">[$1, Page $2]</span>')
        .replace(/^/, '<p>').replace(/$/, '</p>');
}

function formatText(text) {
    return `<p>${esc(text)}</p>`;
}

function esc(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function setStatus(text, type) {
    const el = document.getElementById('systemStatus');
    const dot = el.querySelector('.status-dot');
    const label = el.querySelector('span:last-child');
    label.textContent = text;
    dot.className = 'status-dot ' + type;
}

// ── Auto-resize textarea ──

function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 150) + 'px';
}

document.getElementById('queryInput').addEventListener('input', function() {
    autoResize(this);
});

document.getElementById('queryInput').addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        submitQuery();
    }
    if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        submitQuery();
    }
});
