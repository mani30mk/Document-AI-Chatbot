    // ── Config ────────────────────────────────────────────────
    const RENDER_BACKEND = 'https://document-ai-chatbot-71ey.onrender.com';
    const LOCAL_BACKEND = 'http://localhost:8000';

    function getApiBase() {
      const origin = window.location.origin;
      if (origin && origin.startsWith('http')) {
        const host = window.location.hostname;
        // Deployed in production on Render or custom domain
        if (host !== 'localhost' && host !== '127.0.0.1') {
          return origin;
        }
        // Localhost running directly on port 8000
        if (window.location.port === '8000') {
          return origin;
        }
        // Opened via local dev server (port 5500, 3000, etc.)
        return LOCAL_BACKEND;
      }
      // Opened via file:///
      return RENDER_BACKEND;
    }

    let API_BASE = getApiBase();

    // Embedding service URL — set this to your Render embedding service URL
    // When empty, warm-on-demand only pings the main app
    const EMBED_SERVICE_URL = 'https://document-ai-chatbot-7ch2.onrender.com';

    /** Fire-and-forget wake-up pings to embedding service + main app */
    function warmUpServices() {
      // Wake main app
      fetch(`${API_BASE}/health`).catch(() => { });
      // Wake embedding service (if configured)
      if (EMBED_SERVICE_URL) {
        fetch(`${EMBED_SERVICE_URL}/health`, { mode: 'no-cors' }).catch(() => { });
      }
    }

    // ── State ─────────────────────────────────────────────────
    let sessionId = null;
    let chatOpen = false;
    let turnCount = 0;
    let files = [];   // { name, size, ext, slides, text, url, rawFile }
    let activeIdx = null;
    let isSending = false;
    let chatHistory = []; // [ { role, content, youtube_sources, sources_used, timestamp } ]
    let currentPptMode = 'original'; // 'original' | 'google' | 'outline'
    let currentPptFilename = '';
    let currentPptSlides = [];
    let currentPptIndex = 0;

    // ── LocalStorage Persistence ──────────────────────────────
    function saveLocalState() {
      try {
        if (sessionId) localStorage.setItem('study_rag_session_id', sessionId);
        localStorage.setItem('study_rag_chat_history', JSON.stringify(chatHistory));

        const storableFiles = files.map(f => ({
          name: f.name,
          size: f.size || 0,
          ext: f.ext,
          slides: f.slides || null,
          text: f.text || null,
        }));
        localStorage.setItem('study_rag_files', JSON.stringify(storableFiles));

        if (activeIdx !== null && activeIdx >= 0) {
          localStorage.setItem('study_rag_active_idx', activeIdx.toString());
        } else {
          localStorage.removeItem('study_rag_active_idx');
        }
      } catch (e) {
        console.warn('Could not save to localStorage:', e);
      }
    }

    function getDeviceId() {
      let id = localStorage.getItem('device_id');
      if (!id) {
        id = (typeof crypto !== 'undefined' && crypto.randomUUID)
          ? crypto.randomUUID()
          : 'dev_' + Math.random().toString(36).substring(2, 15) + Date.now().toString(36);
        localStorage.setItem('device_id', id);
      }
      return id;
    }

    function formatRelativeTime(dateInput) {
      if (!dateInput) return '';
      const now = new Date();
      const past = new Date(dateInput);
      const diffSec = Math.floor((now - past) / 1000);
      if (isNaN(diffSec) || diffSec < 0) return 'Just now';
      if (diffSec < 60) return 'Just now';
      const diffMin = Math.floor(diffSec / 60);
      if (diffMin < 60) return `${diffMin}m ago`;
      const diffHour = Math.floor(diffMin / 60);
      if (diffHour < 24) return `${diffHour}h ago`;
      const diffDay = Math.floor(diffHour / 24);
      if (diffDay === 1) return 'Yesterday';
      if (diffDay < 7) return `${diffDay}d ago`;
      const diffWeek = Math.floor(diffDay / 7);
      if (diffWeek < 4) return `${diffWeek}w ago`;
      return past.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    }

    function toggleHistorySidebar() {
      const sb = document.getElementById('historySidebar');
      if (!sb) return;
      const isCollapsed = sb.classList.toggle('collapsed');
      try {
        localStorage.setItem('study_rag_sidebar_collapsed', isCollapsed ? '1' : '0');
      } catch (_) { }
    }

    let sidebarSessions = [];

    async function loadSidebarSessions() {
      const container = document.getElementById('historySessionList');
      if (!container) return;

      try {
        const devId = getDeviceId();
        const res = await fetch(`${API_BASE}/sessions?device_id=${encodeURIComponent(devId)}`);
        if (!res.ok) {
          container.innerHTML = '<div class="history-empty">Could not load chats</div>';
          return;
        }
        const list = await res.json();
        sidebarSessions = Array.isArray(list) ? list : [];

        if (sidebarSessions.length === 0) {
          container.innerHTML = '<div class="history-empty">No past sessions on this device yet.<br/>Upload files or ask questions to save one!</div>';
          return;
        }

        container.innerHTML = sidebarSessions.map(s => {
          const isActive = s.session_id === sessionId;
          const relTime = formatRelativeTime(s.updated_at);
          const safeTitle = escapeHtml(s.title || 'New chat');
          return `
        <div class="history-item ${isActive ? 'active' : ''}" 
             data-sid="${s.session_id}" 
             onclick="loadSessionById('${s.session_id}')" 
             title="${safeTitle}">
          <div class="history-item-top">
            <span class="history-item-icon"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg></span>
            <span class="history-item-title">${safeTitle}</span>
            <button class="session-delete-btn" onclick="deleteSession('${s.session_id}', event)" title="Delete session"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg></button>
          </div>
          <div class="history-item-time">${relTime}</div>
        </div>
      `;
        }).join('');
      } catch (err) {
        console.warn('loadSidebarSessions error:', err);
        if (container) {
          container.innerHTML = '<div class="history-empty">Failed to load chat history</div>';
        }
      }
    }

    async function deleteSession(sessionIdToDelete, event) {
      if (event) event.stopPropagation();
      if (!confirm('Delete this chat session? This cannot be undone.')) return;

      try {
        const res = await fetch(`${API_BASE}/session/${sessionIdToDelete}`, { method: 'DELETE' });
        if (!res.ok) throw new Error('Failed to delete session');

        // If the deleted session is the one currently open, start a fresh one
        if (sessionIdToDelete === sessionId) {
          await createNewSession(true);
        } else {
          await loadSidebarSessions();
        }
        showToast('Session deleted.');
      } catch (e) {
        console.error('deleteSession error:', e);
        showToast('Could not delete session. Please try again.');
      }
    }

    function highlightActiveSessionInSidebar() {
      const items = document.querySelectorAll('.history-item');
      items.forEach(el => {
        if (el.getAttribute('data-sid') === sessionId) {
          el.classList.add('active');
        } else {
          el.classList.remove('active');
        }
      });
    }

    async function loadSessionById(sid) {
      if (!sid) return;
      if (sid === sessionId) {
        highlightActiveSessionInSidebar();
        return;
      }

      try {
        showToast('Loading session…');
        const res = await fetch(`${API_BASE}/session/${sid}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        sessionId = data.session_id;
        localStorage.setItem('study_rag_session_id', sessionId);
        setStatus(true);

        // Clean up previous blob URLs
        files.forEach(f => {
          if (f.url && f.url.startsWith('blob:')) {
            try { URL.revokeObjectURL(f.url); } catch (_) { }
          }
        });

        // Reconstruct files
        files = [];
        if (data.files && data.files.length > 0) {
          data.files.forEach(fname => {
            files.push({
              name: fname,
              size: 0,
              ext: fname.split('.').pop().toLowerCase(),
              slides: (data.slides && data.slides[fname]) ? data.slides[fname] : null,
              text: (data.docs && data.docs[fname]) ? data.docs[fname] : null,
              url: null,
              rawFile: null,
            });
          });
          renderFileList();
          document.getElementById('fileCountTag').textContent = files.length + ' file' + (files.length !== 1 ? 's' : '');
          selectFile(0);
        } else {
          renderFileList();
          document.getElementById('fileCountTag').textContent = '0 files';
          document.getElementById('viewer').innerHTML = '<div class="viewer-empty"><p>Upload files and click one to preview it</p></div>';
          document.getElementById('contentTitle').textContent = 'Select a file to preview';
          activeIdx = null;
          currentPptSlides = [];
        }

        // Reconstruct chat history
        if (data.history && data.history.length > 0) {
          chatHistory = data.history.map(item => ({
            role: (item.role === 'assistant' || item.role === 'bot') ? 'bot' : 'user',
            content: item.content,
            sources_used: item.sources_used || 0,
            youtube_sources: item.youtube_sources || [],
            timestamp: item.timestamp || Date.now(),
          }));
          turnCount = chatHistory.filter(h => h.role === 'user').length;
          document.getElementById('historyBadge').textContent = `${turnCount} turn${turnCount !== 1 ? 's' : ''}`;
          renderChatHistory();
        } else {
          chatHistory = [];
          turnCount = 0;
          document.getElementById('historyBadge').textContent = '0 turns';
          document.getElementById('chatMessages').innerHTML = '<div class="bubble bot">Hi! Upload your study files on the left, then ask me anything about them.</div>';
        }

        // Reset summary panel
        const summaryTitle = document.getElementById('summaryDocTitle');
        if (summaryTitle) summaryTitle.textContent = 'Document Summary';
        const summaryText = document.getElementById('summaryPanelText');
        if (summaryText) summaryText.textContent = 'Select or upload a document and switch to this tab to see an AI-generated summary.';

        saveLocalState();
        highlightActiveSessionInSidebar();
        showToast('Session loaded');
      } catch (e) {
        console.error('Failed to load session:', e);
        showToast('Failed to load session history');
      }
    }

    async function createNewSession(skipConfirm = false) {
      if (!skipConfirm && (files.length > 0 || turnCount > 0)) {
        if (!confirm('Start a new study session? Current session is saved in your history.')) return;
      }
      localStorage.removeItem('study_rag_files');
      localStorage.removeItem('study_rag_chat_history');
      localStorage.removeItem('study_rag_active_idx');
      files.forEach(f => {
        if (f.url && f.url.startsWith('blob:')) {
          try { URL.revokeObjectURL(f.url); } catch (_) { }
        }
      });
      files = [];
      chatHistory = [];
      activeIdx = null;
      currentPptSlides = [];
      renderFileList();
      document.getElementById('viewer').innerHTML = '<div class="viewer-empty"><p>Upload files and click one to preview it</p></div>';
      document.getElementById('contentTitle').textContent = 'Select a file to preview';
      document.getElementById('fileCountTag').textContent = '0 files';
      document.getElementById('chatMessages').innerHTML = '<div class="bubble bot">Hi! Upload your study files on the left, then ask me anything about them.</div>';
      turnCount = 0;
      document.getElementById('historyBadge').textContent = '0 turns';
      const summaryTitle = document.getElementById('summaryDocTitle');
      if (summaryTitle) summaryTitle.textContent = 'Document Summary';
      const summaryText = document.getElementById('summaryPanelText');
      if (summaryText) summaryText.textContent = 'Select or upload a document and switch to this tab to see an AI-generated summary.';

      try {
        const res = await fetch(`${API_BASE}/session/new`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ device_id: getDeviceId() })
        });
        if (res.ok) {
          const data = await res.json();
          sessionId = data.session_id;
          localStorage.setItem('study_rag_session_id', sessionId);
          saveLocalState();
          setStatus(true);
          showToast('Started fresh session');
        }
      } catch (e) {
        console.error('Failed to create new session:', e);
        showToast('Failed to create new session');
      }

      await loadSidebarSessions();
    }

    function startNewSession() {
      createNewSession(false);
    }

    async function initSession() {
      warmUpServices(); // Eagerly wake up both services on boot

      // Auto-probe localhost; if unreachable, automatically switch to Render cloud backend
      if (API_BASE === LOCAL_BACKEND && window.location.port !== '8000') {
        try {
          const probeCtrl = new AbortController();
          const probeTimer = setTimeout(() => probeCtrl.abort(), 1200);
          const probe = await fetch(`${LOCAL_BACKEND}/health`, { signal: probeCtrl.signal });
          clearTimeout(probeTimer);
          if (!probe.ok) throw new Error();
        } catch {
          console.log('Local backend not active on :8000, switching to Render cloud backend:', RENDER_BACKEND);
          API_BASE = RENDER_BACKEND;
        }
      }

      // Restore sidebar collapse state preference if saved
      try {
        if (localStorage.getItem('study_rag_sidebar_collapsed') === '1') {
          document.getElementById('historySidebar')?.classList.add('collapsed');
        }
      } catch (_) { }

      // Always start a brand new session on fresh visit tagged with this device's ID
      try {
        const res = await fetch(`${API_BASE}/session/new`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ device_id: getDeviceId() })
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        sessionId = data.session_id;
        localStorage.setItem('study_rag_session_id', sessionId);
        saveLocalState();
        setStatus(true);
      } catch (err) {
        console.error('Session init error:', err);
        setStatus(false);
        showToast('Cannot reach backend. Is the server running?');
      }

      // Fetch session history for this device
      await loadSidebarSessions();
    }

    // ── Media Modals (Diagram Zoom & Video Player) ─────────────
    function openImageModal(src, title) {
      const modal = document.getElementById('imageModal');
      const img = document.getElementById('imageModalImg');
      const titleEl = document.getElementById('imageModalTitle');
      if (img) img.src = src;
      if (titleEl) titleEl.textContent = title || 'Diagram Preview';
      if (modal) modal.classList.remove('hidden');
    }

    function closeImageModal() {
      const modal = document.getElementById('imageModal');
      const img = document.getElementById('imageModalImg');
      if (modal) modal.classList.add('hidden');
      if (img) img.src = '';
    }

    function openVideoModal(videoId, title) {
      const modal = document.getElementById('videoModal');
      const iframe = document.getElementById('videoModalIframe');
      const titleEl = document.getElementById('videoModalTitle');
      if (iframe) iframe.src = `https://www.youtube.com/embed/${videoId}?autoplay=1`;
      if (titleEl) titleEl.textContent = title || 'Video Tutorial';
      if (modal) modal.classList.remove('hidden');
    }

    function closeVideoModal() {
      const modal = document.getElementById('videoModal');
      const iframe = document.getElementById('videoModalIframe');
      if (modal) modal.classList.add('hidden');
      if (iframe) iframe.src = '';
    }

    window.addEventListener('keydown', e => {
      if (e.key === 'Escape') {
        closeImageModal();
        closeVideoModal();
      }
    });

    // Auto-initialize session on load
    window.addEventListener('DOMContentLoaded', initSession);

    function setStatus(ok) {
      document.getElementById('statusDot').className = 'status-dot ' + (ok ? 'ok' : 'err');
      document.getElementById('statusLabel').textContent = ok ? 'Connected' : 'Disconnected';
      const sub = document.getElementById('chatSubtitle');
      if (sub) {
        sub.textContent = ok
          ? (files.length ? `${files.length} file(s) loaded` : 'Upload files to begin')
          : 'Connecting…';
      }
    }

    // ── File upload ───────────────────────────────────────────
    const EXT_ICONS = { pdf: 'PDF', docx: 'DOCX', pptx: 'PPTX', txt: 'TXT' };

    document.getElementById('fileInput').addEventListener('change', async function () {
      warmUpServices();  // Wake up services while files are being read
      await uploadFiles(Array.from(this.files));
      this.value = '';
    });

    const zone = document.getElementById('uploadZone');
    zone.addEventListener('mouseenter', () => warmUpServices());
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag'));
    zone.addEventListener('drop', async e => {
      e.preventDefault(); zone.classList.remove('drag');
      warmUpServices();  // Wake up services while files are being read
      await uploadFiles(Array.from(e.dataTransfer.files));
    });

    function updateProgress(percent, text, state) {
      const container = document.getElementById('uploadProgressContainer');
      const bar = document.getElementById('progressBar');
      const status = document.getElementById('progressStatus');
      const pct = document.getElementById('progressPercent');

      container.style.display = 'flex';
      bar.style.width = Math.min(100, Math.max(0, percent)) + '%';
      bar.className = 'progress-bar-fill' + (state ? ' ' + state : '');
      status.textContent = text;
      pct.textContent = Math.round(percent) + '%';
    }

    function hideProgress(delay = 2500) {
      setTimeout(() => {
        const container = document.getElementById('uploadProgressContainer');
        container.style.display = 'none';
      }, delay);
    }

    async function uploadFiles(fileList) {
      if (!fileList.length) return;

      if (!sessionId) {
        updateProgress(10, 'Connecting to server…', '');
        await initSession();
        if (!sessionId) {
          updateProgress(100, 'Cannot connect to server', 'error');
          showToast('Backend connection failed. Is the server running?');
          hideProgress(4000);
          return;
        }
      }

      const form = new FormData();
      fileList.forEach(f => form.append('files', f));

      updateProgress(5, `Uploading ${fileList.length} file(s)…`, '');

      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${API_BASE}/upload/${sessionId}`);

      let serverProcessingInterval = null;

      // 1. Client-to-server upload progress (0% - 65%)
      xhr.upload.onprogress = function (e) {
        if (e.lengthComputable) {
          const pct = Math.round((e.loaded / e.total) * 65);
          updateProgress(pct, `Uploading ${fileList.length} file(s) (${pct}%)…`, '');
        }
      };

      // 2. Upload complete, server is parsing text and indexing vectors (68% - 99%)
      xhr.upload.onload = function () {
        let currentPct = 68;
        updateProgress(currentPct, 'Parsing document text & structure…', 'indexing');

        const steps = [
          { pct: 74, text: 'Extracting content & splitting chunks…' },
          { pct: 80, text: 'Connecting to AI embedding service…' },
          { pct: 86, text: 'Vectorizing document chunks with FastEmbed…' },
          { pct: 90, text: 'Building semantic vector index…' },
          { pct: 93, text: 'Synchronizing multi-doc context…' },
          { pct: 95, text: 'Finalizing vector database index…' },
        ];
        const extraMessages = [
          'Computing embeddings (large files take a bit longer)…',
          'Finalizing semantic vector index…',
          'Almost ready, preparing study assistant…',
          'Verifying knowledge base…',
        ];
        let stepIdx = 0;
        let extraIdx = 0;

        serverProcessingInterval = setInterval(() => {
          if (stepIdx < steps.length) {
            updateProgress(steps[stepIdx].pct, steps[stepIdx].text, 'indexing');
            currentPct = steps[stepIdx].pct;
            stepIdx++;
          } else {
            // Gentle continuous creep up to 99% with active rotating status messages
            if (currentPct < 99) {
              currentPct += 1;
            }
            const msg = extraMessages[extraIdx % extraMessages.length];
            extraIdx++;
            updateProgress(currentPct, msg, 'indexing');
          }
        }, 2800);
      };

      function cleanupProcessingInterval() {
        if (serverProcessingInterval) {
          clearInterval(serverProcessingInterval);
          serverProcessingInterval = null;
        }
      }

      // 3. Response received
      xhr.onload = function () {
        cleanupProcessingInterval();
        let data;
        try {
          data = JSON.parse(xhr.responseText);
        } catch {
          data = {};
        }

        if (xhr.status >= 200 && xhr.status < 300) {
          if (data.error) {
            updateProgress(100, data.error, 'error');
            showToast('Upload error: ' + data.error);
            hideProgress(5000);
            return;
          }

          updateProgress(100, `Indexed ${data.chunks || 0} chunks!`, 'done');
          const startIdx = files.length;
          fileList.forEach(f => files.push({
            name: f.name,
            size: f.size,
            ext: f.name.split('.').pop().toLowerCase(),
            url: URL.createObjectURL(f),
            rawFile: f,
            slides: (data.slides && data.slides[f.name]) ? data.slides[f.name] : null,
            text: (data.docs && data.docs[f.name]) ? data.docs[f.name] : null,
          }));
          renderFileList();
          saveLocalState();
          showToast(`${fileList.length} file(s) indexed (${data.chunks} chunks)`);
          const sub = document.getElementById('chatSubtitle');
          if (sub) sub.textContent = `${files.length} file(s) loaded`;
          document.getElementById('fileCountTag').textContent = files.length + ' file' + (files.length !== 1 ? 's' : '');
          hideProgress(2500);

          // Auto-display the newly uploaded file immediately
          if (files.length > 0) {
            selectFile(startIdx);
          }
          loadSidebarSessions();
        } else {
          const errorDetail = data.detail || data.error || `Server error (${xhr.status})`;
          updateProgress(100, errorDetail, 'error');
          showToast(`Upload failed: ${errorDetail}`);
          hideProgress(5000);
        }
      };

      xhr.onerror = function () {
        cleanupProcessingInterval();
        updateProgress(100, 'Network connection error', 'error');
        showToast('Network error — cannot reach server');
        hideProgress(4000);
      };

      xhr.timeout = 180000;
      xhr.ontimeout = function () {
        cleanupProcessingInterval();
        updateProgress(100, 'Upload timed out. Please try again.', 'error');
        showToast('Upload timed out — server took too long to respond');
        hideProgress(5000);
      };

      xhr.send(form);
    }

    function renderFileList() {
      const list = document.getElementById('fileList');
      if (!files.length) {
        list.innerHTML = '<div class="empty-state">No files yet.<br/>Upload study material above.</div>';
        return;
      }
      list.innerHTML = files.map((f, i) => `
    <div class="file-card${activeIdx === i ? ' active' : ''}" onclick="selectFile(${i})">
      <span class="file-ext-badge ${f.ext}">${f.ext.toUpperCase()}</span>
      <div class="file-info">
        <div class="fname">${f.name}</div>
        <div class="fmeta">${fmtSize(f.size)}</div>
      </div>
      <button class="del-btn" onclick="removeFile(event,${i})" title="Remove">×</button>
    </div>
  `).join('');
    }

    function renderPptSlide(idx) {
      if (!currentPptSlides.length) return;
      currentPptIndex = Math.max(0, Math.min(idx, currentPptSlides.length - 1));
      const s = currentPptSlides[currentPptIndex];

      const titleEl = document.getElementById('pptSlideTitle');
      const badgeEl = document.getElementById('pptSlideBadge');
      const bodyEl = document.getElementById('pptSlideBody');
      const selectEl = document.getElementById('pptSlideSelect');
      const firstBtn = document.getElementById('pptFirstBtn');
      const prevBtn = document.getElementById('pptPrevBtn');
      const nextBtn = document.getElementById('pptNextBtn');
      const lastBtn = document.getElementById('pptLastBtn');

      if (titleEl) titleEl.textContent = s.title || `Slide ${s.slide_number}`;
      if (badgeEl) badgeEl.textContent = `Slide ${s.slide_number} of ${currentPptSlides.length}`;
      if (selectEl) selectEl.value = currentPptIndex;

      if (firstBtn) firstBtn.disabled = currentPptIndex === 0;
      if (prevBtn) prevBtn.disabled = currentPptIndex === 0;
      if (nextBtn) nextBtn.disabled = currentPptIndex === currentPptSlides.length - 1;
      if (lastBtn) lastBtn.disabled = currentPptIndex === currentPptSlides.length - 1;

      // Highlight active thumbnail in outline
      document.querySelectorAll('.ppt-thumb-item').forEach((item, i) => {
        item.classList.toggle('active', i === currentPptIndex);
        if (i === currentPptIndex) {
          item.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        }
      });

      if (bodyEl) {
        let bodyContent = '';

        // Render bullets if present
        if (s.bullets && s.bullets.length > 0) {
          bodyContent += `
        <ul class="ppt-bullet-list">
          ${s.bullets.map(b => {
            const text = typeof b === 'string' ? b : (b.text || '');
            const level = typeof b === 'object' && b.level ? b.level : 0;
            const indent = level * 20;
            const bulletMarker = level === 0 ? '•' : (level === 1 ? '◦' : '▪');
            return `<li class="ppt-bullet-item" style="margin-left: ${indent}px;"><span class="ppt-bullet-marker">${bulletMarker}</span>${text}</li>`;
          }).join('')}
        </ul>
      `;
        }

        // Render tables if present
        if (s.tables && s.tables.length > 0) {
          s.tables.forEach(tableData => {
            if (!tableData || !tableData.length) return;
            const [headers, ...rows] = tableData;
            bodyContent += `
          <table class="ppt-table">
            <thead>
              <tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr>
            </thead>
            <tbody>
              ${rows.map(row => `<tr>${row.map(cell => `<td>${cell}</td>`).join('')}</tr>`).join('')}
            </tbody>
          </table>
        `;
          });
        }

        // Render images / diagrams if present
        if (s.images && s.images.length > 0) {
          bodyContent += `
        <div class="ppt-images-container">
          ${s.images.map((img, imgIdx) => {
            const imgSrc = (img.startsWith('http://') || img.startsWith('https://') || img.startsWith('data:'))
              ? img
              : (API_BASE.replace(/\/+$/, '') + (img.startsWith('/') ? img : '/' + img));
            return `
            <div class="ppt-img-wrapper" onclick="openImageModal('${imgSrc}', 'Slide ${s.slide_number} - Diagram ${imgIdx + 1}')" title="Click to zoom diagram">
              <img src="${imgSrc}" alt="Slide ${s.slide_number} Figure ${imgIdx + 1}" class="ppt-slide-img" />
              <div class="ppt-img-zoom-hint">Zoom</div>
            </div>
            `;
          }).join('')}
        </div>
      `;
        }

        // Fallback to raw text if no bullets, tables, or images
        if (!bodyContent && s.raw_text) {
          bodyContent = `<div style="white-space: pre-wrap; line-height: 1.6;">${s.raw_text}</div>`;
        }

        if (!bodyContent) {
          bodyContent = `<div style="color: var(--muted); font-style: italic; margin: auto;">(Slide with visual graphics or diagrams)</div>`;
        }

        // Append speaker notes if present
        if (s.notes) {
          bodyContent += `
        <div class="ppt-notes-drawer">
          <div class="ppt-notes-title">Speaker Notes</div>
          <div>${s.notes}</div>
        </div>
      `;
        }

        bodyEl.innerHTML = bodyContent;
        applySlideTheme();
      }
    }

    let slideTheme = localStorage.getItem('study_rag_slide_theme') || 'white';

    function applySlideTheme() {
      const slide = document.querySelector('.ppt-slide');
      if (slide) {
        slide.classList.toggle('theme-white', slideTheme === 'white');
      }
    }

    function toggleSlideTheme() {
      slideTheme = slideTheme === 'white' ? 'dark' : 'white';
      localStorage.setItem('study_rag_slide_theme', slideTheme);
      applySlideTheme();
      showToast(`Slide theme: ${slideTheme === 'white' ? 'Classic White' : 'Dark Mode'}`);
    }

    function downloadCurrentFile() {
      if (activeIdx === null || !files[activeIdx]) return;
      const f = files[activeIdx];
      if (f.url) {
        const a = document.createElement('a');
        a.href = f.url;
        a.download = f.name;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      } else {
        showToast('Download link not available for this session');
      }
    }

    function prevPptSlide() {
      if (currentPptIndex > 0) renderPptSlide(currentPptIndex - 1);
    }

    function nextPptSlide() {
      if (currentPptIndex < currentPptSlides.length - 1) renderPptSlide(currentPptIndex + 1);
    }

    function togglePptSidebar() {
      const sb = document.getElementById('pptSidebar');
      if (sb) sb.classList.toggle('collapsed');
    }

    function togglePptFullscreen() {
      const elem = document.getElementById('pptAppRoot');
      if (!elem) return;
      if (!document.fullscreenElement) {
        elem.requestFullscreen().catch(() => { });
      } else {
        document.exitFullscreen().catch(() => { });
      }
    }

    // Arrow key & keyboard navigation for PPT
    window.addEventListener('keydown', e => {
      if (activeIdx !== null && files[activeIdx] && (files[activeIdx].ext === 'pptx' || files[activeIdx].ext === 'ppt')) {
        if (document.activeElement && (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'TEXTAREA')) {
          return;
        }
        if (e.key === 'ArrowLeft' || e.key === 'PageUp') prevPptSlide();
        else if (e.key === 'ArrowRight' || e.key === 'PageDown' || e.key === ' ') {
          e.preventDefault();
          nextPptSlide();
        } else if (e.key === 'Home') renderPptSlide(0);
        else if (e.key === 'End') renderPptSlide(currentPptSlides.length - 1);
        else if (e.key === 'f' || e.key === 'F') togglePptFullscreen();
      }
    });

    let currentDocMode = 'original'; // 'original' | 'google' | 'outline'

    function loadUniversalDocumentViewer(f) {
      const container = document.getElementById('docViewerMount');
      if (!container) return;

      const ext = (f.ext || f.name.split('.').pop()).toLowerCase();

      // If PPT, pre-fetch slides in background for outline mode
      if (ext === 'pptx' || ext === 'ppt') {
        currentPptFilename = f.name;
        if (f.slides && f.slides.length) {
          currentPptSlides = f.slides;
        } else {
          fetch(`${API_BASE}/slides/${sessionId}/${encodeURIComponent(f.name)}`)
            .then(r => r.json())
            .then(d => {
              if (d.slides) {
                currentPptSlides = d.slides;
                f.slides = d.slides;
                saveLocalState();
              }
            })
            .catch(() => { });
        }
      }

      // Tabs based on document type
      let tabsHtml = '';
      if (ext === 'pptx' || ext === 'ppt') {
        tabsHtml = `
      <button class="ppt-view-tab ${currentDocMode === 'original' ? 'active' : ''}" data-mode="original" onclick="switchDocViewMode('original')">
        Original Presentation
      </button>
      <button class="ppt-view-tab ${currentDocMode === 'google' ? 'active' : ''}" data-mode="google" onclick="switchDocViewMode('google')">
        Google Viewer
      </button>
      <button class="ppt-view-tab ${currentDocMode === 'outline' ? 'active' : ''}" data-mode="outline" onclick="switchDocViewMode('outline')">
        Slide Outline & Content
      </button>
    `;
      } else if (ext === 'docx' || ext === 'doc') {
        tabsHtml = `
      <button class="ppt-view-tab ${currentDocMode === 'original' ? 'active' : ''}" data-mode="original" onclick="switchDocViewMode('original')">
        Original Word Document
      </button>
      <button class="ppt-view-tab ${currentDocMode === 'google' ? 'active' : ''}" data-mode="google" onclick="switchDocViewMode('google')">
        Google Docs Viewer
      </button>
      <button class="ppt-view-tab ${currentDocMode === 'outline' ? 'active' : ''}" data-mode="outline" onclick="switchDocViewMode('outline')">
        Extracted Text
      </button>
    `;
      } else if (ext === 'pdf') {
        tabsHtml = `
      <button class="ppt-view-tab ${currentDocMode === 'original' ? 'active' : ''}" data-mode="original" onclick="switchDocViewMode('original')">
        Original PDF
      </button>
      <button class="ppt-view-tab ${currentDocMode === 'google' ? 'active' : ''}" data-mode="google" onclick="switchDocViewMode('google')">
        Google Viewer
      </button>
      <button class="ppt-view-tab ${currentDocMode === 'outline' ? 'active' : ''}" data-mode="outline" onclick="switchDocViewMode('outline')">
        Extracted Text
      </button>
    `;
      } else if (ext === 'xlsx' || ext === 'xls') {
        tabsHtml = `
      <button class="ppt-view-tab ${currentDocMode === 'original' ? 'active' : ''}" data-mode="original" onclick="switchDocViewMode('original')">
        Original Spreadsheet
      </button>
      <button class="ppt-view-tab ${currentDocMode === 'google' ? 'active' : ''}" data-mode="google" onclick="switchDocViewMode('google')">
        Google Sheets Viewer
      </button>
    `;
      } else {
        tabsHtml = `
      <button class="ppt-view-tab active" data-mode="original" onclick="switchDocViewMode('original')">
        Original File (${ext.toUpperCase()})
      </button>
    `;
      }

      container.innerHTML = `
    <div class="ppt-original-viewer" id="docViewerRoot">
      <div class="ppt-view-nav">
        <div class="ppt-view-tabs">
          ${tabsHtml}
        </div>
        <div class="ppt-view-actions">
          <button class="ppt-ctrl-btn" onclick="downloadCurrentFile()" title="Download original file">
            Download Original (${ext.toUpperCase()})
          </button>
          <button class="ppt-ctrl-btn" onclick="toggleDocFullscreen()" title="Fullscreen">
            Fullscreen
          </button>
        </div>
      </div>
      <div class="ppt-view-body" id="docViewBody"></div>
    </div>
  `;

      renderDocCurrentMode();
    }

    function switchDocViewMode(mode) {
      currentDocMode = mode;
      document.querySelectorAll('.ppt-view-tab').forEach(tab => {
        tab.classList.toggle('active', tab.getAttribute('data-mode') === mode);
      });
      renderDocCurrentMode();
    }

    function renderDocCurrentMode() {
      const body = document.getElementById('docViewBody');
      if (!body) return;

      const f = files[activeIdx];
      if (!f) return;

      const ext = (f.ext || f.name.split('.').pop()).toLowerCase();
      const rawUrl = `${API_BASE}/raw/${sessionId}/${encodeURIComponent(f.name)}`;
      const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';

      // 1. ORIGINAL MODE
      if (currentDocMode === 'original') {
        if (ext === 'pdf') {
          const pdfUrl = f.url || rawUrl;
          body.innerHTML = `<iframe class="ppt-view-iframe" src="${pdfUrl}" allowfullscreen="true" title="${f.name}"></iframe>`;
        } else if (ext === 'pptx' || ext === 'ppt' || ext === 'docx' || ext === 'doc' || ext === 'xlsx' || ext === 'xls') {
          const officeUrl = `https://view.officeapps.live.com/op/embed.aspx?src=${encodeURIComponent(rawUrl)}`;
          body.innerHTML = `
        ${isLocal ? `
          <div class="ppt-local-notice">
            <div>
              <strong>Notice:</strong> Microsoft Office Online Viewer requires a public URL. When deployed on Render (<a href="https://document-ai-chatbot-71ey.onrender.com" target="_blank" style="color: inherit; text-decoration: underline;">document-ai-chatbot-71ey.onrender.com</a>), your original ${ext.toUpperCase()} file renders with 100% fidelity here. In local mode, you can <a href="javascript:downloadCurrentFile()" style="color: inherit; font-weight: bold; text-decoration: underline;">download the original file</a> or view the <strong>Outline</strong> tab.
            </div>
          </div>
        ` : `
          <div class="ppt-local-notice" style="background: rgba(59, 110, 246, 0.08); border-color: rgba(59, 110, 246, 0.2); font-size: 11px; padding: 6px 12px; display: flex; align-items: center; justify-content: space-between; gap: 8px;">
            <span>Viewing via Microsoft Online. If it shows <em>"File not found"</em> (e.g. during server cold start), switch to:</span>
            <div style="display: flex; gap: 6px; flex-shrink: 0;">
              <button class="btn btn-sm" onclick="switchDocViewMode('outline')" style="padding: 2px 8px; font-size: 11px; cursor: pointer;">Slides / Text</button>
              <button class="btn btn-sm" onclick="switchDocViewMode('google')" style="padding: 2px 8px; font-size: 11px; cursor: pointer;">Google Viewer</button>
            </div>
          </div>
        `}
        <iframe class="ppt-view-iframe" src="${officeUrl}" allowfullscreen="true" title="Original ${ext.toUpperCase()} File"></iframe>
      `;
        } else {
          renderOriginalTextView(body, f);
        }
      }
      // 2. GOOGLE VIEWER MODE
      else if (currentDocMode === 'google') {
        const gviewUrl = `https://docs.google.com/gview?url=${encodeURIComponent(rawUrl)}&embedded=true`;
        body.innerHTML = `
      ${isLocal ? `
        <div class="ppt-local-notice">
          <div>
            <strong>Notice:</strong> Google Viewer requires a public domain. When deployed, it embeds directly.
          </div>
        </div>
      ` : ''}
      <iframe class="ppt-view-iframe" src="${gviewUrl}" allowfullscreen="true" title="Google Viewer"></iframe>
    `;
      }
      // 3. OUTLINE / EXTRACTED TEXT MODE
      else if (currentDocMode === 'outline') {
        if (ext === 'pptx' || ext === 'ppt') {
          renderPptOutlineMode(body);
        } else {
          renderDocxOutlineMode(body, f);
        }
      }
    }

    function renderPptOutlineMode(body) {
      const thumbsHtml = (currentPptSlides || []).map((s, i) => `
    <div class="ppt-thumb-item${i === currentPptIndex ? ' active' : ''}" onclick="renderPptSlide(${i})">
      <span class="ppt-thumb-num">Slide ${s.slide_number}</span>
      <span class="ppt-thumb-title">${s.title || `Slide ${s.slide_number}`}</span>
    </div>
  `).join('');

      const selectOptionsHtml = (currentPptSlides || []).map((s, i) => `
    <option value="${i}">Slide ${s.slide_number} of ${currentPptSlides.length}</option>
  `).join('');

      body.innerHTML = `
    <div class="ppt-app" id="pptOutlineRoot">
      <div class="ppt-sidebar" id="pptSidebar">
        <div class="ppt-sidebar-head">
          <span>Slides (${currentPptSlides.length})</span>
          <button class="clear-btn" onclick="togglePptSidebar()" title="Collapse outline">✕</button>
        </div>
        <div class="ppt-thumbs-list">
          ${thumbsHtml}
        </div>
      </div>
      <div class="ppt-main">
        <div class="ppt-canvas-area">
          <div class="ppt-slide">
            <div class="ppt-slide-header">
              <div class="ppt-slide-title" id="pptSlideTitle"></div>
              <div class="ppt-slide-badge" id="pptSlideBadge"></div>
            </div>
            <div class="ppt-slide-body" id="pptSlideBody"></div>
          </div>
        </div>
        <div class="ppt-controls-bar">
          <div class="ppt-ctrl-group">
            <button class="ppt-ctrl-btn" onclick="togglePptSidebar()" title="Toggle Slides Outline">Slides</button>
            <button class="ppt-ctrl-btn" id="pptFirstBtn" onclick="renderPptSlide(0)" title="First slide">|◀</button>
            <button class="ppt-ctrl-btn" id="pptPrevBtn" onclick="prevPptSlide()" title="Previous slide (Left Arrow)">◀</button>
          </div>
          <div class="ppt-ctrl-group">
            <select class="ppt-select" id="pptSlideSelect" onchange="renderPptSlide(parseInt(this.value, 10))">
              ${selectOptionsHtml}
            </select>
          </div>
          <div class="ppt-ctrl-group">
            <button class="ppt-ctrl-btn" id="pptNextBtn" onclick="nextPptSlide()" title="Next slide (Right Arrow / Space)">▶</button>
            <button class="ppt-ctrl-btn" id="pptLastBtn" onclick="renderPptSlide(currentPptSlides.length - 1)" title="Last slide">▶|</button>
            <button class="ppt-ctrl-btn" onclick="toggleSlideTheme()" title="Toggle Classic White / Dark Canvas">Theme</button>
          </div>
        </div>
      </div>
    </div>
  `;
      renderPptSlide(currentPptIndex);
    }

    function renderDocxOutlineMode(body, f) {
      let text = f.text || '';
      if (!text) {
        body.innerHTML = `<div class="viewer-empty"><p>Loading extracted text…</p></div>`;
        fetch(`${API_BASE}/document/${sessionId}/${encodeURIComponent(f.name)}`)
          .then(r => r.json())
          .then(d => {
            f.text = d.text || '';
            saveLocalState();
            renderDocxOutlineMode(body, f);
          })
          .catch(err => {
            body.innerHTML = `<div class="viewer-empty"><p>Could not extract text (${err.message})</p></div>`;
          });
        return;
      }

      body.innerHTML = `
    <div class="docx-container">
      <div class="docx-paper">${text || '(Empty document)'}</div>
    </div>
  `;
    }

    function renderOriginalTextView(body, f) {
      body.innerHTML = `<div id="txtViewer" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; white-space: pre-wrap; font-family: monospace; background: var(--surface2); padding: 24px; overflow-y: auto; color: var(--text); line-height: 1.6;">Loading text…</div>`;
      if (f.rawFile) {
        const reader = new FileReader();
        reader.onload = e => {
          const el = document.getElementById('txtViewer');
          if (el) el.textContent = e.target.result;
        };
        reader.readAsText(f.rawFile);
      } else if (f.text) {
        setTimeout(() => {
          const el = document.getElementById('txtViewer');
          if (el) el.textContent = f.text;
        }, 30);
      } else {
        fetch(`${API_BASE}/raw/${sessionId}/${encodeURIComponent(f.name)}`)
          .then(r => r.text())
          .then(t => {
            f.text = t;
            saveLocalState();
            const el = document.getElementById('txtViewer');
            if (el) el.textContent = t;
          })
          .catch(() => {
            const el = document.getElementById('txtViewer');
            if (el) el.textContent = 'Could not load text file content.';
          });
      }
    }

    function toggleDocFullscreen() {
      const elem = document.getElementById('docViewerRoot');
      if (!elem) return;
      if (!document.fullscreenElement) {
        elem.requestFullscreen().catch(() => { });
      } else {
        document.exitFullscreen().catch(() => { });
      }
    }

    function selectFile(i) {
      activeIdx = i;
      saveLocalState();
      renderFileList();
      const f = files[i];
      if (!f) return;
      document.getElementById('contentTitle').textContent = f.name;

      document.getElementById('viewer').innerHTML = `
    <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; display: flex; flex-direction: column;">
      <div id="viewerContent" style="flex: 1; position: relative; overflow: hidden;">
        <div id="docViewerMount" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></div>
      </div>
    </div>
  `;

      loadUniversalDocumentViewer(f);
    }

    function removeFile(e, i) {
      e.stopPropagation();
      if (files[i] && files[i].url) URL.revokeObjectURL(files[i].url);
      files.splice(i, 1);
      if (activeIdx === i) {
        activeIdx = null;
        document.getElementById('viewer').innerHTML = '<div class="viewer-empty"><p>Select a file to preview it</p></div>';
        document.getElementById('contentTitle').textContent = 'Select a file to preview';
      } else if (activeIdx > i) activeIdx--;
      document.getElementById('fileCountTag').textContent = files.length + ' file' + (files.length !== 1 ? 's' : '');
      renderFileList();
      saveLocalState();
      showToast('File removed from view (still in vector store for this session)');
    }

    // ── Chat Formatting & Helpers ─────────────────────────────
    function escapeHtml(str) {
      if (!str) return '';
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function parseInlineStylesToHtml(text) {
      if (!text) return '';
      // Split by inline code: `code`
      const codeParts = text.split(/(`[^`\n]+`)/g);
      return codeParts.map(part => {
        if (part.startsWith('`') && part.endsWith('`')) {
          const codeContent = part.slice(1, -1);
          return `<code class="inline-code">${escapeHtml(codeContent)}</code>`;
        }
        // Parse bold **text**
        const boldParts = part.split(/(\*\*[^*\n]+\*\*)/g);
        return boldParts.map(bPart => {
          if (bPart.startsWith('**') && bPart.endsWith('**')) {
            const boldContent = bPart.slice(2, -2);
            return `<strong class="bold-text">${escapeHtml(boldContent)}</strong>`;
          }
          return escapeHtml(bPart);
        }).join('');
      }).join('');
    }

    function formatMarkdownToHtml(rawText) {
      if (!rawText) return '';

      // Split by fenced code blocks: ```lang ... ```
      const parts = rawText.split(/(```[\s\S]*?```)/g);
      let html = '<div class="message-content-wrapper">';

      parts.forEach(part => {
        if (part.startsWith('```') && part.endsWith('```')) {
          const inner = part.slice(3, -3);
          const firstNewLine = inner.indexOf('\n');
          let lang = 'text';
          let code = inner;
          if (firstNewLine !== -1) {
            lang = inner.substring(0, firstNewLine).trim() || 'text';
            code = inner.substring(firstNewLine + 1);
          }
          if (code.endsWith('\n')) code = code.slice(0, -1);

          const safeLang = escapeHtml(lang);
          const safeCode = escapeHtml(code);

          html += `
        <div class="code-block-container">
          <div class="code-block-header">
            <span class="code-block-lang">${safeLang}</span>
            <button class="code-block-copy-btn" onclick="copyCodeFromBlock(this)">Copy code</button>
          </div>
          <pre class="code-block-pre"><code class="code-block-code">${safeCode}</code></pre>
        </div>
      `;
        } else {
          // Process text lines (lists, paragraphs, headers)
          const lines = part.split('\n');
          let inUl = false;
          let inOl = false;

          const closeList = () => {
            if (inUl) { html += '</ul>'; inUl = false; }
            if (inOl) { html += '</ol>'; inOl = false; }
          };

          lines.forEach(line => {
            const bulletMatch = line.match(/^\s*[-*+]\s+(.*)/);
            const numberMatch = line.match(/^\s*\d+\.\s+(.*)/);

            if (bulletMatch) {
              if (inOl) closeList();
              if (!inUl) { html += '<ul class="message-list-ul">'; inUl = true; }
              html += `<li class="message-list-item">${parseInlineStylesToHtml(bulletMatch[1])}</li>`;
            } else if (numberMatch) {
              if (inUl) closeList();
              if (!inOl) { html += '<ol class="message-list-ol">'; inOl = true; }
              html += `<li class="message-list-item">${parseInlineStylesToHtml(numberMatch[1])}</li>`;
            } else {
              closeList();
              const trimmed = line.trim();
              if (!trimmed) {
                // empty line
              } else if (trimmed.startsWith('### ')) {
                html += `<h4 style="margin: 6px 0 2px 0; font-size: 13px; font-weight: 600; color: #f1f5f9;">${parseInlineStylesToHtml(trimmed.slice(4))}</h4>`;
              } else if (trimmed.startsWith('## ')) {
                html += `<h3 style="margin: 8px 0 4px 0; font-size: 14px; font-weight: 600; color: #f1f5f9;">${parseInlineStylesToHtml(trimmed.slice(3))}</h3>`;
              } else if (trimmed.startsWith('# ')) {
                html += `<h2 style="margin: 10px 0 4px 0; font-size: 15px; font-weight: 700; color: #f1f5f9;">${parseInlineStylesToHtml(trimmed.slice(2))}</h2>`;
              } else {
                html += `<p class="message-paragraph">${parseInlineStylesToHtml(line)}</p>`;
              }
            }
          });
          closeList();
        }
      });

      html += '</div>';
      return html;
    }

    function copyCodeFromBlock(btn) {
      const container = btn.closest('.code-block-container');
      if (!container) return;
      const codeEl = container.querySelector('.code-block-code');
      if (!codeEl) return;
      navigator.clipboard.writeText(codeEl.textContent).then(() => {
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = 'Copy code'; }, 2000);
      }).catch(() => {
        btn.textContent = 'Failed';
        setTimeout(() => { btn.textContent = 'Copy code'; }, 2000);
      });
    }

    function renderFormattedMessage(element, rawText) {
      element.innerHTML = formatMarkdownToHtml(rawText);
    }

    function buildYouTubeSourcesElement(sources) {
      if (!sources || !sources.length) return null;
      const ytBox = document.createElement('div');
      ytBox.className = 'yt-sources-box';
      ytBox.innerHTML = `
    <div class="yt-sources-title">Recommended Video Tutorials:</div>
    <div class="yt-cards-container">
      ${sources.map(v => {
        const safeTitleAttr = escapeHtml(v.title || '');
        const safeTitleJs = (v.title || '').replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '&quot;');
        const safeChannel = escapeHtml(v.channel || '');
        const safeDuration = escapeHtml(v.duration || '');
        const safeThumb = escapeHtml(v.thumbnail || '');
        const safeUrl = escapeHtml(v.url || `https://www.youtube.com/watch?v=${v.id}`);
        const safeId = escapeHtml(v.id || '');
        return `
          <div class="yt-card">
            <div class="yt-thumb-wrap" onclick="openVideoModal('${safeId}', '${safeTitleJs}')" title="Play tutorial">
              <img src="${safeThumb}" class="yt-thumb" alt="${safeTitleAttr}" />
              ${safeDuration ? `<span class="yt-duration">${safeDuration}</span>` : ''}
              <div class="yt-play-overlay">▶</div>
            </div>
            <div class="yt-card-body">
              <div class="yt-card-title" title="${safeTitleAttr}">${safeTitleAttr}</div>
              <div class="yt-card-channel">${safeChannel}</div>
              <div class="yt-card-actions">
                <button class="yt-watch-btn" onclick="openVideoModal('${safeId}', '${safeTitleJs}')">▶ Watch Here</button>
                <a href="${safeUrl}" target="_blank" rel="noopener noreferrer" class="yt-link-btn" title="Open in YouTube">↗ YouTube</a>
              </div>
            </div>
          </div>
        `;
      }).join('')}
    </div>
  `;
      return ytBox;
    }

    // ── Chat ──────────────────────────────────────────────────
    function toggleChat() {
      chatOpen = !chatOpen;
      document.getElementById('chatWindow').classList.toggle('hidden', !chatOpen);
      const fab = document.getElementById('chatFab');
      fab.classList.toggle('open', chatOpen);
      fab.innerHTML = chatOpen
        ? '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>'
        : '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>';
      if (chatOpen) document.getElementById('chatInput').focus();
    }

    async function sendMessage() {
      if (isSending) return;
      const input = document.getElementById('chatInput');
      const q = input.value.trim();
      if (!q) return;

      if (!sessionId) {
        await initSession();
        if (!sessionId) {
          addBubble('Connecting to server… please try again in a moment.', 'error');
          return;
        }
      }
      if (!files.length) {
        addBubble('Upload at least one study file first, then ask your question.', 'error');
        return;
      }

      input.value = '';
      addBubble(q, 'user');

      // Save user turn to local chat history
      chatHistory.push({ role: 'user', content: q, timestamp: Date.now() });
      saveLocalState();

      const typingBubble = addBubble('Thinking…', 'bot typing');
      isSending = true;
      document.getElementById('sendBtn').disabled = true;

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 75000);

      try {
        const res = await fetch(`${API_BASE}/ask`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId, question: q }),
          signal: controller.signal,
        });
        clearTimeout(timeoutId);
        const data = await res.json();

        if (!res.ok) {
          typingBubble.textContent = data.detail || 'Something went wrong.';
          typingBubble.className = 'bubble error';
        } else {
          typingBubble.classList.remove('typing');
          renderFormattedMessage(typingBubble, data.answer);

          // YouTube Video Tutorials recommendation
          const ytEl = buildYouTubeSourcesElement(data.youtube_sources);
          if (ytEl) typingBubble.appendChild(ytEl);

          // Sources tag
          const src = document.createElement('div');
          src.className = 'sources-tag';
          src.textContent = `↳ ${data.sources_used} chunk(s) retrieved`;
          typingBubble.after(src);
          turnCount++;
          document.getElementById('historyBadge').textContent = turnCount + ' turn' + (turnCount !== 1 ? 's' : '');

          // Save bot turn to local chat history
          chatHistory.push({
            role: 'bot',
            content: data.answer,
            sources_used: data.sources_used,
            youtube_sources: data.youtube_sources || [],
            timestamp: Date.now(),
          });
          saveLocalState();
          loadSidebarSessions();
        }
      } catch (err) {
        clearTimeout(timeoutId);
        if (err && err.name === 'AbortError') {
          typingBubble.textContent = 'The request timed out (server took longer than 75s). The server may be waking up from sleep or busy. Please try asking again in a few moments.';
        } else {
          const isLocal = API_BASE.includes('localhost') || API_BASE.includes('127.0.0.1');
          if (isLocal) {
            typingBubble.textContent = 'Connection error: Local backend on :8000 is not running. Run `uvicorn main:app --reload --port 8000` or use cloud: ' + RENDER_BACKEND;
          } else {
            typingBubble.textContent = 'Connection error: Cloud backend is waking up or temporarily busy. Please wait 15 seconds and retry. (' + (err ? err.message : '') + ')';
          }
        }
        typingBubble.className = 'bubble error';
      } finally {
        isSending = false;
        document.getElementById('sendBtn').disabled = false;
        scrollChat();
      }
    }

    async function clearHistory() {
      if (!sessionId) return;
      try {
        await fetch(`${API_BASE}/session/${sessionId}/history`, { method: 'DELETE' });
      } catch { }
      chatHistory = [];
      saveLocalState();
      document.getElementById('chatMessages').innerHTML = '<div class="bubble bot">Conversation cleared. Ask a new question!</div>';
      turnCount = 0;
      document.getElementById('historyBadge').textContent = '0 turns';
      showToast('Conversation history cleared');
    }

    function getActiveDocName() {
      if (activeIdx !== null && files[activeIdx]) {
        return files[activeIdx].name;
      }
      return files.length > 0 ? files[0].name : 'your documents';
    }

    let activeChatPanelTab = 'chat'; // 'chat' | 'summary'

    function switchChatPanelTab(tab) {
      activeChatPanelTab = tab;
      const chatView = document.getElementById('chatViewContainer');
      const summaryView = document.getElementById('summaryViewContainer');
      const tabChat = document.getElementById('tabChatPill');
      const tabSummary = document.getElementById('tabSummaryPill');
      const chatActions = document.getElementById('chatSpecificActions');

      if (tab === 'chat') {
        if (chatView) chatView.style.display = 'flex';
        if (summaryView) summaryView.style.display = 'none';
        if (tabChat) tabChat.classList.add('active');
        if (tabSummary) tabSummary.classList.remove('active');
        if (chatActions) chatActions.style.display = 'flex';
        const input = document.getElementById('chatInput');
        if (input) input.focus();
      } else {
        if (chatView) chatView.style.display = 'none';
        if (summaryView) summaryView.style.display = 'flex';
        if (tabChat) tabChat.classList.remove('active');
        if (tabSummary) tabSummary.classList.add('active');
        if (chatActions) chatActions.style.display = 'none';

        // If summary not loaded for current file, generate it
        const docName = getActiveDocName();
        const titleEl = document.getElementById('summaryDocTitle');
        if (titleEl) titleEl.textContent = `Summary: ${docName}`;
        const textEl = document.getElementById('summaryPanelText');
        if (textEl && (textEl.textContent.includes('Select or upload') || textEl.textContent.trim() === '')) {
          summarizeActiveFile();
        }
      }
    }

    function renderChatHistory() {
      const msgs = document.getElementById('chatMessages');
      if (!msgs) return;
      msgs.innerHTML = '';
      turnCount = 0;

      const docName = getActiveDocName();

      if (!chatHistory || chatHistory.length === 0) {
        msgs.innerHTML = `
      <div class="chat-welcome-state">
        <div class="chat-welcome-icon-box">
          <span class="chat-welcome-icon"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg></span>
        </div>
        <div class="chat-welcome-title">Chat Started</div>
        <div class="chat-welcome-desc">
          Ask anything about the contents of <strong>${docName}</strong>. The assistant will search the document structure and provide grounded answers.
        </div>
      </div>
    `;
        const input = document.getElementById('chatInput');
        if (input) input.placeholder = `Ask anything about ${docName}…`;
        document.getElementById('historyBadge').textContent = '0 turns';
        return;
      }

      chatHistory.forEach(item => {
        const d = document.createElement('div');
        d.className = 'bubble ' + item.role;
        if (item.role === 'bot') {
          renderFormattedMessage(d, item.content);
        } else {
          d.textContent = item.content;
        }
        msgs.appendChild(d);

        if (item.role === 'user') {
          turnCount++;
        } else if (item.role === 'bot') {
          const ytEl = buildYouTubeSourcesElement(item.youtube_sources);
          if (ytEl) d.appendChild(ytEl);

          if (item.sources_used) {
            const src = document.createElement('div');
            src.className = 'sources-tag';
            src.textContent = `↳ ${item.sources_used} chunk(s) retrieved`;
            msgs.appendChild(src);
          }
        }
      });

      document.getElementById('historyBadge').textContent = turnCount + ' turn' + (turnCount !== 1 ? 's' : '');
      scrollChat();
    }

    function addBubble(text, type) {
      const msgs = document.getElementById('chatMessages');
      const welcome = msgs.querySelector('.chat-welcome-state');
      if (welcome) welcome.remove();

      const d = document.createElement('div');
      d.className = 'bubble ' + type;
      d.textContent = text;
      msgs.appendChild(d);
      scrollChat();
      return d;
    }

    function scrollChat() {
      const msgs = document.getElementById('chatMessages');
      if (msgs) msgs.scrollTop = msgs.scrollHeight;
    }

    // ── Helpers ───────────────────────────────────────────────
    function fmtSize(b) {
      return b < 1024 ? b + ' B' : b < 1048576 ? (b / 1024).toFixed(1) + ' KB' : (b / 1048576).toFixed(1) + ' MB';
    }

    function loaderName(ext) {
      return { pdf: 'PyPDFLoader', docx: 'UnstructuredWord', pptx: 'UnstructuredPPTX', txt: 'TextLoader' }[ext] || 'TextLoader';
    }

    function showToast(msg) {
      const t = document.getElementById('toast');
      t.textContent = msg;
      t.classList.add('show');
      setTimeout(() => t.classList.remove('show'), 3000);
    }

    function summarizeActiveFile() {
      const docName = getActiveDocName();
      if (docName && docName !== 'your documents') {
        summarizeFile(docName);
      }
    }

    let summarizationInFlight = false;

    async function summarizeFile(filename) {
      if (summarizationInFlight) return;
      summarizationInFlight = true;

      try {
        if (!sessionId) {
          await initSession();
          if (!sessionId) {
            showToast('Connecting to server… please try again.');
            return;
          }
        }

        // Find file object from files array with fuzzy matching
        const targetNorm = filename ? filename.toLowerCase().replace(/[\s\-_]+/g, '') : '';
        const f = files.find(file =>
          file.name === filename ||
          (file.name && file.name.toLowerCase().replace(/[\s\-_]+/g, '') === targetNorm)
        ) || (activeIdx !== null ? files[activeIdx] : null) || (files.length > 0 ? files[0] : null);

        const targetFilename = f ? f.name : filename;
        const clientText = f ? f.text : null;

        const titleEl = document.getElementById('summaryDocTitle');
        if (titleEl) titleEl.textContent = `Summary: ${targetFilename}`;

        // Render loading spinner into summaryPanelText BEFORE switching tabs
        // so switchChatPanelTab doesn't see an empty panel and trigger a duplicate fetch
        const textEl = document.getElementById('summaryPanelText');
        if (textEl) {
          textEl.innerHTML = `
        <div class="summary-loading-card">
          <div class="summary-spinner-ring"></div>
          <div class="summary-loading-title">Generating Executive Summary</div>
          <div class="summary-loading-sub">Analyzing document text & synthesizing key insights with Gemini AI…</div>
          <div class="summary-shimmer-container">
            <div class="summary-shimmer-line" style="width: 90%;"></div>
            <div class="summary-shimmer-line" style="width: 75%;"></div>
            <div class="summary-shimmer-line" style="width: 85%;"></div>
            <div class="summary-shimmer-line" style="width: 65%;"></div>
          </div>
        </div>
      `;
        }

        const btn = document.getElementById('btnRefreshSummary');
        if (btn) {
          btn.textContent = 'Summarizing…';
          btn.disabled = true;
        }

        // Open the panel and switch to Summary tab (after loading card is in place)
        if (!chatOpen) toggleChat();
        switchChatPanelTab('summary');

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 90000);

        try {
          const res = await fetch(`${API_BASE}/summarize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              session_id: sessionId,
              filename: targetFilename,
              text: clientText
            }),
            signal: controller.signal,
          });
          clearTimeout(timeoutId);

          const data = await res.json();
          if (!res.ok) throw new Error(data.detail || 'Failed to summarize');

          if (textEl) renderFormattedMessage(textEl, data.summary);
          showToast('Summary generated successfully!');
        } catch (e) {
          clearTimeout(timeoutId);
          if (textEl) {
            textEl.innerHTML = `
          <div class="summary-loading-card" style="background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(99, 102, 241, 0.25);">
            <div style="display:flex; justify-content:center; align-items:center; margin-bottom:8px;"><svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#6366f1" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg></div>
            <div class="summary-loading-title">Document Summary Ready to Generate</div>
            <div class="summary-loading-sub">The cloud server was busy or warming up. Click below to load your summary.</div>
            <div style="margin-top: 6px;">
              <button class="summary-btn-action" style="background: #6366f1; color: #fff; border-color: #6366f1; font-weight: 600; padding: 7px 18px; border-radius: 8px;" onclick="summarizeFile('${targetFilename}')">
                Load Summary Now
              </button>
            </div>
          </div>
        `;
          }
          showToast('Document summary ready — click Load Summary');
        } finally {
          if (btn) {
            btn.textContent = 'Regenerate';
            btn.disabled = false;
          }
        }
      } finally {
        summarizationInFlight = false;
      }
    }
  
