/* ========================================================================
   Video Annotation Platform — Frontend Logic
   ======================================================================== */

'use strict';

// ─── State ────────────────────────────────────────────────────────────────
let currentJobId = null;
let pollTimer    = null;
let currentFile  = null;

// ─── DOM refs (populated on DOMContentLoaded) ────────────────────────────
let dropZone, fileInput, fileSelectedBar, fileName, clearBtn;
let submitBtn, sensitivitySlider, sensitivityVal, annotatorSelect, maxFramesInput;
let outputSection, statusDot, statusText, progressFill;
let resultsSection, errorPanel;
let taskText;
let metaGrid, gallery, stepsBody, jsonPre, jsonToggle, jsonBody;
let lightbox, lightboxImg;

// ─── Init ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  dropZone       = document.getElementById('drop-zone');
  fileInput      = document.getElementById('file-input');
  fileSelectedBar= document.getElementById('file-selected');
  fileName       = document.getElementById('file-name');
  clearBtn       = document.getElementById('clear-btn');

  submitBtn      = document.getElementById('submit-btn');
  sensitivitySlider = document.getElementById('sensitivity');
  sensitivityVal = document.getElementById('sensitivity-val');
  annotatorSelect= document.getElementById('annotator');
  maxFramesInput = document.getElementById('max-frames');

  outputSection  = document.getElementById('output-section');
  statusDot      = document.getElementById('status-dot');
  statusText     = document.getElementById('status-text');
  progressFill   = document.getElementById('progress-fill');

  resultsSection = document.getElementById('results-section');
  errorPanel     = document.getElementById('error-panel');

  taskText       = document.getElementById('task-text');
  metaGrid       = document.getElementById('meta-grid');
  gallery        = document.getElementById('gallery');
  stepsBody      = document.getElementById('steps-body');
  jsonPre        = document.getElementById('json-pre');
  jsonToggle     = document.getElementById('json-toggle');
  jsonBody       = document.getElementById('json-body');

  lightbox       = document.getElementById('lightbox');
  lightboxImg    = document.getElementById('lightbox-img');

  bindEvents();
});

// ─── Event binding ────────────────────────────────────────────────────────
function bindEvents() {
  // Drag-and-drop
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer?.files?.[0];
    if (file) setFile(file);
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files?.[0]) setFile(fileInput.files[0]);
  });

  clearBtn.addEventListener('click', clearFile);

  submitBtn.addEventListener('click', startAnnotation);

  sensitivitySlider.addEventListener('input', () => {
    sensitivityVal.textContent = sensitivitySlider.value;
  });

  jsonToggle.addEventListener('click', () => {
    const open = jsonBody.classList.toggle('open');
    jsonToggle.classList.toggle('open', open);
  });

  // Lightbox close
  lightbox.addEventListener('click', e => {
    if (e.target === lightbox) closeLightbox();
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeLightbox();
  });
}

// ─── File handling ────────────────────────────────────────────────────────
function setFile(file) {
  const allowed = ['.mp4', '.avi', '.mov', '.mkv', '.webm'];
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (!allowed.includes(ext)) {
    showToast(`Unsupported format "${ext}". Use: ${allowed.join(', ')}`, 'error');
    return;
  }
  currentFile = file;
  fileName.textContent = file.name + ' (' + formatBytes(file.size) + ')';
  fileSelectedBar.classList.add('visible');
  submitBtn.disabled = false;
}

function clearFile() {
  currentFile = null;
  fileInput.value = '';
  fileSelectedBar.classList.remove('visible');
  submitBtn.disabled = true;
}

// ─── Annotation flow ──────────────────────────────────────────────────────
async function startAnnotation() {
  if (!currentFile) return;

  // Reset UI
  stopPolling();
  resultsSection.classList.remove('visible');
  errorPanel.classList.remove('visible');
  outputSection.classList.add('visible');
  setStatus('pending', `Uploading ${currentFile.name}…`);
  submitBtn.disabled = true;

  const formData = new FormData();
  formData.append('video', currentFile);
  formData.append('sensitivity', sensitivitySlider.value);
  formData.append('annotator', annotatorSelect.value);
  formData.append('max_frames', maxFramesInput.value || '0');

  try {
    const res = await fetch('/annotate', { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Upload failed');
    }
    const { job_id } = await res.json();
    currentJobId = job_id;
    setStatus('running', 'Annotation in progress…');
    startPolling();
  } catch (err) {
    setStatus('error', 'Upload failed');
    showError(err.message);
    submitBtn.disabled = false;
  }
}

// ─── Polling ──────────────────────────────────────────────────────────────
function startPolling() {
  pollTimer = setInterval(pollStatus, 1500);
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

async function pollStatus() {
  if (!currentJobId) return;
  try {
    const res = await fetch(`/status/${currentJobId}`);
    if (!res.ok) return;
    const data = await res.json();

    const { status, progress, total, error } = data;

    if (total > 0) {
      const pct = Math.round((progress / total) * 100);
      progressFill.style.width = pct + '%';
      setStatus('running', `Annotating frames… ${progress} / ${total}`, pct);
    }

    if (status === 'done') {
      stopPolling();
      progressFill.style.width = '100%';
      setStatus('done', 'Annotation complete!');
      await loadResult();
      submitBtn.disabled = false;
    } else if (status === 'error') {
      stopPolling();
      setStatus('error', 'Annotation failed');
      showError(error || 'Unknown error');
      submitBtn.disabled = false;
    }
  } catch (_) { /* network hiccup, keep polling */ }
}

// ─── Load + render result ─────────────────────────────────────────────────
async function loadResult() {
  const res = await fetch(`/result/${currentJobId}`);
  if (!res.ok) { showError('Could not load results'); return; }
  const episode = await res.json();

  resultsSection.classList.add('visible');
  renderTaskBanner(episode);
  renderMetadata(episode);
  renderGallery(episode);
  renderStepsTable(episode);
  renderJson(episode);
}

function renderTaskBanner(ep) {
  const instruction = ep.metadata?.language_instruction || 'Task not detected';
  taskText.textContent = instruction;
}

function renderMetadata(ep) {
  const m = ep.metadata || {};
  const cards = [
    { label: 'Duration',   value: (m.duration_seconds || 0).toFixed(1), unit: 's' },
    { label: 'Key Frames', value: m.key_frames_count || 0,              unit: '' },
    { label: 'Total Frames', value: m.total_frames || 0,                unit: '' },
    { label: 'FPS',        value: (m.fps || 0).toFixed(1),              unit: '' },
    { label: 'Resolution', value: `${m.resolution?.width || '?'}×${m.resolution?.height || '?'}`, unit: '' },
    { label: 'Annotator',  value: (m.annotation_model || 'unknown').replace('claude-', 'Claude '), unit: '' },
  ];

  metaGrid.innerHTML = cards.map(c => `
    <div class="meta-card">
      <div class="meta-label">${c.label}</div>
      <div class="meta-value">${c.value}<span class="meta-unit">${c.unit}</span></div>
    </div>
  `).join('');
}

function renderGallery(ep) {
  const steps = ep.steps || [];
  gallery.innerHTML = '';

  steps.forEach((step, i) => {
    const imgPath = step.annotated_image_path || step.image_path;
    if (!imgPath) return;

    const filename = imgPath.split('/').pop();
    const url = `/frames/${currentJobId}/${filename}`;

    const item = document.createElement('div');
    item.className = 'gallery-item';
    item.innerHTML = `
      <img src="${url}" alt="Frame ${step.step_id}" loading="lazy"
           onerror="this.style.display='none'">
      <div class="frame-label">Step ${step.step_id} · ${step.timestamp.toFixed(2)}s</div>
    `;
    item.querySelector('img').addEventListener('click', () => openLightbox(url));
    gallery.appendChild(item);
  });

  if (!gallery.children.length) {
    gallery.innerHTML = '<p style="color:var(--text-muted);font-size:13px;padding:8px 0">No annotated frames available.</p>';
  }
}

function renderStepsTable(ep) {
  const steps = ep.steps || [];
  stepsBody.innerHTML = '';

  steps.forEach(step => {
    const obs    = step.observation || {};
    const act    = step.action      || {};
    const objects = (obs.objects || []).map(o => o.label || '').filter(Boolean);
    const actionType = act.action_type || 'unknown';
    const badgeClass = getBadgeClass(actionType);

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${step.step_id}</td>
      <td>${step.timestamp.toFixed(3)}s</td>
      <td><span class="action-badge ${badgeClass}">${actionType}</span></td>
      <td>
        <div class="objects-list">
          ${objects.slice(0, 4).map(o => `<span class="obj-chip">${o}</span>`).join('')}
          ${objects.length > 4 ? `<span class="obj-chip">+${objects.length - 4}</span>` : ''}
        </div>
      </td>
      <td class="instruction-cell">${step.language_annotation || '—'}</td>
    `;
    stepsBody.appendChild(tr);
  });

  if (!stepsBody.children.length) {
    stepsBody.innerHTML = '<tr><td colspan="5" style="color:var(--text-muted);text-align:center;padding:20px">No steps found</td></tr>';
  }
}

function renderJson(ep) {
  const highlighted = syntaxHighlight(JSON.stringify(ep, null, 2));
  jsonPre.innerHTML = highlighted;
}

// ─── UI helpers ───────────────────────────────────────────────────────────
function setStatus(type, message, pct) {
  statusDot.className = `status-dot ${type}`;
  statusText.innerHTML = `<strong>${message}</strong>`;
  if (type === 'running' && pct !== undefined) {
    statusText.innerHTML += ` <span>${pct}% complete</span>`;
  }
}

function showError(msg) {
  errorPanel.classList.add('visible');
  errorPanel.querySelector('strong').textContent = 'Error';
  errorPanel.querySelector('pre').textContent = msg;
}

function getBadgeClass(actionType) {
  const map = { idle: 'idle', grasp: 'grasp', place: 'place', pick_up: 'pick_up',
                move_down: 'move_down', move_up: 'move_up', move_left: 'move_left',
                move_right: 'move_right' };
  return map[actionType] || '';
}

function openLightbox(src) {
  lightboxImg.src = src;
  lightbox.classList.add('open');
  document.body.style.overflow = 'hidden';
}

function closeLightbox() {
  lightbox.classList.remove('open');
  document.body.style.overflow = '';
  lightboxImg.src = '';
}

function formatBytes(bytes) {
  if (bytes < 1024)        return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// ─── JSON syntax highlighter ──────────────────────────────────────────────
function syntaxHighlight(json) {
  return json
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(
      /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
      match => {
        if (/^"/.test(match)) {
          if (/:$/.test(match)) return `<span class="json-key">${match}</span>`;
          return `<span class="json-string">${match}</span>`;
        }
        if (/true|false/.test(match)) return `<span class="json-bool">${match}</span>`;
        if (/null/.test(match))       return `<span class="json-null">${match}</span>`;
        return `<span class="json-number">${match}</span>`;
      }
    );
}

// ─── Toast notifications (lightweight) ───────────────────────────────────
function showToast(message, type = 'info') {
  const el = document.createElement('div');
  el.style.cssText = `
    position:fixed; bottom:24px; right:24px; z-index:9999;
    background:${type === 'error' ? 'rgba(248,113,113,0.15)' : 'rgba(91,124,250,0.15)'};
    border:1px solid ${type === 'error' ? 'rgba(248,113,113,0.4)' : 'rgba(91,124,250,0.4)'};
    color:${type === 'error' ? '#f87171' : '#7c9dfa'};
    padding:12px 18px; border-radius:8px; font-size:13px;
    box-shadow:0 4px 20px rgba(0,0,0,0.4);
    animation: fadeIn 0.2s ease;
  `;
  el.textContent = message;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 4000);
}
