const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const processingView = document.getElementById('processing-view');
const resultView = document.getElementById('result-view');
const audioPlayerSection = document.getElementById('audio-player-section'); // New Top Section
const customWordsInput = document.getElementById('custom-words');
const downloadBtn = document.getElementById('download-btn');
const resetBtn = document.getElementById('reset-btn');
const jsonReport = document.getElementById('json-report');
const redactionLog = document.getElementById('redaction-log');
const aiSummary = document.getElementById('ai-summary');
const playPauseBtn = document.getElementById('play-pause-btn');
const timeDisplay = document.getElementById('time-display');

// Noise Reduction Elements
const denoiseToggle = document.getElementById('denoise-toggle');
const denoiseSliderGroup = document.getElementById('denoise-slider-group');
const denoiseIntensity = document.getElementById('denoise-intensity');
const denoiseValue = document.getElementById('denoise-value');

// Stats Elements
const statDuration = document.getElementById('stat-duration');
const statCount = document.getElementById('stat-count');
const statTime = document.getElementById('stat-time');

// Global Ref for file
let currentFile = null;
let wavesurfer = null;
let wsRegions = null;

// --- Events ---

dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

resetBtn.addEventListener('click', resetUI);
if (playPauseBtn) playPauseBtn.addEventListener('click', togglePlayPause);

// Noise Reduction Toggle Handler
if (denoiseToggle) {
    denoiseToggle.addEventListener('change', () => {
        if (denoiseSliderGroup) {
            denoiseSliderGroup.style.display = denoiseToggle.checked ? 'block' : 'none';
        }
    });
}

// Noise Reduction Intensity Slider Handler
if (denoiseIntensity && denoiseValue) {
    denoiseIntensity.addEventListener('input', () => {
        denoiseValue.textContent = denoiseIntensity.value;
    });
}

// --- Logic ---

async function handleFile(file) {
    if (!file.type.startsWith('audio/')) {
        alert('Please upload an audio file.');
        return;
    }

    currentFile = file; // Store for waveform loading

    // UI Transition
    dropZone.classList.add('hidden');
    processingView.classList.remove('hidden');

    // Build Form Data
    const formData = new FormData();
    formData.append('file', file);
    formData.append('entities', customWordsInput.value);

    const mode = document.querySelector('input[name="mode"]:checked').value;
    formData.append('mode', mode);

    // Add noise reduction parameters
    const denoise = denoiseToggle ? denoiseToggle.checked : false;
    formData.append('denoise', denoise);
    if (denoise && denoiseIntensity) {
        formData.append('denoise_intensity', denoiseIntensity.value);
    }

    // Add detection mode
    const detectionModeEl = document.querySelector('input[name="detection_mode"]:checked');
    const detectionMode = detectionModeEl ? detectionModeEl.value : 'gemini';
    formData.append('detection_mode', detectionMode);

    try {
        const response = await fetch('/redact', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            try {
                const err = await response.json();
                throw new Error(err.error || 'Server returned error');
            } catch (e) {
                throw new Error('Server returned error ' + response.status);
            }
        }

        // --- NDJSON STREAM READER ---
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep partial line

            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    handleStreamMessage(JSON.parse(line));
                } catch (e) {
                    console.error("JSON parse error", e, line);
                }
            }
        }

    } catch (error) {
        console.error(error);
        alert(`Error: ${error.message}`);
        resetUI();
    }
}

function handleStreamMessage(msg) {
    if (msg.status === 'progress') {
        const statusText = document.querySelector('.status-text');
        const subStatus = document.querySelector('.sub-status');
        statusText.innerText = msg.message;
        if (msg.step && msg.total_steps) {
            subStatus.innerText = `Step ${msg.step} of ${msg.total_steps}`;
        }
    } else if (msg.status === 'complete') {
        const data = msg.data;
        if (!data.download_url && data.redacted_file) {
            data.download_url = `/download/${data.redacted_file.split(/[\\/\\\\]/).pop()}`;
        }
        showResults(data);
    } else if (msg.status === 'error') {
        throw new Error(msg.error);
    }
}

async function showResults(data) {
    processingView.classList.add('hidden');
    resultView.classList.remove('hidden');
    audioPlayerSection.classList.remove('hidden'); // Show Top Player

    // Update Stats
    statDuration.innerText = `${data.original_duration_seconds.toFixed(1)}s`;
    statCount.innerText = data.redaction_count;
    statTime.innerText = `${data.compute_time_seconds.toFixed(2)}s`;

    // Setup Download
    downloadBtn.href = data.download_url;
    downloadBtn.download = `redacted_${data.original_file.split('/').pop()}`;

    // JSON Report
    jsonReport.innerHTML = syntaxHighlight(data);

    // Display AI Summary
    if (data.summary) {
        aiSummary.innerHTML = formatSummary(data.summary);
    } else {
        aiSummary.innerHTML = '<span class="text-muted">No summary available.</span>';
    }

    // Initialize Waveform
    await initWaveform(data);

    // Render Redaction Log
    renderRedactionLog(data.detections);
}

async function initWaveform(reportData) {
    if (wavesurfer) {
        wavesurfer.destroy();
    }

    // Create WaveSurfer
    wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: '#475569',
        progressColor: '#06b6d4', // Primary Cyan
        cursorColor: '#f8fafc',
        barWidth: 2,
        barRadius: 2,
        height: 100,
        normalize: true,
        plugins: [
            WaveSurfer.Regions.create(),
            WaveSurfer.Timeline.create({
                container: '#waveform-timeline',
                primaryColor: '#94a3b8',
                secondaryColor: '#64748b',
                primaryLabelColor: '#94a3b8',
                secondaryLabelColor: '#64748b',
            })
        ]
    });

    wsRegions = wavesurfer.plugins[0]; // Regions plugin

    // Load original audio
    if (currentFile) {
        const url = URL.createObjectURL(currentFile); // Uses the file from memory
        wavesurfer.load(url);
    }

    // Event listeners
    wavesurfer.on('ready', () => {
        // Add Regions
        if (reportData.detections) {
            reportData.detections.forEach((d, index) => {
                const color = getRegionColor(d.label);
                wsRegions.addRegion({
                    start: d.start,
                    end: d.end,
                    content: d.label,
                    color: color,
                    drag: false,
                    resize: false
                });
            });
        }
        updateTimeDisplay();
    });

    wavesurfer.on('audioprocess', updateTimeDisplay);
    wavesurfer.on('seek', updateTimeDisplay);
    wavesurfer.on('finish', () => {
        if (playPauseBtn) {
            playPauseBtn.innerHTML = `
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="5 3 19 12 5 21 5 3"></polygon>
                </svg>
            `;
        }
    });

    wavesurfer.on('interaction', () => {
        if (playPauseBtn) {
            const isPlaying = wavesurfer.isPlaying();
            playPauseBtn.innerHTML = isPlaying
                ? `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="6" y1="4" x2="6" y2="20"></line><line x1="18" y1="4" x2="18" y2="20"></line></svg>`
                : `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>`;
        }
    });


    // Add region click handler
    wsRegions.on('region-clicked', (region, e) => {
        region.play();
        e.stopPropagation(); // Prevent seeking to click position if separate
    });
}

function getRegionColor(label) {
    const l = label.toUpperCase();
    // Use semi-transparent RGBA
    if (l === 'PER' || l === 'PERSON') return 'rgba(244, 63, 94, 0.4)'; // Rose
    if (l === 'ORG' || l === 'ORGANIZATION') return 'rgba(59, 130, 246, 0.4)'; // Blue
    if (l === 'LOC' || l === 'LOCATION') return 'rgba(16, 185, 129, 0.4)'; // Emerald
    if (l.includes('EMAIL')) return 'rgba(245, 158, 11, 0.4)'; // Amber
    if (l.includes('PRICE')) return 'rgba(139, 92, 246, 0.4)'; // Purple
    return 'rgba(100, 116, 139, 0.4)'; // Slate default
}

function togglePlayPause() {
    if (!wavesurfer) return;
    wavesurfer.playPause();

    const isPlaying = wavesurfer.isPlaying();
    playPauseBtn.innerHTML = isPlaying
        ? `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="6" y1="4" x2="6" y2="20"></line><line x1="18" y1="4" x2="18" y2="20"></line></svg>`
        : `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>`;
}

function updateTimeDisplay() {
    if (!wavesurfer) return;
    const curr = formatTime(wavesurfer.getCurrentTime());
    const total = formatTime(wavesurfer.getDuration());
    timeDisplay.innerText = `${curr} / ${total}`;
}


function renderRedactionLog(detections) {
    redactionLog.innerHTML = ''; // Clear previous

    if (!detections || detections.length === 0) {
        redactionLog.innerHTML = '<div class="text-center text-muted">No redactions found.</div>';
        return;
    }

    // Sort by start time just to be safe
    detections.sort((a, b) => a.start - b.start);

    detections.forEach((d, i) => {
        const card = document.createElement('div');
        card.className = 'redaction-card';

        // Format time
        const startStr = formatTime(d.start);
        const endStr = formatTime(d.end);

        // Determine badge class
        let badgeClass = 'badge-default';
        const label = d.label.toUpperCase();
        if (label === 'PER' || label === 'PERSON') badgeClass = 'badge-per';
        else if (label === 'ORG' || label === 'ORGANIZATION') badgeClass = 'badge-org';
        else if (label === 'LOC' || label === 'LOCATION') badgeClass = 'badge-loc';
        else if (label.includes('EMAIL') || label.includes('PHONE')) badgeClass = 'badge-email';
        else if (label.includes('PRICE') || label.includes('MONEY')) badgeClass = 'badge-price';

        card.innerHTML = `
            <div class="redaction-time">${startStr} - ${endStr}</div>
            <div class="redaction-content" title="${d.text}">${d.text}</div>
            <div class="redaction-badge ${badgeClass}">${d.label}</div>
        `;

        // Click to jump
        card.addEventListener('click', () => {
            if (wavesurfer) {
                wavesurfer.setTime(d.start);
                wavesurfer.play();
                const isPlaying = wavesurfer.isPlaying();
                if (playPauseBtn) playPauseBtn.innerHTML = isPlaying
                    ? `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="6" y1="4" x2="6" y2="20"></line><line x1="18" y1="4" x2="18" y2="20"></line></svg>`
                    : `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>`;
            }
        });

        redactionLog.appendChild(card);
    });
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 10); // One decimal place
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms}`;
}

// Pretty JSON Syntax Highlighter
function syntaxHighlight(json) {
    if (typeof json !== 'string') {
        json = JSON.stringify(json, undefined, 2);
    }
    json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
        var cls = 'number';
        if (/^"/.test(match)) {
            if (/:$/.test(match)) {
                cls = 'key';
            } else {
                cls = 'string';
            }
        } else if (/true|false/.test(match)) {
            cls = 'boolean';
        } else if (/null/.test(match)) {
            cls = 'null';
        }
        return '<span class="' + cls + '">' + match + '</span>';
    });
}

function resetUI() {
    resultView.classList.add('hidden');
    audioPlayerSection.classList.add('hidden'); // Hide Top Player
    processingView.classList.add('hidden');
    dropZone.classList.remove('hidden');
    fileInput.value = '';

    // Clear global file ref
    currentFile = null;

    if (wavesurfer) {
        wavesurfer.destroy();
        wavesurfer = null;
    }

    redactionLog.innerHTML = '';
    if (aiSummary) {
        aiSummary.innerHTML = 'Processing summary...';
    }
}

/**
 * Formats the AI summary for display, converting markdown-like formatting to HTML
 */
function formatSummary(summary) {
    if (!summary) return '';

    // Convert markdown-style bullet points to HTML
    let html = summary
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // Bold
        .replace(/\*(.*?)\*/g, '<em>$1</em>')              // Italic
        .replace(/^[-•]\s+(.*)$/gm, '<li>$1</li>')         // Bullet points
        .replace(/\n/g, '<br>');                           // Line breaks

    // Wrap list items in <ul> if there are any
    if (html.includes('<li>')) {
        html = html.replace(/(<li>.*?<\/li>)/gs, '<ul class="summary-list">$1</ul>');
        // Clean up multiple <ul> tags
        html = html.replace(/<\/ul><br><ul class="summary-list">/g, '');
    }

    return html;
}
