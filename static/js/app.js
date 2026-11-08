document.addEventListener('DOMContentLoaded', () => {
    const canvasHelper = new InteractiveCanvas('editorCanvas');
    
    // UI Elements
    const videoSelect = document.getElementById('videoSelect');
    const btnLoadVideo = document.getElementById('btnLoadVideo');
    const fileUpload = document.getElementById('fileUpload');
    const uploadStatus = document.getElementById('uploadStatus');

    const videoNameSpan = document.getElementById('videoName');
    const videoResSpan = document.getElementById('videoRes');
    const videoFramesSpan = document.getElementById('videoFrames');
    const videoDurationSpan = document.getElementById('videoDuration');
    const videoFpsSpan = document.getElementById('videoFps');

    const scrubber = document.getElementById('scrubber');
    const timecodeDisplay = document.getElementById('timecodeDisplay');
    const frameCounter = document.getElementById('frameCounter');

    const btnFirst = document.getElementById('btnFirst');
    const btnPrev10 = document.getElementById('btnPrev10');
    const btnPrev = document.getElementById('btnPrev');
    const btnPlay = document.getElementById('btnPlay');
    const btnNext = document.getElementById('btnNext');
    const btnNext10 = document.getElementById('btnNext10');
    const btnLast = document.getElementById('btnLast');

    const btnModePositive = document.getElementById('btnModePositive');
    const btnModeNegative = document.getElementById('btnModeNegative');
    const btnClearFrame = document.getElementById('btnClearFrame');

    const btnTrackBackward = document.getElementById('btnTrackBackward');
    const btnTrackBoth = document.getElementById('btnTrackBoth');
    const btnTrackForward = document.getElementById('btnTrackForward');
    const progressContainer = document.getElementById('progressContainer');
    const progressFill = document.getElementById('progressFill');
    const progressStatusText = document.getElementById('progressStatusText');
    const progressPercentText = document.getElementById('progressPercentText');

    const btnExportModal = document.getElementById('btnExportModal');
    const exportModal = document.getElementById('exportModal');
    const btnCloseModal = document.getElementById('btnCloseModal');
    const btnDoExport = document.getElementById('btnDoExport');
    const exportFormatSelect = document.getElementById('exportFormatSelect');
    const exportResult = document.getElementById('exportResult');

    const modelSelect = document.getElementById('modelSelect');
    const gpuBadge = document.getElementById('gpuBadge');

    // State
    let currentFrameIdx = 0;
    let totalFrames = 0;
    let videoFps = 30.0;
    let isPlaying = false;
    let playInterval = null;
    let trackingPollInterval = null;

    // Load initial system info & videos
    initSystem();

    function initSystem() {
        fetch('/api/system_info')
            .then(res => res.json())
            .then(data => {
                if (data.cuda_available) {
                    gpuBadge.innerHTML = `<span class="badge-gpu-dot"></span> ${data.device_name} (${data.vram_gb} GB VRAM) • Unlimited`;
                } else {
                    gpuBadge.innerHTML = `<span style="color:#f85149">● CPU Mode</span>`;
                }

                // Populate models
                modelSelect.innerHTML = '';
                for (const [key, m] of Object.entries(data.models)) {
                    const opt = document.createElement('option');
                    opt.value = key;
                    opt.textContent = `${key.toUpperCase()} - ${m.description} ${m.downloaded ? '✓' : '(Needs Download)'}`;
                    if (key === data.active_model) opt.selected = true;
                    modelSelect.appendChild(opt);
                }
            });

        refreshVideosList();
    }

    // Model Switch Listener
    modelSelect.addEventListener('change', () => {
        const selectedModel = modelSelect.value;
        const formData = new FormData();
        formData.append('model_key', selectedModel);

        gpuBadge.innerHTML = `<span style="color:#d29922">Loading ${selectedModel.toUpperCase()}...</span>`;
        fetch('/api/models/select', {
            method: 'POST',
            body: formData
        })
        .then(async res => {
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Failed to switch model');
            }
            return res.json();
        })
        .then(data => {
            initSystem();
        })
        .catch(err => {
            alert('Model error: ' + err.message);
            initSystem();
        });
    });

    function refreshVideosList() {
        fetch('/api/videos/list')
            .then(res => res.json())
            .then(data => {
                videoSelect.innerHTML = '<option value="">-- Choose Video from inputs/ --</option>';
                data.videos.forEach(v => {
                    const opt = document.createElement('option');
                    opt.value = v.filename;
                    opt.textContent = `${v.filename} (${v.size_mb} MB)`;
                    videoSelect.appendChild(opt);
                });
            });
    }

    const uploadProgressBox = document.getElementById('uploadProgressBox');
    const uploadProgressBar = document.getElementById('uploadProgressBar');
    const btnOpenFolder = document.getElementById('btnOpenFolder');

    if (btnOpenFolder) {
        btnOpenFolder.addEventListener('click', () => {
            fetch('/api/open_inputs_folder', { method: 'POST' });
        });
    }

    // Video Upload with Live Progress Bar
    fileUpload.addEventListener('change', () => {
        if (!fileUpload.files.length) return;
        const file = fileUpload.files[0];
        const formData = new FormData();
        formData.append('file', file);

        if (uploadProgressBox) uploadProgressBox.style.display = 'flex';
        if (uploadProgressBar) uploadProgressBar.style.width = '0%';
        const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
        uploadStatus.textContent = `Uploading ${file.name} (0 / ${sizeMb} MB)...`;

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/api/videos/upload', true);

        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
                const percent = Math.round((e.loaded / e.total) * 100);
                if (uploadProgressBar) uploadProgressBar.style.width = percent + '%';
                const loadedMb = (e.loaded / (1024 * 1024)).toFixed(1);
                uploadStatus.textContent = `Uploading: ${percent}% (${loadedMb} / ${sizeMb} MB)`;
            }
        };

        xhr.onload = () => {
            if (xhr.status === 200) {
                const data = JSON.parse(xhr.responseText);
                if (uploadProgressBar) uploadProgressBar.style.width = '100%';
                uploadStatus.textContent = `✓ Uploaded: ${data.filename}`;
                refreshVideosList();
                setTimeout(() => {
                    if (uploadProgressBox) uploadProgressBox.style.display = 'none';
                }, 3500);
                videoSelect.value = data.filename;
            } else {
                uploadStatus.textContent = `Upload failed (${xhr.status}). Tip: Click 'Open Folder' and paste file directly!`;
            }
        };

        xhr.onerror = () => {
            uploadStatus.textContent = `Upload error! Tip: Click 'Open Folder' and drop the file directly into inputs.`;
        };

        xhr.send(formData);
    });

    // Load Video
    btnLoadVideo.addEventListener('click', () => {
        const filename = videoSelect.value;
        if (!filename) return alert('Please select or upload a video file first.');

        btnLoadVideo.disabled = true;
        btnLoadVideo.textContent = 'Extracting Frames & Loading SAM 2...';

        const formData = new FormData();
        formData.append('filename', filename);

        fetch('/api/videos/load', {
            method: 'POST',
            body: formData
        })
        .then(async res => {
            if (!res.ok) {
                let errText = await res.text();
                try {
                    const errJson = JSON.parse(errText);
                    if (errJson.detail) errText = errJson.detail;
                } catch(e) {}
                throw new Error(errText || `Server error (${res.status})`);
            }
            return res.json();
        })
        .then(data => {
            btnLoadVideo.disabled = false;
            btnLoadVideo.textContent = 'Load Video';

            if (data.status === 'loaded') {
                const info = data.info;
                videoNameSpan.textContent = info.filename;
                videoResSpan.textContent = `${info.width} x ${info.height}`;
                videoFramesSpan.textContent = info.total_frames;
                videoDurationSpan.textContent = `${info.duration} s`;
                videoFpsSpan.textContent = `${info.fps} fps`;

                totalFrames = info.total_frames;
                videoFps = info.fps;

                scrubber.max = totalFrames - 1;
                scrubber.value = 0;
                currentFrameIdx = 0;

                if (btnTrackBackward) btnTrackBackward.disabled = false;
                if (btnTrackBoth) btnTrackBoth.disabled = false;
                if (btnTrackForward) btnTrackForward.disabled = false;
                btnExportModal.disabled = false;

                loadFrame(0);
            }
        })
        .catch(err => {
            btnLoadVideo.disabled = false;
            btnLoadVideo.textContent = 'Load Video';
            alert('Failed to load video: ' + err.message);
        });
    });

    // Load Frame Image & Mask
    function loadFrame(idx) {
        if (totalFrames === 0) return;
        currentFrameIdx = Math.max(0, Math.min(idx, totalFrames - 1));
        scrubber.value = currentFrameIdx;

        // Update timecode
        updateTimecodeDisplay();

        // 1. Fetch Frame Image
        const img = new Image();
        img.src = `/api/frame/${currentFrameIdx}`;
        img.onload = () => {
            canvasHelper.setFrameImage(img);

            // 2. Fetch existing mask if available
            fetch(`/api/mask/${currentFrameIdx}`)
                .then(res => res.json())
                .then(data => {
                    if (data.has_mask && data.mask_b64) {
                        const maskImg = new Image();
                        maskImg.src = `data:image/png;base64,${data.mask_b64}`;
                        maskImg.onload = () => canvasHelper.setMaskImage(maskImg);
                    } else {
                        canvasHelper.setMaskImage(null);
                    }
                });
        };
    }

    function updateTimecodeDisplay() {
        const fps = videoFps || 30.0;
        const totalSecs = Math.floor(currentFrameIdx / fps);
        const frames = Math.floor(currentFrameIdx % fps);
        const hh = String(Math.floor(totalSecs / 3600)).padStart(2, '0');
        const mm = String(Math.floor((totalSecs % 3600) / 60)).padStart(2, '0');
        const ss = String(totalSecs % 60).padStart(2, '0');
        const ff = String(frames).padStart(2, '0');

        timecodeDisplay.textContent = `${hh}:${mm}:${ss}:${ff}`;
        frameCounter.textContent = `Frame ${currentFrameIdx} / ${totalFrames - 1}`;
    }

    // Scrubber event
    scrubber.addEventListener('input', (e) => {
        loadFrame(parseInt(e.target.value));
    });

    // Point Added Event from Canvas
    canvasHelper.onPointAdded = (x, y, label) => {
        fetch('/api/prompt/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                frame_idx: currentFrameIdx,
                x: x,
                y: y,
                label: label
            })
        })
        .then(res => res.json())
        .then(data => {
            if (data.mask_b64) {
                const maskImg = new Image();
                maskImg.src = `data:image/png;base64,${data.mask_b64}`;
                maskImg.onload = () => canvasHelper.setMaskImage(maskImg);
            }
        });
    };

    // Mode Toggle
    btnModePositive.addEventListener('click', () => {
        canvasHelper.setMode(1);
        btnModePositive.classList.add('active-positive');
        btnModeNegative.classList.remove('active-negative');
    });

    btnModeNegative.addEventListener('click', () => {
        canvasHelper.setMode(0);
        btnModeNegative.classList.add('active-negative');
        btnModePositive.classList.remove('active-positive');
    });

    // Clear Frame Prompts
    btnClearFrame.addEventListener('click', () => {
        fetch('/api/prompt/clear', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frame_idx: currentFrameIdx })
        })
        .then(res => res.json())
        .then(() => {
            canvasHelper.clearMask();
        });
    });

    // Transport Controls
    btnFirst.addEventListener('click', () => loadFrame(0));
    btnPrev10.addEventListener('click', () => loadFrame(currentFrameIdx - 10));
    btnPrev.addEventListener('click', () => loadFrame(currentFrameIdx - 1));
    btnNext.addEventListener('click', () => loadFrame(currentFrameIdx + 1));
    btnNext10.addEventListener('click', () => loadFrame(currentFrameIdx + 10));
    btnLast.addEventListener('click', () => loadFrame(totalFrames - 1));

    btnPlay.addEventListener('click', togglePlayback);

    function togglePlayback() {
        if (isPlaying) {
            clearInterval(playInterval);
            isPlaying = false;
            btnPlay.textContent = '▶ Play';
        } else {
            isPlaying = true;
            btnPlay.textContent = '⏸ Pause';
            const intervalMs = 1000 / (videoFps || 30);
            playInterval = setInterval(() => {
                if (currentFrameIdx >= totalFrames - 1) {
                    togglePlayback();
                } else {
                    loadFrame(currentFrameIdx + 1);
                }
            }, intervalMs);
        }
    }

    // Keyboard Shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        if (e.code === 'Space') {
            e.preventDefault();
            togglePlayback();
        } else if (e.code === 'ArrowRight') {
            loadFrame(currentFrameIdx + 1);
        } else if (e.code === 'ArrowLeft') {
            loadFrame(currentFrameIdx - 1);
        }
    });

    function setTrackingButtonsDisabled(disabled) {
        if (btnTrackBackward) btnTrackBackward.disabled = disabled;
        if (btnTrackBoth) btnTrackBoth.disabled = disabled;
        if (btnTrackForward) btnTrackForward.disabled = disabled;
    }

    function triggerTracking(direction) {
        setTrackingButtonsDisabled(true);
        progressContainer.style.display = 'flex';
        progressFill.style.width = '0%';
        const dirLabel = direction === 'backward' ? 'Backward' : (direction === 'forward' ? 'Forward' : 'Both Directions');
        progressStatusText.textContent = `Starting ${dirLabel} Tracking...`;

        fetch('/api/track/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                direction: direction,
                start_frame: currentFrameIdx
            })
        })
        .then(async res => {
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Could not start tracking');
            }
            return res.json();
        })
        .then(data => {
            pollTrackingStatus();
        })
        .catch(err => {
            alert('Tracking Error: ' + err.message);
            setTrackingButtonsDisabled(false);
            progressContainer.style.display = 'none';
        });
    }

    if (btnTrackBackward) btnTrackBackward.addEventListener('click', () => triggerTracking('backward'));
    if (btnTrackBoth) btnTrackBoth.addEventListener('click', () => triggerTracking('both'));
    if (btnTrackForward) btnTrackForward.addEventListener('click', () => triggerTracking('forward'));

    function pollTrackingStatus() {
        trackingPollInterval = setInterval(() => {
            fetch('/api/track/status')
                .then(res => res.json())
                .then(st => {
                    if (st.is_tracking) {
                        progressFill.style.width = `${st.percent}%`;
                        progressPercentText.textContent = `${st.percent}%`;
                        const dir = st.direction === 'backward' ? '◀ Backward' : (st.direction === 'forward' ? 'Forward ▶' : '◀⚡▶');
                        progressStatusText.textContent = `[${dir}] Frame ${st.current_frame} (${st.percent}%)`;
                    } else if (st.done) {
                        clearInterval(trackingPollInterval);
                        progressFill.style.width = '100%';
                        progressStatusText.textContent = 'Tracking Completed!';
                        setTrackingButtonsDisabled(false);
                        // Reload current frame mask
                        loadFrame(currentFrameIdx);
                    } else if (st.error) {
                        clearInterval(trackingPollInterval);
                        alert('Tracking Error: ' + st.error);
                        setTrackingButtonsDisabled(false);
                    }
                });
        }, 500);
    }

    // Export Modal
    const btnOpenOutputsFolder = document.getElementById('btnOpenOutputsFolder');
    if (btnOpenOutputsFolder) {
        btnOpenOutputsFolder.addEventListener('click', () => {
            fetch('/api/open_outputs_folder', { method: 'POST' });
        });
    }

    btnExportModal.addEventListener('click', () => {
        exportModal.style.display = 'flex';
        exportResult.innerHTML = '';
    });

    btnCloseModal.addEventListener('click', () => {
        exportModal.style.display = 'none';
    });

    btnDoExport.addEventListener('click', () => {
        const fmt = exportFormatSelect.value;
        btnDoExport.disabled = true;
        btnDoExport.textContent = 'Exporting...';
        exportResult.innerHTML = '<span style="color:#58a6ff;">Rendering export files... Please wait.</span>';

        fetch('/api/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ format: fmt, frame_idx: currentFrameIdx })
        })
        .then(res => res.json())
        .then(data => {
            btnDoExport.disabled = false;
            btnDoExport.textContent = 'Export';

            if (data.status === 'ok') {
                let downloadHtml = '';
                if (data.url) {
                    downloadHtml = `<a href="${data.url}" download class="btn btn-success" style="text-decoration:none; display:inline-block; margin-top:0.4rem; padding:0.4rem 0.8rem; font-weight:700;">📥 Download ${data.file}</a>`;
                }

                let folderHtml = '';
                if (data.folder) {
                    folderHtml = `<p style="font-size:0.75rem; color:#8b949e; margin-top:0.4rem; margin-bottom:0;">Folder: <code style="background:#0d1117; padding:2px 5px; border-radius:3px;">${data.folder}</code></p>`;
                }

                exportResult.innerHTML = `
                    <div style="color:#4ade80; margin-top:0.6rem; padding:0.6rem; background:#161b22; border-radius:6px; border:1px solid #238636;">
                        <b>✓ Export Complete!</b><br>
                        ${downloadHtml}
                        ${folderHtml}
                    </div>
                `;
            } else {
                exportResult.innerHTML = `<span style="color:#f85149">Error: ${data.detail || 'Export failed'}</span>`;
            }
        })
        .catch(err => {
            btnDoExport.disabled = false;
            btnDoExport.textContent = 'Export';
            exportResult.innerHTML = `<span style="color:#f85149">Export failed: ${err}</span>`;
        });
    });
});

