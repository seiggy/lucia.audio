// Lucia.Audio Dashboard

// ─── Tab switching ───
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    });
});

// ─── File input preview ───
document.getElementById('voice-audio').addEventListener('change', function(e) {
    const file = e.target.files[0];
    const label = document.getElementById('file-label');
    const preview = document.getElementById('upload-preview');
    recordedBlob = null; // clear any recording

    if (file) {
        label.textContent = file.name;
        const url = URL.createObjectURL(file);
        preview.src = url;
        preview.style.display = 'block';
    } else {
        label.textContent = 'Choose WAV file...';
        preview.style.display = 'none';
    }
});

// ─── Mic recording ───
let mediaRecorder = null;
let recordedChunks = [];
let recordedBlob = null;
let recordStartTime = 0;
let timerInterval = null;
let audioContext = null;
let analyserNode = null;
let levelRAF = null;

const recordBtn = document.getElementById('record-btn');
const recordTimer = document.getElementById('record-timer');
const recordLevel = document.getElementById('record-level');
const preview = document.getElementById('upload-preview');

recordBtn.addEventListener('click', async () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        stopRecording();
    } else {
        await startRecording();
    }
});

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: { channelCount: 1, sampleRate: 44100, echoCancellation: false, noiseSuppression: false }
        });

        // Level meter
        audioContext = new AudioContext();
        const source = audioContext.createMediaStreamSource(stream);
        analyserNode = audioContext.createAnalyser();
        analyserNode.fftSize = 256;
        source.connect(analyserNode);
        updateLevel();

        recordedChunks = [];
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) recordedChunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
            stream.getTracks().forEach(t => t.stop());
            cancelAnimationFrame(levelRAF);
            if (audioContext) { audioContext.close(); audioContext = null; }
            recordLevel.style.width = '0%';

            // Convert webm to WAV
            const webmBlob = new Blob(recordedChunks, { type: 'audio/webm' });
            recordedBlob = await convertToWav(webmBlob);

            const url = URL.createObjectURL(recordedBlob);
            preview.src = url;
            preview.style.display = 'block';

            // Clear file input so form uses recording
            document.getElementById('voice-audio').value = '';
            document.getElementById('file-label').textContent = 'Choose WAV file...';
        };

        mediaRecorder.start(100);
        recordStartTime = Date.now();
        recordBtn.textContent = '⏹ Stop Recording';
        recordBtn.classList.add('recording');
        timerInterval = setInterval(updateTimer, 100);

    } catch (err) {
        alert('Microphone access denied: ' + err.message);
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }
    clearInterval(timerInterval);
    recordBtn.textContent = '⏺ Start Recording';
    recordBtn.classList.remove('recording');
}

function updateTimer() {
    const elapsed = (Date.now() - recordStartTime) / 1000;
    const mins = Math.floor(elapsed / 60).toString().padStart(1, '0');
    const secs = Math.floor(elapsed % 60).toString().padStart(2, '0');
    const tenths = Math.floor((elapsed * 10) % 10);
    recordTimer.textContent = `${mins}:${secs}.${tenths}`;
}

function updateLevel() {
    if (!analyserNode) return;
    const data = new Uint8Array(analyserNode.frequencyBinCount);
    analyserNode.getByteFrequencyData(data);
    const avg = data.reduce((a, b) => a + b, 0) / data.length;
    recordLevel.style.width = Math.min(100, (avg / 128) * 100) + '%';
    levelRAF = requestAnimationFrame(updateLevel);
}

// Convert webm/opus blob to 16-bit PCM WAV using AudioContext
async function convertToWav(blob) {
    const arrayBuffer = await blob.arrayBuffer();
    const ctx = new OfflineAudioContext(1, 1, 44100); // temp context for decoding
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer);

    // Resample to 44100 mono
    const numSamples = audioBuffer.length;
    const sampleRate = audioBuffer.sampleRate;
    const channelData = audioBuffer.getChannelData(0);

    // Encode to WAV
    const wavBuffer = encodeWav(channelData, sampleRate);
    return new Blob([wavBuffer], { type: 'audio/wav' });
}

function encodeWav(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    // RIFF header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');

    // fmt chunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);           // chunk size
    view.setUint16(20, 1, true);            // PCM
    view.setUint16(22, 1, true);            // mono
    view.setUint32(24, sampleRate, true);   // sample rate
    view.setUint32(28, sampleRate * 2, true); // byte rate
    view.setUint16(32, 2, true);            // block align
    view.setUint16(34, 16, true);           // bits per sample

    // data chunk
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    // PCM samples
    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        offset += 2;
    }

    return buffer;
}

function writeString(view, offset, str) {
    for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
    }
}

// ─── Upload form (handles both file and recording) ───
document.getElementById('upload-form').addEventListener('submit', async function(e) {
    e.preventDefault();

    const btn = document.getElementById('upload-btn');
    const btnText = btn.querySelector('.btn-text');
    const btnLoading = btn.querySelector('.btn-loading');
    const fileInput = document.getElementById('voice-audio');

    // Validate: must have either a file or a recording
    if (!fileInput.files[0] && !recordedBlob) {
        alert('Please upload a WAV file or record from your microphone.');
        return;
    }

    btn.disabled = true;
    btnText.style.display = 'none';
    btnLoading.style.display = 'inline';

    const formData = new FormData(this);

    // If we have a recording, replace the file field
    if (recordedBlob && !fileInput.files[0]) {
        formData.set('audio', recordedBlob, 'recording.wav');
    }

    try {
        const resp = await fetch('/api/voices', {
            method: 'POST',
            body: formData,
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Upload failed');
        }

        window.location.reload();
    } catch (err) {
        alert('Error: ' + err.message);
    } finally {
        btn.disabled = false;
        btnText.style.display = 'inline';
        btnLoading.style.display = 'none';
    }
});

// TTS test form
document.getElementById('tts-form').addEventListener('submit', async function(e) {
    e.preventDefault();

    const btn = document.getElementById('tts-btn');
    const btnText = btn.querySelector('.btn-text');
    const btnLoading = btn.querySelector('.btn-loading');
    const result = document.getElementById('tts-result');
    const audio = document.getElementById('tts-audio');
    const timing = document.getElementById('tts-timing');

    btn.disabled = true;
    btnText.style.display = 'none';
    btnLoading.style.display = 'inline';
    result.style.display = 'none';

    const formData = new FormData(this);
    const startTime = performance.now();

    try {
        const resp = await fetch('/api/tts', {
            method: 'POST',
            body: formData,
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Synthesis failed');
        }

        const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);

        audio.src = url;
        timing.textContent = `Generated in ${elapsed}s`;
        result.style.display = 'block';
        audio.play();
    } catch (err) {
        alert('Error: ' + err.message);
    } finally {
        btn.disabled = false;
        btnText.style.display = 'inline';
        btnLoading.style.display = 'none';
    }
});

// Delete voice
async function deleteVoice(id, name) {
    if (!confirm(`Delete voice "${name}"? This cannot be undone.`)) return;

    try {
        const resp = await fetch(`/api/voices/${id}`, { method: 'DELETE' });
        if (!resp.ok) throw new Error('Delete failed');
        window.location.reload();
    } catch (err) {
        alert('Error: ' + err.message);
    }
}

// Play voice reference audio
function playVoiceRef(id) {
    const audio = new Audio(`/api/voices/${id}/audio`);
    audio.play();
}
