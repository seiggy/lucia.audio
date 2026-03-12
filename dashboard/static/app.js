// Lucia.Audio Dashboard

// File input preview
document.getElementById('voice-audio').addEventListener('change', function(e) {
    const file = e.target.files[0];
    const label = document.getElementById('file-label');
    const preview = document.getElementById('upload-preview');

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

// Upload form
document.getElementById('upload-form').addEventListener('submit', async function(e) {
    e.preventDefault();

    const btn = document.getElementById('upload-btn');
    const btnText = btn.querySelector('.btn-text');
    const btnLoading = btn.querySelector('.btn-loading');

    btn.disabled = true;
    btnText.style.display = 'none';
    btnLoading.style.display = 'inline';

    const formData = new FormData(this);

    try {
        const resp = await fetch('/api/voices', {
            method: 'POST',
            body: formData,
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Upload failed');
        }

        // Reload page to show new voice
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
