import { useState, useRef, useCallback } from 'react';
import DotField from './components/DotField';
import './App.css';

// All backend requests go through the Vite proxy (see vite.config.js),
// which forwards /gradio_api/* to http://127.0.0.1:7860. This avoids CORS.
const API_PREFIX = '/gradio_api';
const FN_INDEX = 0; // first registered .click() in bird_complete_local.py = predict_bird

function randomSessionHash() {
  return Math.random().toString(36).slice(2) + Math.random().toString(36).slice(2);
}

function parseBirdResult(text) {
  if (!text || typeof text !== 'string') {
    return { species: 'Unknown', confidence: '—', habitat: '—', migration: '—', similar: '', raw: String(text || '') };
  }
  const get = (re) => {
    const m = text.match(re);
    return m ? m[1].trim() : '';
  };
  const species = get(/SPECIES\s*:\s*([^\n]+)/i) || 'Unknown';
  const confidence =
    get(/((?:HIGH|MODERATE|LOW)\s+CONFIDENCE\s*\([^)]+\)[^\n]*)/i) || '—';
  const habitat = get(/FOUND IN\s*:\s*([^\n]+)/i) || '—';
  const migration = get(/MIGRATION\s*:\s*([^\n]+)/i) || '—';
  const similarMatch = text.match(/LOOKS SIMILAR TO\s*:\s*([^\n]+)[\s\S]*?How to tell apart\s*:\s*([^\n]+)/i);
  const similar = similarMatch ? `${similarMatch[1].trim()} — ${similarMatch[2].trim()}` : '';
  return { species, confidence, habitat, migration, similar, raw: text };
}

function App() {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  const uploadSectionRef = useRef(null);

  const scrollToUpload = () => {
    uploadSectionRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) {
      setError('Please select a valid image file.');
      return;
    }
    setError(null);
    setResult(null);
    setImageFile(file);
    const reader = new FileReader();
    reader.onload = (e) => setImagePreview(e.target.result);
    reader.readAsDataURL(file);
  }, []);

  const onFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const identifyBird = async () => {
    if (!imageFile) return;
    setIsLoading(true);
    setError(null);
    setResult(null);

    const session_hash = randomSessionHash();

    try {
      // 1) Upload the image to Gradio so we get a server-side path back.
      const fd = new FormData();
      fd.append('files', imageFile);
      const uploadRes = await fetch(
        `${API_PREFIX}/upload?upload_id=${session_hash}`,
        { method: 'POST', body: fd }
      );
      if (!uploadRes.ok) {
        throw new Error(`Upload failed (${uploadRes.status})`);
      }
      const uploaded = await uploadRes.json();
      const serverPath = Array.isArray(uploaded) ? uploaded[0] : uploaded;
      if (!serverPath) throw new Error('No path returned from upload');

      // 2) Submit the prediction job to the queue.
      const payload = {
        data: [
          {
            path: serverPath,
            orig_name: imageFile.name,
            size: imageFile.size,
            mime_type: imageFile.type,
            meta: { _type: 'gradio.FileData' },
          },
        ],
        event_data: null,
        fn_index: FN_INDEX,
        trigger_id: null,
        session_hash,
      };
      const joinRes = await fetch(`${API_PREFIX}/queue/join`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!joinRes.ok) {
        const detail = await joinRes.text().catch(() => '');
        throw new Error(`queue/join failed (${joinRes.status}) ${detail}`);
      }

      // 3) Stream events via SSE until the job completes.
      const text = await new Promise((resolve, reject) => {
        const source = new EventSource(
          `${API_PREFIX}/queue/data?session_hash=${session_hash}`
        );
        const timeout = setTimeout(() => {
          source.close();
          reject(new Error('Timed out waiting for prediction'));
        }, 120000);

        source.onmessage = (evt) => {
          try {
            const msg = JSON.parse(evt.data);
            if (msg.msg === 'process_completed') {
              clearTimeout(timeout);
              source.close();
              if (msg.success === false) {
                reject(new Error(msg.output?.error || 'Prediction failed'));
                return;
              }
              const data = msg.output?.data;
              resolve(Array.isArray(data) ? data[0] : data);
            } else if (msg.msg === 'unexpected_error' || msg.msg === 'queue_full') {
              clearTimeout(timeout);
              source.close();
              reject(new Error(msg.message || msg.msg));
            }
          } catch (parseErr) {
            // ignore keep-alive / non-JSON
          }
        };

        source.onerror = () => {
          clearTimeout(timeout);
          source.close();
          reject(new Error('Connection to backend lost (SSE)'));
        };
      });

      setResult(parseBirdResult(text));
    } catch (err) {
      setError(
        `Could not reach the backend at http://127.0.0.1:7860. ` +
          `Make sure bird_complete_local.py is running. (${err.message})`
      );
    } finally {
      setIsLoading(false);
    }
  };

  const clearImage = () => {
    setImageFile(null);
    setImagePreview(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="app">
      {/* HERO */}
      <section className="hero">
        <div className="hero-bg">
          <DotField />
        </div>
        <div className="hero-content">
          <h1 className="hero-title">
            <span className="hero-emoji">🐦</span> Bird Species Identifier
          </h1>
          <p className="hero-subtitle">
            AI-powered identification of 200 bird species
          </p>
          <button className="cta-button" onClick={scrollToUpload}>
            Try it Now <span className="cta-arrow">↓</span>
          </button>
        </div>
      </section>

      {/* UPLOAD & RESULTS */}
      <section className="upload-section" ref={uploadSectionRef}>
        <div className="container">
          <h2 className="section-title">Upload a Bird Photo</h2>
          <p className="section-subtitle">
            Drop an image below and let the model identify the species
          </p>

          <div className="upload-grid">
            <div
              className={`dropzone ${isDragging ? 'dragging' : ''} ${
                imagePreview ? 'has-image' : ''
              }`}
              onDrop={onDrop}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onClick={() => !imagePreview && fileInputRef.current?.click()}
            >
              <input
                type="file"
                accept="image/*"
                ref={fileInputRef}
                onChange={onFileChange}
                style={{ display: 'none' }}
              />
              {imagePreview ? (
                <div className="preview-wrapper">
                  <img src={imagePreview} alt="Preview" className="preview-image" />
                  <button
                    className="clear-button"
                    onClick={(e) => {
                      e.stopPropagation();
                      clearImage();
                    }}
                  >
                    ✕
                  </button>
                </div>
              ) : (
                <div className="dropzone-empty">
                  <div className="dropzone-icon">📷</div>
                  <p className="dropzone-text">
                    <strong>Drag & drop</strong> a bird photo here
                  </p>
                  <p className="dropzone-hint">or click to browse</p>
                </div>
              )}
            </div>

            <div className="results-panel">
              <button
                className="identify-button"
                onClick={identifyBird}
                disabled={!imageFile || isLoading}
              >
                {isLoading ? (
                  <>
                    <span className="spinner" /> Identifying...
                  </>
                ) : (
                  'Identify Bird'
                )}
              </button>

              {error && <div className="error-box">{error}</div>}

              {result && (
                <div className="result-box">
                  <div className="result-row">
                    <span className="result-label">Species</span>
                    <span className="result-value species">{result.species}</span>
                  </div>
                  <div className="result-row">
                    <span className="result-label">Confidence</span>
                    <span className="result-value">{result.confidence}</span>
                  </div>
                  <div className="result-row">
                    <span className="result-label">Habitat</span>
                    <span className="result-value">{result.habitat}</span>
                  </div>
                  <div className="result-row">
                    <span className="result-label">Migration</span>
                    <span className="result-value">{result.migration}</span>
                  </div>
                  {result.similar && (
                    <div className="result-row">
                      <span className="result-label">Similar Species</span>
                      <span className="result-value">{result.similar}</span>
                    </div>
                  )}
                  <details className="result-raw">
                    <summary>Full output (top 5 predictions)</summary>
                    <pre>{result.raw}</pre>
                  </details>
                </div>
              )}

              {!result && !error && !isLoading && (
                <div className="placeholder-box">
                  Results will appear here after identification.
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* FEATURE CARDS */}
      <section className="features-section">
        <div className="container">
          <h2 className="section-title">What You Get</h2>
          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon">🌍</div>
              <h3 className="feature-title">Habitat Info</h3>
              <p className="feature-text">
                Discover where each species lives, its preferred environment, and global range.
              </p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">✈️</div>
              <h3 className="feature-title">Migration Data</h3>
              <p className="feature-text">
                Learn migration patterns, seasonal movements, and travel routes for each bird.
              </p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">⚠️</div>
              <h3 className="feature-title">Similar Species Warnings</h3>
              <p className="feature-text">
                Get alerted to look-alike species so you can spot the subtle differences.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="footer">
        Built with EfficientNet V2 + PyTorch | React Bits DotField UI
      </footer>
    </div>
  );
}

export default App;
