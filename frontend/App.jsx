const { useEffect, useRef, useState } = React;

function Tag({ type, title, children }) {
  return <span className={`tag ${type}`} title={title}>{children}</span>;
}

const MODEL_OPTIONS = [
  { value: "auto", label: "Auto (GPS-based)", description: "Picks indoor or outdoor model from your GPS location" },
  { value: "indoor", label: "Indoor", description: "Always use indoor.pt" },
  { value: "outdoor", label: "Outdoor", description: "Always use outdoor.pt" },
  { value: "base", label: "Base (YOLO)", description: "Always use yolo26n.pt" },
];

function SelectField({ id, label, value, onChange, options }) {
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="select-input"
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value} title={opt.description}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}

function NumberField({ id, label, value, onChange, ...props }) {
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        type="number"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        {...props}
      />
    </div>
  );
}

function buildSummary(result) {
  const tags = [];

  Object.entries(result.objects || {}).forEach(([name, data]) => {
    tags.push({ type: "object", label: `${name}: ${data.count}` });
  });

  Object.entries(result.actions || {}).forEach(([name, data]) => {
    tags.push({ type: "action", label: `${name}: ${data.count}` });
  });

  (result.doors || []).forEach((door) => {
    tags.push({ type: "door", label: door.description });
  });

  if (result.gps) {
    tags.push({ type: "gps", label: `GPS: ${result.gps.source}` });
    if (result.gps.poi_result?.nearest?.name) {
      tags.push({ type: "gps", label: `Nearest: ${result.gps.poi_result.nearest.name}` });
    }
  }

  if (result.floor) {
    tags.push({
      type: "floor",
      label: `Floor ${result.floor.floor} (${result.floor.range_m.min}-${result.floor.range_m.max} m)`,
    });
  }

  if (result.model_info) {
    const mi = result.model_info;
    tags.push({
      type: "model",
      label: `Model: ${mi.selected} (${mi.mode})`,
      title: mi.reason,
    });
  }

  return tags.length ? tags : [{ type: "object", label: "No detections" }];
}

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [lat, setLat] = useState(24.967545);
  const [lng, setLng] = useState(121.187578);
  const [altitude, setAltitude] = useState(105);
  const [radius, setRadius] = useState("100");
  const [modelMode, setModelMode] = useState("auto");
  const [sampleEvery, setSampleEvery] = useState("1");
  const [maxFrames, setMaxFrames] = useState("30");
  const [cameraInterval, setCameraInterval] = useState("1");
  const [result, setResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [cameraStatus, setCameraStatus] = useState("");
  const [isCameraActive, setIsCameraActive] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef = useRef(null);
  const requestInFlightRef = useRef(false);
  const frameNumberRef = useRef(0);

  const isVideo = file?.type.startsWith("video/");

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      stopCamera();
    };
  }, [previewUrl]);

  function appendLocationFields(formData) {
    if (lat) formData.append("lat", lat);
    if (lng) formData.append("lng", lng);
    if (altitude) formData.append("altitude", altitude);
    if (radius) formData.append("gps_radius", radius);
    formData.append("model", modelMode);
  }

  function handleFileChange(event) {
    const selectedFile = event.target.files?.[0];
    if (!selectedFile) return;

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setResult(null);
  }

  async function analyzeUpload() {
    if (!file) return;
    stopCamera();
    setIsAnalyzing(true);
    setResult(null);

    const formData = new FormData();
    formData.append("image", file);
    appendLocationFields(formData);

    if (isVideo) {
      if (sampleEvery) formData.append("sample_interval_sec", sampleEvery);
      if (maxFrames) formData.append("max_frames", maxFrames);
    }

    try {
      const response = await fetch("/api/analyze", { method: "POST", body: formData });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Analysis failed");
      setResult(data);
    } catch (error) {
      setResult({ error: error.message });
    } finally {
      setIsAnalyzing(false);
    }
  }

  async function analyzeCameraFrame() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!streamRef.current || requestInFlightRef.current || !video || video.videoWidth === 0) {
      return;
    }

    requestInFlightRef.current = true;
    frameNumberRef.current += 1;
    setCameraStatus(`Inferencing frame ${frameNumberRef.current}...`);

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);

    try {
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.85));
      if (!blob) throw new Error("Could not capture webcam frame");

      const formData = new FormData();
      formData.append("image", blob, `webcam-frame-${frameNumberRef.current}.jpg`);
      appendLocationFields(formData);

      const response = await fetch("/api/analyze", { method: "POST", body: formData });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Webcam inference failed");

      setResult({
        ...data,
        media_type: "webcam",
        webcam: {
          frame_number: frameNumberRef.current,
          captured_at: new Date().toISOString(),
        },
      });
      setCameraStatus(`Live inferencing. Last frame: ${frameNumberRef.current}`);
    } catch (error) {
      setCameraStatus(`Webcam error: ${error.message}`);
    } finally {
      requestInFlightRef.current = false;
    }
  }

  async function startCamera() {
    try {
      if (streamRef.current) return;
      setResult(null);
      frameNumberRef.current = 0;

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false,
      });

      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      setIsCameraActive(true);
      setCameraStatus("Camera started. Waiting for first frame...");

      const intervalMs = Math.max(200, Number(cameraInterval || 1) * 1000);
      timerRef.current = setInterval(analyzeCameraFrame, intervalMs);
      videoRef.current.onloadedmetadata = () => analyzeCameraFrame();
    } catch (error) {
      setCameraStatus(`Could not start webcam: ${error.message}`);
      stopCamera();
    }
  }

  function stopCamera() {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    requestInFlightRef.current = false;
    if (videoRef.current) videoRef.current.srcObject = null;
    setIsCameraActive(false);
    setCameraStatus(frameNumberRef.current ? "Camera stopped." : "");
  }

  const summaryTags = result && !result.error ? buildSummary(result) : [];

  return (
    <main className="app">
      <header className="app-header">
        <h1>Surrounding Awareness</h1>
        <p>Upload media or stream your webcam for continuous surrounding awareness.</p>
      </header>

      <section className="panel">
        <h2>Upload & Settings</h2>
        <label className={`file-drop ${file ? "has-file" : ""}`}>
          <span>{file ? file.name : "Click or drag to upload an image or video"}</span>
          <input type="file" accept="image/*,video/*" onChange={handleFileChange} />
        </label>

        {file && isVideo && (
          <video className="preview" src={previewUrl} controls muted playsInline />
        )}
        {file && !isVideo && <img className="preview" src={previewUrl} alt="Selected upload" />}

        <div className="grid">
          <NumberField id="lat-input" label="Latitude" value={lat} onChange={setLat} step="any" />
          <NumberField id="lng-input" label="Longitude" value={lng} onChange={setLng} step="any" />
          <NumberField id="altitude-input" label="Altitude (m)" value={altitude} onChange={setAltitude} step="any" />
          <NumberField id="radius-input" label="Radius (m)" value={radius} onChange={setRadius} min="1" max="5000" />
        </div>

        <div className="grid two model-row">
          <SelectField
            id="model-select"
            label="Model"
            value={modelMode}
            onChange={setModelMode}
            options={MODEL_OPTIONS}
          />
          <div className="field model-hint">
            <label>Selected model</label>
            <div className="model-badge">
              <span className={`model-icon model-${modelMode}`}></span>
              {MODEL_OPTIONS.find((o) => o.value === modelMode)?.description}
            </div>
          </div>
        </div>

        {isVideo && (
          <div className="grid two">
            <NumberField id="sample-input" label="Sample every (sec)" value={sampleEvery} onChange={setSampleEvery} min="0.1" step="0.1" />
            <NumberField id="max-frames-input" label="Max frames" value={maxFrames} onChange={setMaxFrames} min="1" max="300" />
          </div>
        )}

        <button className="button" disabled={!file || isAnalyzing} onClick={analyzeUpload}>
          {isAnalyzing ? "Analyzing..." : isVideo ? "Analyze Video" : "Analyze Image"}
        </button>
      </section>

      <section className="panel">
        <h2>Live Webcam</h2>
        <video ref={videoRef} className={`preview ${isCameraActive ? "" : "hidden"}`} autoPlay muted playsInline />
        <canvas ref={canvasRef} className="hidden" />

        <div className="grid two">
          <NumberField id="camera-interval-input" label="Infer every (sec)" value={cameraInterval} onChange={setCameraInterval} min="0.2" step="0.1" />
        </div>

        <div className="actions">
          <button className="button secondary" disabled={isCameraActive} onClick={startCamera}>
            Start Webcam
          </button>
          <button className="button stop" disabled={!isCameraActive} onClick={stopCamera}>
            Stop Webcam
          </button>
        </div>
        <div className="status">{cameraStatus}</div>
      </section>

      {result && (
        <section className="panel">
          <h2>Results</h2>
          {result.error ? (
            <pre>{`Error: ${result.error}`}</pre>
          ) : (
            <>
              <div className="summary">
                {summaryTags.map((tag, index) => (
                  <Tag key={`${tag.type}-${tag.label}-${index}`} type={tag.type} title={tag.title}>
                    {tag.label}
                  </Tag>
                ))}
              </div>
              <pre>{JSON.stringify(result, null, 2)}</pre>
            </>
          )}
        </section>
      )}
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
