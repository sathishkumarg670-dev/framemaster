import { useEffect, useMemo, useState } from "react";

const API = "http://127.0.0.1:8000";

const COLORS = [
  "", "black", "white", "gray", "red", "green", "blue",
  "yellow", "orange", "brown", "purple", "pink",
];

function formatSeconds(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${Number(value).toFixed(2)}s`;
}

function percent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function cropUrl(path) {
  if (!path) return null;
  return `${API}/crops/image?path=${encodeURIComponent(path)}`;
}

function CameraBadge({ camera }) {
  return <span className="badge">{camera || "Unknown camera"}</span>;
}

function Dashboard() {
  const [health, setHealth] = useState(null);
  const [videos, setVideos] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [searchResults, setSearchResults] = useState([]);
  const [crossCamera, setCrossCamera] = useState([]);
  const [selected, setSelected] = useState(null);

  const [targetType, setTargetType] = useState("person");
  const [upper, setUpper] = useState("");
  const [lower, setLower] = useState("");
  const [backpack, setBackpack] = useState("");
  const [headwear, setHeadwear] = useState("");
  const [camera, setCamera] = useState("");
  const [objectType, setObjectType] = useState("");
  const [objectColor, setObjectColor] = useState("");
  const [brand, setBrand] = useState("");
  const [feature, setFeature] = useState("");
  const [limit, setLimit] = useState(20);

  const [selectedVideos, setSelectedVideos] = useState([]);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [reporting, setReporting] = useState(false);

  async function getJson(path, options) {
    const response = await fetch(`${API}${path}`, options);
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `${response.status} ${response.statusText}`);
    }
    return response.json();
  }

  async function refresh() {
    try {
      const [h, v, c, j] = await Promise.all([
        getJson("/health"),
        getJson("/videos"),
        getJson("/candidates?limit=50"),
        getJson("/jobs"),
      ]);
      setHealth(h);
      setVideos(v.videos || []);
      setCrossCamera(c.candidates || []);
      setJobs(j.jobs || []);
    } catch (error) {
      setMessage(`Backend connection failed: ${error.message}`);
    }
  }

  useEffect(() => {
    refresh();
    const timer = setInterval(async () => {
      try {
        const j = await getJson("/jobs");
        setJobs(j.jobs || []);
        if ((j.jobs || []).some((job) => ["completed", "failed"].includes(job.status))) {
          const h = await getJson("/health");
          setHealth(h);
        }
      } catch {
        // Keep the dashboard usable if the backend is temporarily unavailable.
      }
    }, 2500);
    return () => clearInterval(timer);
  }, []);

  const cameraOptions = useMemo(() => {
    const values = new Set();
    (health?.profile_statistics?.cameras
      ? Object.keys(health.profile_statistics.cameras)
      : []
    ).forEach((item) => values.add(item));
    videos.forEach((video) => {
      if (video.video_id) values.add(video.video_id);
    });
    return [...values].sort();
  }, [health, videos]);

  async function investigate() {
    setBusy(true);
    setMessage("");
    setSelected(null);

    const body = {
      target_type: targetType,
      upper_body_color: upper || null,
      lower_body_color: lower || null,
      backpack: backpack || null,
      headwear: headwear || null,
      object_type: objectType || null,
      object_color: objectColor || null,
      brand: brand || null,
      distinctive_feature: feature || null,
      camera_id: camera || null,
      limit: Number(limit),
    };

    try {
      const data = await getJson("/investigate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      setSearchResults(data.candidates || []);
      setMessage(
        data.status === "object_index_not_ready"
          ? data.message
          : `${data.candidate_count || 0} candidate records returned.`
      );
    } catch (error) {
      setMessage(`Investigation search failed: ${error.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function processVideo(videoId) {
    setMessage("");
    try {
      const data = await getJson(`/videos/${encodeURIComponent(videoId)}/process`, {
        method: "POST",
      });
      setMessage(`Processing started: ${data.job_id}`);
      await refresh();
    } catch (error) {
      setMessage(`Processing failed: ${error.message}`);
    }
  }

  async function uploadVideo(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    setMessage("Uploading video...");
    const form = new FormData();
    form.append("file", file);

    try {
      const data = await getJson("/videos/upload", {
        method: "POST",
        body: form,
      });
      setMessage(`Uploaded ${data.filename}. You can process it from the video database.`);
      await refresh();
    } catch (error) {
      setMessage(`Upload failed: ${error.message}`);
    } finally {
      event.target.value = "";
    }
  }

  async function generateReport() {
    setReporting(true);
    try {
      const investigation = {
        target_type: targetType,
        upper_body_color: upper || null,
        lower_body_color: lower || null,
        backpack: backpack || null,
        headwear: headwear || null,
        object_type: objectType || null,
        object_color: objectColor || null,
        brand: brand || null,
        distinctive_feature: feature || null,
        camera_id: camera || null,
        limit: Number(limit),
      };

      const response = await fetch(`${API}/report/pdf`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          investigation,
          candidates: searchResults,
          selected_videos: videos.filter((v) => selectedVideos.includes(v.video_id)),
        }),
      });

      if (!response.ok) throw new Error(await response.text());
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `visionforge-investigation-${Date.now()}.pdf`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      setMessage("Investigation PDF generated.");
    } catch (error) {
      setMessage(`PDF generation failed: ${error.message}`);
    } finally {
      setReporting(false);
    }
  }

  function resetSearch() {
    setUpper("");
    setLower("");
    setBackpack("");
    setHeadwear("");
    setCamera("");
    setObjectType("");
    setObjectColor("");
    setBrand("");
    setFeature("");
    setSearchResults([]);
    setSelected(null);
    setMessage("");
  }

  function toggleVideo(videoId) {
    setSelectedVideos((current) =>
      current.includes(videoId)
        ? current.filter((id) => id !== videoId)
        : [...current, videoId]
    );
  }

  const stats = health?.profile_statistics || {};

  return (
    <div className="vf-shell">
      <style>{`
        * { box-sizing: border-box; }
        body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background: #071018; color: #e8f1f7; }
        button, input, select { font: inherit; }
        .vf-shell { min-height: 100vh; background: radial-gradient(circle at top right, #123044 0, #071018 45%, #050a0e 100%); padding: 28px; }
        .vf-wrap { max-width: 1500px; margin: auto; }
        .hero { display:flex; justify-content:space-between; gap:24px; align-items:flex-start; margin-bottom:22px; }
        .eyebrow { color:#64d7ff; font-size:12px; letter-spacing:2px; font-weight:800; }
        h1 { margin:6px 0; font-size:34px; letter-spacing:-1px; }
        h2 { margin:0; font-size:18px; }
        .muted { color:#8da2af; font-size:13px; }
        .live { border:1px solid #1e819d; background:#0b2935; color:#78e5ff; padding:8px 12px; border-radius:999px; font-size:12px; font-weight:800; }
        .grid { display:grid; gap:16px; }
        .stats { grid-template-columns: repeat(5, minmax(0,1fr)); margin-bottom:16px; }
        .card { background:rgba(10,20,28,.88); border:1px solid #1b3441; border-radius:14px; padding:18px; box-shadow:0 10px 35px rgba(0,0,0,.18); }
        .stat-value { font-size:26px; font-weight:800; margin-top:6px; }
        .main { grid-template-columns: minmax(360px, 1fr) minmax(360px, 1fr); }
        .wide { grid-column:1 / -1; }
        .section-head { display:flex; justify-content:space-between; gap:12px; align-items:center; margin-bottom:15px; }
        .filters { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }
        label { display:block; color:#9eb2be; font-size:11px; text-transform:uppercase; letter-spacing:.8px; margin-bottom:6px; }
        input, select { width:100%; border:1px solid #27404d; background:#08141c; color:#eaf3f7; padding:10px 11px; border-radius:9px; outline:none; }
        input:focus, select:focus { border-color:#54c9e9; }
        .actions { display:flex; flex-wrap:wrap; gap:9px; margin-top:14px; }
        button { border:1px solid #2a4654; background:#0d202a; color:#e8f4f8; padding:10px 14px; border-radius:9px; cursor:pointer; font-weight:700; }
        button:hover { border-color:#53c9e8; }
        button.primary { background:#0b6e87; border-color:#1ca6c9; }
        button.ghost { background:transparent; }
        button:disabled { opacity:.5; cursor:not-allowed; }
        .tabs { display:flex; gap:8px; margin-bottom:15px; }
        .tab.active { background:#0d718b; border-color:#20b7db; }
        .video-list, .result-list { display:flex; flex-direction:column; gap:8px; max-height:420px; overflow:auto; }
        .row { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:11px; border:1px solid #1a323e; border-radius:10px; background:#08141b; }
        .row-main { min-width:0; }
        .row-title { font-weight:750; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .row-actions { display:flex; align-items:center; gap:7px; flex-shrink:0; }
        .badge { display:inline-block; border:1px solid #2a5263; color:#86dff5; background:#0a2029; padding:3px 7px; border-radius:999px; font-size:10px; margin-right:5px; }
        .result { cursor:pointer; }
        .result.selected { border-color:#35c2e7; background:#0b2029; }
        .result-top { display:flex; justify-content:space-between; gap:10px; }
        .score { font-size:17px; font-weight:850; color:#7be6ff; }
        .result-meta { display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; }
        .mini { color:#93a9b5; font-size:11px; }
        .detail { min-height:260px; }
        .detail-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
        .detail-box { border:1px solid #1b3542; border-radius:10px; padding:12px; background:#08141b; }
        .detail-label { color:#78909d; font-size:10px; text-transform:uppercase; letter-spacing:1px; }
        .detail-value { margin-top:4px; font-weight:750; word-break:break-word; }
        .crop { width:100%; max-height:330px; object-fit:contain; background:#050a0e; border-radius:10px; margin-bottom:12px; }
        .notice { margin:0 0 16px; padding:11px 14px; border:1px solid #2b4b58; background:#0a1b23; border-radius:10px; color:#a9c3cf; font-size:12px; }
        .job-ok { color:#7ce5a4; } .job-running { color:#79dfff; } .job-fail { color:#ff8d8d; }
        .empty { padding:25px; text-align:center; color:#78909d; border:1px dashed #24404d; border-radius:10px; }
        @media (max-width: 1000px) { .stats { grid-template-columns:repeat(2,1fr); } .main { grid-template-columns:1fr; } .filters { grid-template-columns:repeat(2,1fr); } }
        @media (max-width: 620px) { .vf-shell { padding:14px; } .hero { flex-direction:column; } .stats, .filters { grid-template-columns:1fr; } }
      `}</style>

      <div className="vf-wrap">
        <header className="hero">
          <div>
            <div className="eyebrow">AI-ASSISTED MULTI-CAMERA INVESTIGATION</div>
            <h1>VISIONFORGE AI</h1>
            <div className="muted">Evidence retrieval, cross-camera association and investigator review.</div>
          </div>
          <div className="live">● SYSTEM {health?.status === "healthy" ? "HEALTHY" : "OFFLINE"}</div>
        </header>

        <div className="notice">
          VISIONFORGE produces candidate evidence for human review. It does not automatically confirm a person's identity or determine guilt.
        </div>

        <section className="grid stats">
          <div className="card"><div className="muted">CAMERAS</div><div className="stat-value">{stats.camera_count ?? "—"}</div></div>
          <div className="card"><div className="muted">TRACK PROFILES</div><div className="stat-value">{stats.total_profiles ?? "—"}</div></div>
          <div className="card"><div className="muted">RE-ID READY</div><div className="stat-value">{stats.reid_ready ?? "—"}</div></div>
          <div className="card"><div className="muted">TRACK ONLY</div><div className="stat-value">{stats.track_only ?? "—"}</div></div>
          <div className="card"><div className="muted">CCTV VIDEOS</div><div className="stat-value">{videos.length}</div></div>
        </section>

        <div className="grid main">
          <section className="card">
            <div className="section-head">
              <div><h2>CCTV Video Database</h2><div className="muted">Upload new evidence or process an existing video.</div></div>
              <label style={{ margin:0 }}>
                <button className="primary" onClick={() => document.getElementById("vf-upload").click()}>Upload</button>
                <input id="vf-upload" type="file" accept="video/*" hidden onChange={uploadVideo} />
              </label>
            </div>
            <div className="video-list">
              {videos.length === 0 ? <div className="empty">No CCTV videos found.</div> : videos.map((video) => (
                <div className="row" key={video.video_id}>
                  <div className="row-main">
                    <div className="row-title">{video.filename}</div>
                    <div className="mini">{video.video_id} · {video.extension} · {(Number(video.size_bytes || 0) / 1048576).toFixed(1)} MB</div>
                  </div>
                  <div className="row-actions">
                    <input
                      type="checkbox"
                      checked={selectedVideos.includes(video.video_id)}
                      onChange={() => toggleVideo(video.video_id)}
                      title="Include in PDF report"
                      style={{ width:18 }}
                    />
                    <button onClick={() => processVideo(video.video_id)}>Process</button>
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section className="card">
            <div className="section-head">
              <div><h2>Processing Jobs</h2><div className="muted">Background processing status.</div></div>
              <button className="ghost" onClick={refresh}>Refresh</button>
            </div>
            <div className="video-list">
              {jobs.length === 0 ? <div className="empty">No processing jobs.</div> : jobs.slice(0,12).map((job) => (
                <div className="row" key={job.job_id}>
                  <div className="row-main">
                    <div className="row-title">{job.name}</div>
                    <div className="mini">{job.job_id}</div>
                  </div>
                  <strong className={job.status === "completed" ? "job-ok" : job.status === "failed" ? "job-fail" : "job-running"}>{job.status}</strong>
                </div>
              ))}
            </div>
          </section>

          <section className="card wide">
            <div className="section-head">
              <div><h2>Investigator Search</h2><div className="muted">Strict attribute retrieval over canonical investigation profiles.</div></div>
            </div>

            <div className="tabs">
              <button className={targetType === "person" ? "tab active" : "tab"} onClick={() => setTargetType("person")}>PERSON</button>
              <button className={targetType === "object" ? "tab active" : "tab"} onClick={() => setTargetType("object")}>OBJECT</button>
            </div>

            <div className="filters">
              {targetType === "person" ? (
                <>
                  <div><label>Upper Body</label><select value={upper} onChange={(e) => setUpper(e.target.value)}>{COLORS.map((c) => <option key={c} value={c}>{c || "Any"}</option>)}</select></div>
                  <div><label>Lower Body</label><select value={lower} onChange={(e) => setLower(e.target.value)}>{COLORS.map((c) => <option key={c} value={c}>{c || "Any"}</option>)}</select></div>
                  <div><label>Backpack</label><select value={backpack} onChange={(e) => setBackpack(e.target.value)}><option value="">Any</option><option value="yes">Yes</option><option value="no">No</option></select></div>
                  <div><label>Headwear</label><select value={headwear} onChange={(e) => setHeadwear(e.target.value)}><option value="">Any</option><option value="yes">Yes</option><option value="no">No</option></select></div>
                  <div><label>Camera</label><select value={camera} onChange={(e) => setCamera(e.target.value)}><option value="">All cameras</option>{cameraOptions.map((c) => <option key={c} value={c}>{c}</option>)}</select></div>
                </>
              ) : (
                <>
                  <div><label>Object Type</label><input value={objectType} onChange={(e) => setObjectType(e.target.value)} placeholder="backpack, suitcase..." /></div>
                  <div><label>Object Color</label><input value={objectColor} onChange={(e) => setObjectColor(e.target.value)} placeholder="if indexed" /></div>
                  <div><label>Brand</label><input value={brand} onChange={(e) => setBrand(e.target.value)} placeholder="if indexed" /></div>
                  <div><label>Distinctive Feature</label><input value={feature} onChange={(e) => setFeature(e.target.value)} placeholder="if indexed" /></div>
                </>
              )}
              <div><label>Limit</label><input type="number" min="1" max="500" value={limit} onChange={(e) => setLimit(e.target.value)} /></div>
            </div>

            <div className="actions">
              <button className="primary" disabled={busy} onClick={investigate}>{busy ? "Searching..." : "Start Investigation"}</button>
              <button onClick={resetSearch}>Reset</button>
              <button disabled={reporting} onClick={generateReport}>{reporting ? "Generating..." : "PDF Report"}</button>
              <button onClick={refresh}>Refresh Data</button>
            </div>

            {message && <div className="notice" style={{ marginTop:14, marginBottom:0 }}>{message}</div>}
          </section>

          <section className="card">
            <div className="section-head">
              <div><h2>Search Results</h2><div className="muted">{searchResults.length} returned records</div></div>
            </div>
            <div className="result-list">
              {searchResults.length === 0 ? <div className="empty">Run an investigation search.</div> : searchResults.map((item, index) => {
                const isObject = targetType === "object";
                return (
                  <div className={`row result ${selected === item ? "selected" : ""}`} key={`${item.camera_id || item.video_id || "object"}-${item.track_id ?? item.object_id ?? index}-${index}`} onClick={() => setSelected(item)}>
                    <div className="row-main">
                      <div className="result-top">
                        <strong>{item.camera_id || item.video_id || "Object index"}</strong>
                        <span className="score">{isObject ? (item.similarity == null ? "Indexed" : percent(item.similarity)) : percent(item.attribute_score)}</span>
                      </div>
                      <div className="result-meta">
                        {item.track_id != null && <CameraBadge camera={`TRACK ${item.track_id}`} />}
                        {item.object_id && <CameraBadge camera={item.object_id} />}
                        {item.upper_body_color && <CameraBadge camera={`Upper: ${item.upper_body_color}`} />}
                        {item.lower_body_color && <CameraBadge camera={`Lower: ${item.lower_body_color}`} />}
                        {item.class_name && <CameraBadge camera={item.class_name} />}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          <section className="card">
            <div className="section-head">
              <div><h2>Cross-Camera Associations</h2><div className="muted">Precomputed candidate links from the investigation engine.</div></div>
              <span className="badge">{crossCamera.length} loaded</span>
            </div>
            <div className="result-list">
              {crossCamera.length === 0 ? <div className="empty">No cross-camera candidate report is currently available.</div> : crossCamera.map((item, index) => (
                <div className={`row result ${selected === item ? "selected" : ""}`} key={`${item.source_camera}-${item.source_track_id}-${item.target_camera}-${item.target_track_id}-${index}`} onClick={() => setSelected(item)}>
                  <div className="row-main">
                    <div className="result-top">
                      <strong>{item.source_camera} / Track {item.source_track_id} → {item.target_camera} / Track {item.target_track_id}</strong>
                      <span className="score">{percent(item.appearance_similarity)}</span>
                    </div>
                    <div className="result-meta">
                      <CameraBadge camera={`Attributes ${percent(item.attribute_score)}`} />
                      <CameraBadge camera={`Temporal ${item.temporal_analysis?.valid ? "consistent" : "not confirmed"}`} />
                      <span className="mini">{item.identity_confirmed ? "Identity confirmed" : "Candidate evidence only"}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section className="card wide detail">
            <div className="section-head">
              <div><h2>Evidence Detail</h2><div className="muted">Selected record and available evidence.</div></div>
            </div>
            {!selected ? (
              <div className="empty">Select a search result or cross-camera association.</div>
            ) : (
              <div className="detail-grid">
                {selected.representative_crop && <div className="detail-box" style={{ gridColumn:"1 / -1" }}><img className="crop" src={cropUrl(selected.representative_crop)} alt="Representative evidence crop" /></div>}
                <div className="detail-box"><div className="detail-label">Source / Camera</div><div className="detail-value">{selected.source_camera || selected.camera_id || selected.video_id || "—"}</div></div>
                <div className="detail-box"><div className="detail-label">Track / Object</div><div className="detail-value">{selected.source_track_id ?? selected.track_id ?? selected.object_id ?? "—"}</div></div>
                {selected.target_camera && <div className="detail-box"><div className="detail-label">Target Camera / Track</div><div className="detail-value">{selected.target_camera} / {selected.target_track_id}</div></div>}
                <div className="detail-box"><div className="detail-label">Appearance Similarity</div><div className="detail-value">{percent(selected.appearance_similarity)}</div></div>
                <div className="detail-box"><div className="detail-label">Attribute Score</div><div className="detail-value">{percent(selected.attribute_score)}</div></div>
                <div className="detail-box"><div className="detail-label">First Seen</div><div className="detail-value">{formatSeconds(selected.first_seen_sec ?? selected.source_profile?.first_seen_sec)}</div></div>
                <div className="detail-box"><div className="detail-label">Last Seen</div><div className="detail-value">{formatSeconds(selected.last_seen_sec ?? selected.source_profile?.last_seen_sec)}</div></div>
                <div className="detail-box"><div className="detail-label">Temporal Analysis</div><div className="detail-value">{selected.temporal_analysis ? `${selected.temporal_analysis.valid ? "Consistent" : "Not consistent"} · gap ${formatSeconds(selected.temporal_analysis.time_gap_sec)}` : "—"}</div></div>
                <div className="detail-box"><div className="detail-label">Decision State</div><div className="detail-value">{selected.identity_confirmed ? "Confirmed by system" : "Candidate evidence only"}</div></div>
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
