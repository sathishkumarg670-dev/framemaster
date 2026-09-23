import { useEffect, useState } from "react";

const API = "http://127.0.0.1:8000";

function Dashboard() {
  const [candidates, setCandidates] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searching, setSearching] = useState(false);
  const [selected, setSelected] = useState(null);

  const [targetType, setTargetType] = useState("person");

  const [shirtColor, setShirtColor] = useState("");
  const [pantsColor, setPantsColor] = useState("");
  const [backpack, setBackpack] = useState("");
  const [headwear, setHeadwear] = useState("");

  const [objectType, setObjectType] = useState("");
  const [objectColor, setObjectColor] = useState("");
  const [brand, setBrand] = useState("");
  const [distinctiveFeature, setDistinctiveFeature] = useState("");

  const [investigationStarted, setInvestigationStarted] = useState(false);
  const [searchInfo, setSearchInfo] = useState(null);

  // Video database
  const [videos, setVideos] = useState([]);
  const [videosLoading, setVideosLoading] = useState(true);
  const [selectedVideos, setSelectedVideos] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [videoMessage, setVideoMessage] = useState("");
  const [generatingReport, setGeneratingReport] = useState(false);
  const [reportMessage, setReportMessage] = useState("");

  useEffect(() => {
    loadDefaultCandidates();
    loadVideos();
  }, []);

  // ------------------------------------------------------
  // Load candidates
  // ------------------------------------------------------

  async function loadDefaultCandidates() {
    try {
      const response = await fetch(`${API}/candidates?limit=20`);

      if (!response.ok) {
        throw new Error(`Candidate request failed: ${response.status}`);
      }

      const data = await response.json();
      setCandidates(data.candidates || []);
    } catch (error) {
      console.error("API connection failed:", error);
    } finally {
      setLoading(false);
    }
  }

  // ------------------------------------------------------
  // Load CCTV video database
  // ------------------------------------------------------

  async function loadVideos() {
    setVideosLoading(true);
    setVideoMessage("");

    try {
      const response = await fetch(`${API}/videos`);

      if (!response.ok) {
        throw new Error(`Video request failed: ${response.status}`);
      }

      const data = await response.json();
      setVideos(data.videos || []);
    } catch (error) {
      console.error("Video database request failed:", error);
      setVideoMessage("Unable to connect to CCTV video database.");
    } finally {
      setVideosLoading(false);
    }
  }

  // ------------------------------------------------------
  // Select / deselect video
  // ------------------------------------------------------

  function toggleVideo(videoId) {
    setSelectedVideos((current) =>
      current.includes(videoId)
        ? current.filter((id) => id !== videoId)
        : [...current, videoId]
    );
  }

  // ------------------------------------------------------
  // Upload CCTV video
  // ------------------------------------------------------

  async function uploadVideo(event) {
    const file = event.target.files?.[0];

    if (!file) {
      return;
    }

    setUploading(true);
    setVideoMessage("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API}/videos/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `Upload failed: ${response.status}`);
      }

      const data = await response.json();

      setVideoMessage(
        `Uploaded successfully: ${data.filename || file.name}`
      );

      await loadVideos();
    } catch (error) {
      console.error("Video upload failed:", error);
      setVideoMessage("Video upload failed. Check the backend terminal.");
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  }

  // ------------------------------------------------------
  // Generate PDF investigation report
  // ------------------------------------------------------

  async function generateReport() {
    setGeneratingReport(true);
    setReportMessage("");

    const investigation = {
      target_type: targetType,
      shirt_color: shirtColor || null,
      pants_color: pantsColor || null,
      backpack: backpack || null,
      headwear: headwear || null,
      object_type: objectType || null,
      object_color: objectColor || null,
      brand: brand || null,
      distinctive_feature: distinctiveFeature || null,
    };

    const selectedVideoData = videos.filter((video) =>
      selectedVideos.includes(video.video_id)
    );

    try {
      const response = await fetch(`${API}/report/pdf`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          investigation,
          candidates,
          selected_videos: selectedVideoData,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
          errorText || `Report generation failed: ${response.status}`
        );
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");

      link.href = url;
      link.download = `visionforge_investigation_${new Date()
        .toISOString()
        .slice(0, 19)
        .replace(/[:T]/g, "-")}.pdf`;

      document.body.appendChild(link);
      link.click();
      link.remove();

      window.URL.revokeObjectURL(url);
      setReportMessage("PDF report generated successfully.");
    } catch (error) {
      console.error("PDF report generation failed:", error);
      setReportMessage(
        "Unable to generate PDF report. Check that the FastAPI backend is running."
      );
    } finally {
      setGeneratingReport(false);
    }
  }

  // ------------------------------------------------------
  // Start investigation
  // ------------------------------------------------------

  async function startInvestigation() {
    setSearching(true);
    setSelected(null);

    const query = {
      target_type: targetType,
      shirt_color: shirtColor || null,
      pants_color: pantsColor || null,
      backpack: backpack || null,
      headwear: headwear || null,
      object_type: objectType || null,
      object_color: objectColor || null,
      brand: brand || null,
      distinctive_feature: distinctiveFeature || null,
    };

    try {
      const response = await fetch(`${API}/investigate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(query),
      });

      if (!response.ok) {
        throw new Error(`Investigation failed: ${response.status}`);
      }

      const data = await response.json();

      setCandidates(data.candidates || []);

      setSearchInfo({
        matchingTracks: data.matching_tracks,
        candidateCount: data.candidate_count,
      });

      setInvestigationStarted(true);
    } catch (error) {
      console.error("Investigation request failed:", error);

      setCandidates([]);

      setSearchInfo({
        matchingTracks: 0,
        candidateCount: 0,
      });
    } finally {
      setSearching(false);
    }
  }

  // ------------------------------------------------------
  // Reset investigation
  // ------------------------------------------------------

  function resetInvestigation() {
    setTargetType("person");

    setShirtColor("");
    setPantsColor("");
    setBackpack("");
    setHeadwear("");

    setObjectType("");
    setObjectColor("");
    setBrand("");
    setDistinctiveFeature("");

    setInvestigationStarted(false);
    setSearchInfo(null);
    setSelected(null);

    setLoading(true);
    loadDefaultCandidates();
  }

  // ------------------------------------------------------
  // UI
  // ------------------------------------------------------

  return (
    <div className="app">

      {/* HEADER */}
      <header className="topbar">
        <div>
          <h1>VISIONFORGE AI</h1>
          <p>AI-Assisted Multi-Camera Investigation System</p>
        </div>

        <div className="system-status">
          <span className="status-dot"></span>
          SYSTEM ONLINE
        </div>
      </header>

      <main className="dashboard">

        {/* ==================================================
            CCTV VIDEO SOURCES
        ================================================== */}

        <section className="panel investigation-panel">

          <div className="panel-header">
            <div>
              <h2>CCTV Video Sources</h2>
              <p>
                Select footage from the investigation video database
              </p>
            </div>

            <span className="live-badge">
              {videosLoading ? "LOADING" : `${videos.length} SOURCES`}
            </span>
          </div>

          <div className="investigation-body">

            {/* DATABASE ACTIONS */}

            <div
              style={{
                display: "flex",
                gap: "12px",
                flexWrap: "wrap",
                marginBottom: "20px",
              }}
            >
              <label
                className="search-button"
                style={{
                  cursor: uploading ? "wait" : "pointer",
                  display: "inline-flex",
                  alignItems: "center",
                }}
              >
                {uploading ? "⏳ UPLOADING..." : "⬆️ UPLOAD CCTV VIDEO"}

                <input
                  type="file"
                  accept=".mp4,.avi,.mov,.mkv,.webm,.m4v,video/*"
                  onChange={uploadVideo}
                  disabled={uploading}
                  style={{ display: "none" }}
                />
              </label>

              <button
                className="reset-button"
                onClick={loadVideos}
                disabled={videosLoading}
              >
                🔄 REFRESH DATABASE
              </button>
            </div>

            {/* MESSAGE */}

            {videoMessage && (
              <div
                style={{
                  padding: "12px 14px",
                  marginBottom: "16px",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: "8px",
                  fontSize: "13px",
                }}
              >
                {videoMessage}
              </div>
            )}

            {/* VIDEO LIST */}

            {videosLoading ? (
              <div className="loading">
                Loading CCTV video database...
              </div>
            ) : videos.length === 0 ? (
              <div className="loading">
                No CCTV videos available.
              </div>
            ) : (
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns:
                    "repeat(auto-fit, minmax(260px, 1fr))",
                  gap: "12px",
                }}
              >
                {videos.map((video) => {
                  const isSelected = selectedVideos.includes(
                    video.video_id
                  );

                  return (
                    <div
                      key={video.video_id}
                      onClick={() => toggleVideo(video.video_id)}
                      style={{
                        cursor: "pointer",
                        padding: "16px",
                        borderRadius: "10px",
                        border: isSelected
                          ? "1px solid #00ff9d"
                          : "1px solid rgba(255,255,255,0.1)",
                        background: isSelected
                          ? "rgba(0,255,157,0.08)"
                          : "rgba(255,255,255,0.025)",
                        transition: "0.2s",
                      }}
                    >
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          gap: "10px",
                          marginBottom: "10px",
                        }}
                      >
                        <strong>
                          {isSelected ? "☑" : "☐"}{" "}
                          {video.filename}
                        </strong>

                        <span
                          style={{
                            fontSize: "11px",
                            opacity: 0.65,
                          }}
                        >
                          {video.extension}
                        </span>
                      </div>

                      <div
                        style={{
                          fontSize: "12px",
                          opacity: 0.7,
                          lineHeight: "1.7",
                        }}
                      >
                        <div>
                          VIDEO ID: {video.video_id}
                        </div>

                        <div>
                          SIZE:{" "}
                          {video.size_mb
                            ? `${video.size_mb} MB`
                            : `${(
                                (video.size_bytes || 0) /
                                1024 /
                                1024
                              ).toFixed(2)} MB`}
                        </div>

                        <div>
                          SOURCE: CCTV DATABASE
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* SELECTED SOURCE SUMMARY */}

            <div
              style={{
                marginTop: "18px",
                paddingTop: "16px",
                borderTop: "1px solid rgba(255,255,255,0.08)",
                fontSize: "13px",
              }}
            >
              <strong>
                {selectedVideos.length}
              </strong>{" "}
              video source{selectedVideos.length !== 1 ? "s" : ""} selected
            </div>

            <div
              style={{
                marginTop: "8px",
                fontSize: "12px",
                opacity: 0.55,
              }}
            >
              Current footage is development/test data. New database
              videos can be added dynamically.
            </div>
          </div>
        </section>

        {/* ==================================================
            NEW INVESTIGATION
        ================================================== */}

        <section className="panel investigation-panel">

          <div className="panel-header">
            <div>
              <h2>New Investigation</h2>
              <p>
                Define the person or object to search across CCTV footage
              </p>
            </div>

            <span className="live-badge">
              {searching
                ? "SEARCHING"
                : investigationStarted
                  ? "ACTIVE"
                  : "READY"}
            </span>
          </div>

          <div className="investigation-body">

            {/* TARGET */}

            <div className="target-selector">
              <label>SEARCH TARGET</label>

              <div className="target-buttons">

                <button
                  className={
                    targetType === "person"
                      ? "target-button selected"
                      : "target-button"
                  }
                  onClick={() => setTargetType("person")}
                >
                  👤 PERSON
                </button>

                <button
                  className={
                    targetType === "object"
                      ? "target-button selected"
                      : "target-button"
                  }
                  onClick={() => setTargetType("object")}
                >
                  📦 OBJECT
                </button>

              </div>
            </div>

            {/* PERSON ATTRIBUTES */}

            {targetType === "person" ? (
              <div className="attribute-grid">

                <div className="attribute-field">
                  <label>SHIRT / UPPER COLOR</label>

                  <select
                    value={shirtColor}
                    onChange={(e) => setShirtColor(e.target.value)}
                  >
                    <option value="">Any</option>
                    <option value="black">Black</option>
                    <option value="white">White</option>
                    <option value="red">Red</option>
                    <option value="blue">Blue</option>
                    <option value="green">Green</option>
                    <option value="yellow">Yellow</option>
                    <option value="gray">Gray</option>
                    <option value="dark_gray">Dark Gray</option>
                    <option value="brown">Brown</option>
                  </select>
                </div>

                <div className="attribute-field">
                  <label>PANTS / LOWER COLOR</label>

                  <select
                    value={pantsColor}
                    onChange={(e) => setPantsColor(e.target.value)}
                  >
                    <option value="">Any</option>
                    <option value="black">Black</option>
                    <option value="white">White</option>
                    <option value="blue">Blue</option>
                    <option value="gray">Gray</option>
                    <option value="dark_gray">Dark Gray</option>
                    <option value="brown">Brown</option>
                    <option value="green">Green</option>
                    <option value="beige">Beige</option>
                  </select>
                </div>

                <div className="attribute-field">
                  <label>BACKPACK</label>

                  <select
                    value={backpack}
                    onChange={(e) => setBackpack(e.target.value)}
                  >
                    <option value="">Any</option>
                    <option value="yes">Yes</option>
                    <option value="no">No</option>
                  </select>
                </div>

                <div className="attribute-field">
                  <label>HEADWEAR</label>

                  <select
                    value={headwear}
                    onChange={(e) => setHeadwear(e.target.value)}
                  >
                    <option value="">Any</option>
                    <option value="cap">Cap</option>
                    <option value="hat">Hat</option>
                    <option value="helmet">Helmet</option>
                    <option value="none">None</option>
                  </select>
                </div>

              </div>
            ) : (
              /* OBJECT ATTRIBUTES */

              <div className="attribute-grid">

                <div className="attribute-field">
                  <label>OBJECT TYPE</label>

                  <input
                    type="text"
                    placeholder="e.g. Backpack, Laptop, Phone"
                    value={objectType}
                    onChange={(e) => setObjectType(e.target.value)}
                  />
                </div>

                <div className="attribute-field">
                  <label>OBJECT COLOR</label>

                  <input
                    type="text"
                    placeholder="e.g. Black, Red, Blue"
                    value={objectColor}
                    onChange={(e) => setObjectColor(e.target.value)}
                  />
                </div>

                <div className="attribute-field">
                  <label>BRAND / LOGO</label>

                  <input
                    type="text"
                    placeholder="Optional"
                    value={brand}
                    onChange={(e) => setBrand(e.target.value)}
                  />
                </div>

                <div className="attribute-field">
                  <label>DISTINCTIVE FEATURE</label>

                  <input
                    type="text"
                    placeholder="Describe visible feature"
                    value={distinctiveFeature}
                    onChange={(e) =>
                      setDistinctiveFeature(e.target.value)
                    }
                  />
                </div>

              </div>
            )}

            {/* ACTIONS */}

            <div className="investigation-actions">

              <button
                className="search-button"
                onClick={startInvestigation}
                disabled={searching}
              >
                {searching
                  ? "⏳ SEARCHING..."
                  : "🔍 START INVESTIGATION"}
              </button>

              <button
                className="reset-button"
                onClick={resetInvestigation}
              >
                RESET
              </button>

              <button
                className="search-button"
                onClick={generateReport}
                disabled={generatingReport || !investigationStarted}
                title={
                  !investigationStarted
                    ? "Run an investigation before generating a report."
                    : "Generate and download the investigation PDF."
                }
              >
                {generatingReport
                  ? "⏳ GENERATING PDF..."
                  : "📄 GENERATE PDF REPORT"}
              </button>

            </div>

            {reportMessage && (
              <div
                style={{
                  marginTop: "12px",
                  padding: "12px 14px",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: "8px",
                  fontSize: "13px",
                }}
              >
                {reportMessage}
              </div>
            )}

            {/* SEARCH STATUS */}

            {investigationStarted && searchInfo && (
              <div className="active-query">

                <span>SEARCH COMPLETE</span>

                <strong>
                  {searchInfo.matchingTracks}
                  {" matching tracks · "}
                  {searchInfo.candidateCount}
                  {" candidate associations"}
                </strong>

              </div>
            )}

          </div>
        </section>

        {/* ==================================================
            STATISTICS
        ================================================== */}

        <section className="stats">

          <div className="stat-card">
            <span>VIDEO SOURCES</span>
            <strong>{videos.length}</strong>
          </div>

          <div className="stat-card">
            <span>SELECTED</span>
            <strong>{selectedVideos.length}</strong>
          </div>

          <div className="stat-card">
            <span>TRACKS</span>
            <strong>88</strong>
          </div>

          <div className="stat-card">
            <span>MATCHES</span>
            <strong>1,927</strong>
          </div>

        </section>

        {/* ==================================================
            RESULTS
        ================================================== */}

        <section className="content-grid">

          <div className="panel candidates-panel">

            <div className="panel-header">

              <div>
                <h2>Cross-Camera Candidates</h2>

                <p>
                  {investigationStarted
                    ? "Filtered using investigation attributes"
                    : "AI-generated candidate associations"}
                </p>
              </div>

              <span className="live-badge">LIVE</span>

            </div>

            {loading ? (
              <div className="loading">
                Loading investigation data...
              </div>
            ) : candidates.length === 0 ? (
              <div className="loading">
                No matching candidates found.
              </div>
            ) : (
              <div className="candidate-list">

                {candidates.map((candidate, index) => (

                  <div
                    className="candidate"
                    key={`${candidate.camera_01_track}-${candidate.camera_02_track}-${index}`}
                    onClick={() => setSelected(candidate)}
                  >

                    <div className="candidate-number">
                      {String(index + 1).padStart(2, "0")}
                    </div>

                    <div className="candidate-info">

                      <div className="tracks">

                        <span>
                          CAM 01 · TRACK{" "}
                          {String(candidate.camera_01_track).padStart(
                            3,
                            "0"
                          )}
                        </span>

                        <span className="arrow">→</span>

                        <span>
                          CAM 02 · TRACK{" "}
                          {String(candidate.camera_02_track).padStart(
                            3,
                            "0"
                          )}
                        </span>

                      </div>

                      <div className="evidence-count">

                        {candidate.evidence?.length || 0}
                        {" evidence frames"}

                        {candidate.attribute_match !== undefined && (
                          <>
                            {" · Attribute "}
                            {(candidate.attribute_match * 100).toFixed(0)}
                            {"%"}
                          </>
                        )}

                      </div>

                    </div>

                    <div className="similarity">

                      <strong>
                        {(candidate.similarity * 100).toFixed(1)}%
                      </strong>

                      <span>RE-ID SIMILARITY</span>

                    </div>

                  </div>

                ))}

              </div>
            )}

          </div>

          {/* DETAILS */}

          <aside className="panel details-panel">

            <div className="panel-header">

              <div>
                <h2>Investigation Detail</h2>
                <p>Selected candidate evidence</p>
              </div>

            </div>

            {selected ? (

              <div className="details">

                <div className="detail-block">

                  <label>CAMERA 01</label>

                  <strong>
                    Track{" "}
                    {String(selected.camera_01_track).padStart(3, "0")}
                  </strong>

                </div>

                <div className="detail-arrow">↓</div>

                <div className="detail-block">

                  <label>CAMERA 02</label>

                  <strong>
                    Track{" "}
                    {String(selected.camera_02_track).padStart(3, "0")}
                  </strong>

                </div>

                <div className="similarity-large">

                  {(selected.similarity * 100).toFixed(1)}%

                  <span>Re-ID candidate similarity</span>

                </div>

                {selected.attribute_match !== undefined && (
                  <div className="evidence-summary">

                    <span>Attribute match</span>

                    <strong>
                      {(selected.attribute_match * 100).toFixed(0)}%
                    </strong>

                  </div>
                )}

                <div className="evidence-summary">

                  <span>Evidence frames</span>

                  <strong>
                    {selected.evidence?.length || 0}
                  </strong>

                </div>

                <p className="notice">
                  AI-generated candidate association.
                  Final identification requires human investigator review.
                </p>

              </div>

            ) : (

              <div className="empty-state">

                <div className="empty-icon">⌖</div>

                <h3>Select a candidate</h3>

                <p>
                  Select a cross-camera candidate to inspect
                  similarity and evidence information.
                </p>

              </div>

            )}

          </aside>

        </section>

        {/* ==================================================
            PIPELINE
        ================================================== */}

        <section className="panel timeline-panel">

          <div className="panel-header">

            <div>
              <h2>Investigation Pipeline</h2>
              <p>VISIONFORGE processing stages</p>
            </div>

          </div>

          <div className="pipeline">

            <div className="pipeline-step done">
              <span>01</span>
              Video Input
            </div>

            <div className="pipeline-line"></div>

            <div className="pipeline-step done">
              <span>02</span>
              Detection
            </div>

            <div className="pipeline-line"></div>

            <div className="pipeline-step done">
              <span>03</span>
              Tracking
            </div>

            <div className="pipeline-line"></div>

            <div className="pipeline-step done">
              <span>04</span>
              Re-ID
            </div>

            <div className="pipeline-line"></div>

            <div className="pipeline-step active">
              <span>05</span>
              Cross-Camera
            </div>

            <div className="pipeline-line"></div>

            <div className="pipeline-step active">
              <span>06</span>
              Evidence
            </div>

          </div>

        </section>

      </main>

      <footer>
        VISIONFORGE AI · Investigation Intelligence Platform
      </footer>

    </div>
  );
}

export default Dashboard;
