<div className="candidate-list">

  {candidates.map((candidate, index) => {

    const cameraId = candidate.camera_id || "unknown";
    const trackId = candidate.track_id ?? "N/A";

    const upperColor =
      candidate.upper_body_color || "unknown";

    const lowerColor =
      candidate.lower_body_color || "unknown";

    const attributeScore =
      typeof candidate.attribute_score === "number"
        ? candidate.attribute_score
        : 0;

    const firstSeen =
      typeof candidate.first_seen_sec === "number"
        ? candidate.first_seen_sec
        : null;

    const lastSeen =
      typeof candidate.last_seen_sec === "number"
        ? candidate.last_seen_sec
        : null;

    const representativeCrop =
      candidate.representative_crop || null;

    return (
      <div
        className="candidate"
        key={`${cameraId}-${trackId}-${index}`}
        onClick={() => setSelected(candidate)}
      >

        <div className="candidate-number">
          {String(index + 1).padStart(2, "0")}
        </div>

        <div className="candidate-info">

          <div className="tracks">

            <span>
              {cameraId.toUpperCase()} · TRACK{" "}
              {String(trackId).padStart(3, "0")}
            </span>

            <span className="arrow">
              →
            </span>

            <span>
              {upperColor} / {lowerColor}
            </span>

          </div>

          <div className="evidence-count">

            Attribute{" "}
            {(attributeScore * 100).toFixed(0)}
            {"%"}

            {" · "}

            {firstSeen !== null && lastSeen !== null
              ? `${firstSeen.toFixed(2)}s → ${lastSeen.toFixed(2)}s`
              : "Timeline unavailable"}

          </div>

        </div>

        <div className="similarity">

          <strong>
            {(attributeScore * 100).toFixed(0)}%
          </strong>

          <span>
            ATTRIBUTE MATCH
          </span>

        </div>

      </div>
    );
  })}

</div>

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

        <label>CAMERA</label>

        <strong>
          {(selected.camera_id || "UNKNOWN").toUpperCase()}
        </strong>

      </div>

      <div className="detail-block">

        <label>TRACK</label>

        <strong>
          {selected.track_id !== undefined
            ? `Track ${String(selected.track_id).padStart(3, "0")}`
            : "Unavailable"}
        </strong>

      </div>

      <div className="detail-block">

        <label>PROFILE STATE</label>

        <strong>
          {selected.profile_state || "UNKNOWN"}
        </strong>

      </div>

      <div className="detail-block">

        <label>INVESTIGATION READY</label>

        <strong>
          {selected.investigation_ready
            ? "YES"
            : "NO"}
        </strong>

      </div>

      <div className="similarity-large">

        {typeof selected.attribute_score === "number"
          ? `${(selected.attribute_score * 100).toFixed(0)}%`
          : "N/A"}

        <span>
          Attribute match
        </span>

      </div>

      <div className="evidence-summary">

        <span>
          Upper body
        </span>

        <strong>
          {selected.upper_body_color || "Unknown"}
        </strong>

      </div>

      <div className="evidence-summary">

        <span>
          Lower body
        </span>

        <strong>
          {selected.lower_body_color || "Unknown"}
        </strong>

      </div>

      <div className="evidence-summary">

        <span>
          First seen
        </span>

        <strong>
          {typeof selected.first_seen_sec === "number"
            ? `${selected.first_seen_sec.toFixed(2)}s`
            : "N/A"}
        </strong>

      </div>

      <div className="evidence-summary">

        <span>
          Last seen
        </span>

        <strong>
          {typeof selected.last_seen_sec === "number"
            ? `${selected.last_seen_sec.toFixed(2)}s`
            : "N/A"}
        </strong>

      </div>

      <div className="evidence-summary">

        <span>
          Duration
        </span>

        <strong>
          {typeof selected.duration_sec === "number"
            ? `${selected.duration_sec.toFixed(2)}s`
            : "N/A"}
        </strong>

      </div>

      <div className="evidence-summary">

        <span>
          Representative crop
        </span>

        <strong>
          {selected.representative_crop
            ? "Available"
            : "Unavailable"}
        </strong>

      </div>

      <div className="evidence-summary">

        <span>
          Identity confirmed
        </span>

        <strong>
          {selected.identity_confirmed
            ? "YES"
            : "NO"}
        </strong>

      </div>

      {selected.representative_crop && (

        <div className="evidence-summary">

          <span>
            Evidence path
          </span>

          <strong
            style={{
              fontSize: "10px",
              wordBreak: "break-all",
              textAlign: "right",
              maxWidth: "60%"
            }}
          >
            {selected.representative_crop}
          </strong>

        </div>

      )}

      <p className="notice">
        AI-generated candidate association.
        Final identification requires human investigator review.
      </p>

    </div>

  ) : (

    <div className="empty-state">

      <div className="empty-icon">
        +
      </div>

      <h3>Select a candidate</h3>

      <p>
        Select a candidate to inspect
        attributes, timeline and evidence.
      </p>

    </div>

  )}

</aside>