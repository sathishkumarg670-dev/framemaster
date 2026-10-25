# 🔍 VISIONFORGE AI
### AI-Assisted Multi-Camera Investigation System

**Turning multiple camera feeds into one intelligent investigation.**

VISIONFORGE AI is a computer-vision prototype designed to assist video investigations by detecting and tracking people, comparing their visual appearances across multiple videos, organizing potential cross-camera associations, and generating investigation evidence and reports.

The project combines video processing, object detection, tracking, person re-identification (Re-ID), temporal analysis, a Python API, and a React-based investigation dashboard.

> **Project status:** Research and development prototype. Candidate associations require human review and must not be treated as confirmed identities.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Running the Application](#-running-the-application)
- [ESP32-CAM Recording](#-esp32-cam-recording)
- [Manual Video Processing](#-manual-video-processing)
- [Investigation V1](#-investigation-v1)
- [Testing](#-testing)
- [Current Limitations](#-current-limitations)
- [Future Improvements](#-future-improvements)
- [Privacy and Responsible Use](#-privacy-and-responsible-use)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

Modern surveillance environments may contain footage from multiple cameras, making it difficult and time-consuming to follow a person's movement between different views.

VISIONFORGE AI aims to simplify this process by providing a unified investigation workflow.

The system analyzes video footage, generates track profiles, compares visual appearance information, and presents potentially related tracks to an investigator. It also supports investigation data, camera topology, evidence associations, timeline information, and report generation.

## ❗ Problem Statement

- Large volumes of CCTV footage require time-consuming manual review.
- Tracking a person across different cameras is challenging.
- Relevant video segments and evidence can be overlooked.
- Investigators need a structured way to examine candidate matches and supporting information.

### 💡 Proposed Solution

Develop a modular, software-based investigation system that uses computer vision to assist with detection, tracking, appearance comparison, cross-video candidate ranking, and evidence organization.

## ✨ Key Features

- **Video processing:** Process video footage through the available computer-vision pipeline.
- **Object detection:** Identify supported objects and people in video frames.
- **Multi-object tracking:** Associate detections across frames using tracking IDs.
- **Person Re-ID:** Compare appearance features between track profiles.
- **Cross-video associations:** Generate and rank potentially related tracks across videos.
- **Attribute analysis:** Compare available visual attributes.
- **Temporal analysis:** Examine track timing and potential temporal consistency.
- **Camera topology:** Represent camera relationships to support investigation reasoning.
- **Investigation dashboard:** Present available video, tracking, and candidate information.
- **Evidence associations:** Link candidates to available supporting evidence and representative crops.
- **Investigation reports:** Generate structured investigation outputs and reports.
- **API backend:** Expose functionality through a FastAPI application.

Feature availability and completeness depend on the current implementation, configuration, and available video data.

## 🏗️ System Architecture

The intended processing workflow is:

```text
             CCTV / Video Files
                     |
                     v
             Video Processing
                     |
                     v
              Frame Extraction
                     |
                     v
               Object Detection
                     |
                     v
              Object Tracking
                     |
                     v
          Track Profiles & Appearance
                     |
                     v
           Person Re-Identification
                     |
                     v
       Cross-Video Candidate Generation
                     |
          +----------+----------+
          |                     |
          v                     v
   Temporal Analysis      Camera Topology
          |                     |
          +----------+----------+
                     |
                     v
          Investigation Results
                     |
          +----------+----------+
          |          |          |
          v          v          v
       Dashboard  Evidence    Reports
                  & Timeline
```

This diagram represents the intended logical workflow; not every stage is necessarily fully automated or production-ready.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Video processing | OpenCV |
| Object detection and tracking | Configured computer-vision models and tracking components |
| Person re-identification | Appearance-feature extraction and similarity comparison |
| Backend API | FastAPI |
| Frontend | React |
| Frontend development server | Vite |
| Data storage | SQLite and structured project data |
| Testing | pytest |
| Version control | Git and GitHub |

The exact model versions and dependencies should be confirmed against the project's current configuration files.

## 📁 Project Structure

The following is a simplified overview. Consult the repository for the complete structure.

```text
VISIONFORGE-AI/
├── src/
│   ├── api.py
│   ├── investigation_engine.py
│   └── finalize_investigation_v1.py
├── dashboard/
│   └── src/
├── config/
├── hardware/
├── tests/
├── data/
│   ├── videos/
│   ├── unified/
│   │   └── profiles/
│   └── investigation/
├── .gitignore
├── requirements.txt
└── README.md
```

**Important:** Some directories are created or populated during execution. The exact files available may vary by project version.

## 💻 Prerequisites

- Windows 11 or another compatible operating system.
- Python 3.12 or a compatible Python version supported by the dependencies.
- Node.js and npm.
- Git.
- Sufficient disk space for video files, model weights, and generated data.
- A CPU-compatible environment; an NVIDIA GPU is not assumed.

## ⚙️ Installation

### 1. Clone the repository

```powershell
git clone https://github.com/sathishkumarg670-dev/framemaster.git
cd framemaster
```

### 2. Create a Python virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the virtual environment's Python executable directly in subsequent commands.

### 3. Install Python dependencies

If `requirements.txt` is present and contains the project's current dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install any additional model weights or dependencies required by the project's configuration. Do not assume all pretrained models are included in the repository.

### 4. Install frontend dependencies

```powershell
cd dashboard
npm install
cd ..
```

## 🚀 Running the Application

Open separate terminals from the project root.

### Terminal 1 — Backend

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Backend address:

- API: `http://127.0.0.1:8000`
- API documentation: `http://127.0.0.1:8000/docs`
- Health endpoint, if available: `http://127.0.0.1:8000/health`

### Terminal 2 — Frontend

```powershell
cd dashboard
npm run dev
```

Open the local address printed by Vite, commonly:

`http://localhost:5173/`

The frontend and backend must both be running for the complete interface to work. Confirm that the frontend's API configuration points to the correct backend address.

## 📷 ESP32-CAM Recording

Hardware: AI-Thinker ESP32-CAM on an ESP32-CAM-MB USB shield. The board only captures and streams JPEG frames over Wi-Fi; recording, YOLO tracking and Re-ID run on the PC.

1. Flash `hardware/esp32cam_stream_v2/esp32cam_stream_v2.ino` (board "AI Thinker ESP32-CAM", PSRAM enabled) after entering your Wi-Fi name and password. Never commit real credentials. The older `esp32cam_stream_v1` sketch also works.
2. Check the boot log, if needed: `python hardware/esp32_serial_monitor_v1.py --port COM9`.
3. In the dashboard's **ESP32-CAM Recorder** panel, enter the stream URL (`http://<board-ip>/stream`) or just the serial port (`COM9`) to read the IP from the board, then click **Start Recording**.

Equivalent command line:

```powershell
.\.venv\Scripts\python.exe -m src.esp32_recorder --port COM9 --segment-seconds 60
```

How stored video is handled:

- The stream is saved as `data/videos/esp32_<node>_<YYYYMMDD_HHMMSS>.mp4` segments, at the measured camera fps so pipeline timestamps match real time. Segments are written to `data/videos/.recording/` first and moved into place only when complete.
- Each stored segment with motion is queued for processing (tracking, crops, attributes, Re-ID). Segments without motion are stored but not processed automatically; process them from the video list if needed.
- Jobs run one at a time, each in its own process, so a processing crash cannot stop the API or an active recording.
- The board serves one stream client at a time: while recording, close any browser tab showing the stream.
- A segment log is appended to `data/esp32/segments.jsonl`.

API: `GET /esp32/discover?port=COM9`, `POST /esp32/record/start`, `POST /esp32/record/{node_id}/stop`, `GET /esp32/status`.

## 🎞️ Manual Video Processing

The project supports video-upload and investigation workflows, but the exact command for running a raw video through every processing stage depends on the currently available processing scripts and their arguments.

For the Investigation V1 finalizer, run the following from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.finalize_investigation_v1 --limit 100
```

This command generates investigation outputs from existing track profiles. It does **not**, by itself, guarantee that a new raw video has been decoded, detected, tracked, and converted into profiles.

Check the actual processing scripts and their command-line help before selecting a manual raw-video processing command.

## 🔎 Investigation V1

The Investigation V1 finalizer generates structured candidate associations, chains, and timeline information from available profiles.

Run:

```powershell
.\.venv\Scripts\python.exe -m src.finalize_investigation_v1 --limit 100
```

The following output files were generated successfully in a previous development run:

```text
data/investigation/
├── appearance_candidates_v1.json
├── investigation_report_v1.json
├── investigation_v1.json
└── investigation_timeline.json
```

The exact candidate counts and results depend on the available videos and profiles.

**Interpretation note:** A candidate score indicates similarity or agreement under the configured analysis. It is not a verified probability of identity. Video-relative timestamps also should not be treated as synchronized real-world times unless synchronization has been established independently.

## 🧪 Testing

Run the Python test suite from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests
```

Run Python compilation checks:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src tests
```

Build the frontend:

```powershell
cd dashboard
npm run build
```

These checks validate different aspects of the application. Passing tests does not, by itself, establish the accuracy of the computer-vision results.

## ⚠️ Current Limitations

VISIONFORGE AI is a development prototype, not a validated production surveillance platform.

- Detection and tracking performance depend on footage quality, lighting, camera angles, occlusion, and model configuration.
- Appearance similarity can produce false matches and missed matches.
- Camera transition reasoning depends on correct topology and timing information.
- Cross-camera timestamps may not represent synchronized wall-clock time.
- The accuracy and completeness of generated evidence depend on the underlying pipeline.
- Arbitrary-video processing, robustness, and error recovery require validation.
- Production security, authentication, deployment, and operational monitoring require further work.

## 🗺️ Future Improvements

- Improve end-to-end processing of newly uploaded videos.
- Strengthen cross-camera temporal reasoning and topology validation.
- Improve Re-ID ranking and threshold calibration.
- Expand visual attribute analysis.
- Improve investigation timelines and evidence visualization.
- Add robust processing-job status, cancellation, and error recovery.
- Expand automated tests and performance benchmarks.
- Improve configuration, documentation, and deployment.
- Evaluate accuracy on representative, appropriately authorized datasets.

## 🌱 Privacy and Responsible Use

VISIONFORGE AI should be used only with appropriate authorization and in accordance with applicable privacy, data-protection, and surveillance laws.

- Treat video footage and extracted appearance data as sensitive.
- Limit access to authorized users.
- Protect stored videos, crops, profiles, and reports.
- Apply appropriate retention and deletion policies.
- Review candidate associations manually before drawing conclusions.
- Do not use a similarity score as sole evidence of a person's identity.

**AI-generated associations are investigative leads, not confirmed identities.**

## 🤝 Contributing

Contributions that improve reliability, maintainability, documentation, testing, or investigation usability are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make a focused change.
4. Run the relevant tests and frontend build.
5. Submit a pull request describing the change and its validation.

Please avoid committing large video files, private footage, model weights, credentials, or generated evidence unless explicitly intended and appropriately authorized.

## 📄 License

No license has been specified here. Add a `LICENSE` file before declaring the terms under which others may use, modify, or distribute this project.

---

**VISIONFORGE AI — From video footage to structured investigation insights.**
