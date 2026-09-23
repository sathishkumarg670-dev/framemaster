# VISIONFORGE AI — One-Shot Completion Patch

This patch unifies the investigation product around the canonical profile/search architecture.

Included:
- dynamic supported-video discovery
- canonical TrackProfile readiness
- unified investigation search
- FastAPI integration
- unique video IDs
- durable SQLite investigation/audit store
- topology engine/config
- background job manager
- evidence clip extractor
- separate object-index pipeline
- React dashboard source
- Arduino/OV7670 node protocol skeleton
- smoke tests

Important:
- The current person attributes are still the attributes actually supported by the installed model/pipeline; no attribute is fabricated.
- Object retrieval uses a separate generic visual embedding path and is explicitly not an identity classifier.
- Camera topology is configurable but must be populated from measured camera relationships.
- The Arduino sketch is a transport/heartbeat node; the Uno does not run YOLO/Re-ID.
- Hardware requires physical wiring and testing.
