import json, tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.models import TrackProfile, TrackAttributes
from src.topology import validate_transition

def test_profile_readiness():
    p=TrackProfile(
        video_id="camera_test", track_id=1,
        first_seen_frame=0,last_seen_frame=10,
        first_seen_sec=0,last_seen_sec=1,duration_sec=1,
        total_observations=10,total_crops=2,
        attributes=TrackAttributes(track_id=1,upper_body_color="black",lower_body_color="gray",sample_crops=2),
        embedding=[1.0,0.0]
    )
    p.update_readiness()
    assert p.investigation_ready
    assert p.profile_state=="REID_READY"

def test_topology():
    cfg={"transitions":[{"source":"camera_01","target":"camera_02","min_gap_seconds":5,"max_gap_seconds":20}]}
    assert validate_transition(cfg,"camera_01","camera_02",10) is True
    assert validate_transition(cfg,"camera_01","camera_02",30) is False
