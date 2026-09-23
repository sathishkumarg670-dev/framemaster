/*
VISIONFORGE AI - Arduino camera-node skeleton.
The Uno is a sensor/transport node. AI inference remains on the PC.

Serial protocol:
  {"type":"hello","node_id":"vf_cam_01"}
  {"type":"heartbeat","node_id":"vf_cam_01","ms":12345}
*/
const char* NODE_ID = "vf_cam_01";

void setup() {
  Serial.begin(115200);
  Serial.println("{\"type\":\"hello\",\"node_id\":\"vf_cam_01\"}");
}

void loop() {
  static unsigned long last = 0;
  if (millis() - last >= 2000) {
    last = millis();
    Serial.print("{\"type\":\"heartbeat\",\"node_id\":\"");
    Serial.print(NODE_ID);
    Serial.print("\",\"ms\":");
    Serial.print(millis());
    Serial.println("}");
  }
}
