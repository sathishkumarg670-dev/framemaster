from __future__ import annotations
import json, re, time
from dataclasses import dataclass
import serial

@dataclass
class CameraNode:
    port: str
    baudrate: int = 115200
    node_id: str = "camera_node"

    def open(self):
        # On the ESP32-CAM-MB shield DTR/RTS drive IO0/EN. Force both low before
        # opening so the chip is neither held in reset nor put in download mode.
        self.serial = serial.Serial()
        self.serial.port = self.port
        self.serial.baudrate = self.baudrate
        self.serial.timeout = 1
        self.serial.dtr = False
        self.serial.rts = False
        self.serial.open()
        self.serial.dtr = False
        self.serial.rts = False
        return self

    def close(self):
        self.serial.close()

    def send(self, command: dict):
        self.serial.write((json.dumps(command)+"\n").encode("utf-8"))

    def listen(self):
        while True:
            line=self.serial.readline().decode("utf-8","ignore").strip()
            if line.startswith("{"):
                try: yield json.loads(line)
                except json.JSONDecodeError: continue   # boot ROM noise / partial line


BROWSER_LINE = re.compile(r"http://(\d{1,3}(?:\.\d{1,3}){3})")


def _parse_node_line(line: str) -> dict | None:
    """Return node info from one serial line, or None.

    Understands both firmware styles in hardware/:
      esp32cam_stream_v2 JSON  {"type":"heartbeat","node_id":...,"ip":...,"stream":...}
      esp32cam_stream_v1 text  "OPEN THIS IN YOUR BROWSER:  http://192.168.1.50"
    """
    if line.startswith("{"):
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            return None
        if message.get("type") == "error":
            raise RuntimeError(f"ESP32 reported: {message.get('message')}")
        ip = message.get("ip")
        if not ip or ip == "0.0.0.0":
            return None
        return {
            "node_id": message.get("node_id", "esp32cam"),
            "ip": ip,
            "stream_url": message.get("stream") or f"http://{ip}/stream",
        }
    if "CAMERA INIT FAILED" in line:
        raise RuntimeError("ESP32 reported camera init failed - check the ribbon cable and 5V supply")
    match = BROWSER_LINE.search(line)
    if match and match.group(1) != "0.0.0.0":
        ip = match.group(1)
        return {"node_id": "esp32cam", "ip": ip, "stream_url": f"http://{ip}/stream"}
    return None


def discover_esp32(port: str = "COM9", baudrate: int = 115200, listen: float = 4.0, boot_timeout: float = 30.0) -> dict:
    """Read the node's IP from serial and return its stream URL.

    v2 firmware repeats its IP in a heartbeat every 2 s, so a short listen is
    enough. v1 firmware prints it once at boot, so if nothing is heard the board
    is reset (EN pulse via RTS, as in hardware/esp32_serial_monitor_v1.py) and
    the boot log is read until Wi-Fi connects.
    """
    node = CameraNode(port, baudrate).open()
    try:
        for phase in ("listen", "reset"):
            if phase == "reset":
                node.serial.reset_input_buffer()
                node.serial.rts = True     # EN low  -> chip held in reset
                time.sleep(0.12)
                node.serial.rts = False    # EN high -> chip boots and runs the sketch
            deadline = time.time() + (listen if phase == "listen" else boot_timeout)
            while time.time() < deadline:
                line = node.serial.readline().decode("utf-8", "ignore").strip()
                found = _parse_node_line(line) if line else None
                if found:
                    return {**found, "port": port}
    finally:
        node.close()
    raise TimeoutError(
        f"No IP address seen on {port}. Check Wi-Fi credentials in the sketch, "
        f"or run hardware/esp32_serial_monitor_v1.py --port {port} to see the boot log."
    )
