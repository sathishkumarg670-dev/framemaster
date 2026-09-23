from __future__ import annotations
import json, time
from dataclasses import dataclass
import serial

@dataclass
class CameraNode:
    port: str
    baudrate: int = 115200
    node_id: str = "camera_node"

    def open(self):
        self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
        return self

    def send(self, command: dict):
        self.serial.write((json.dumps(command)+"\n").encode("utf-8"))

    def listen(self):
        while True:
            line=self.serial.readline().decode("utf-8","ignore").strip()
            if line:
                yield json.loads(line)
