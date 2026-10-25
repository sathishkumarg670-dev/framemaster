#!/usr/bin/env python3
"""
esp32_serial_monitor_v1.py  -  minimal ESP32-CAM serial monitor / boot tester

Why not the Arduino IDE Serial Monitor?
  IDE 2.x sometimes opens the port with DTR/RTS asserted. On an ESP32-CAM-MB
  those lines drive IO0 and EN, so the board can sit in download mode (silent)
  or in permanent reset (silent). This script forces both lines LOW before
  opening, then pulses EN to reset the chip, so you always see the boot ROM
  banner if the chip is alive at all.

Usage (Windows CMD / PowerShell, in the folder containing this file):
    python esp32_serial_monitor_v1.py                 -> COM9 @ 115200, resets board
    python esp32_serial_monitor_v1.py --list          -> list all COM ports + VID/PID
    python esp32_serial_monitor_v1.py --port COM5     -> different port
    python esp32_serial_monitor_v1.py --no-reset      -> just listen, don't reset
Press Ctrl+C to quit.
"""

import argparse
import sys
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit("pyserial is missing.  Run:  python -m pip install pyserial")

DEFAULT_PORT = "COM9"
DEFAULT_BAUD = 115200


def list_all_ports() -> None:
    ports = list(list_ports.comports())
    if not ports:
        print("No COM ports found. The USB adapter is not enumerating at all.")
        print("-> check the cable (must be a DATA cable, not charge-only)")
        print("-> check Device Manager for a yellow '!' under Other devices")
        return
    print(f"{len(ports)} port(s) found:\n")
    for p in ports:
        vidpid = f"{p.vid:04X}:{p.pid:04X}" if p.vid is not None else "----:----"
        print(f"  {p.device:<8} VID:PID {vidpid}  {p.description}")
    print("\n  0403:6001 = FTDI FT232     -> FTDI VCP driver")
    print("  1A86:7523 = CH340G/CH340C   -> CH340 driver (most ESP32-CAM-MB shields)")
    print("  10C4:EA60 = CP2102          -> Silicon Labs VCP driver")


def monitor(port: str, baud: int, do_reset: bool) -> None:
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.timeout = 0.2
    # Force handshake lines low BEFORE opening so the ESP32 is not held
    # in reset (EN) or in bootloader mode (IO0).
    ser.dtr = False
    ser.rts = False

    try:
        ser.open()
    except serial.SerialException as exc:
        print(f"Could not open {port}: {exc}")
        print("-> the Arduino IDE Serial Monitor may still own the port; close it.")
        print("-> run with --list to see which ports actually exist.")
        return

    ser.dtr = False
    ser.rts = False
    time.sleep(0.1)
    ser.reset_input_buffer()

    if do_reset:
        print(f"[monitor] pulsing EN to reset the board on {port}...")
        ser.rts = True      # EN low  -> chip held in reset
        time.sleep(0.12)
        ser.rts = False     # EN high -> chip boots, IO0 is high -> runs the sketch
        time.sleep(0.05)

    print(f"[monitor] listening on {port} @ {baud} baud.  Ctrl+C to stop.")
    print("-" * 62)

    last_data = time.time()
    warned = False
    try:
        while True:
            chunk = ser.read(4096)
            if chunk:
                last_data = time.time()
                warned = False
                sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                sys.stdout.flush()
            elif not warned and time.time() - last_data > 6:
                warned = True
                print("\n[monitor] 6 s of silence. If you saw nothing at all:")
                print("[monitor]  * press the RST button on the board now")
                print("[monitor]  * if still silent, the chip is not booting or")
                print("[monitor]    TX is not wired through - see the troubleshooting table")
                print("-" * 62)
    except KeyboardInterrupt:
        print("\n[monitor] stopped.")
    finally:
        ser.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal ESP32 serial monitor")
    ap.add_argument("--port", default=DEFAULT_PORT, help=f"serial port (default {DEFAULT_PORT})")
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD, help=f"baud rate (default {DEFAULT_BAUD})")
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    ap.add_argument("--no-reset", action="store_true", help="do not pulse EN on connect")
    args = ap.parse_args()

    if args.list:
        list_all_ports()
        return
    monitor(args.port, args.baud, not args.no_reset)


if __name__ == "__main__":
    main()
