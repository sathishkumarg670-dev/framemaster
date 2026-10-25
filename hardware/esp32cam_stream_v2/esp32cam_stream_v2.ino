/* =====================================================================
   esp32cam_stream_v2.ino
   AI-Thinker ESP32-CAM  ->  MJPEG live stream in a browser

   v2 CHANGES FOR VISIONFORGE (src/esp32_recorder.py records this stream):
   - JSON line {"type":"heartbeat","node_id":..,"ip":..,"stream":..} every 2 s,
     also while streaming, so the PC finds the IP without resetting the board.
   - Hardware-JPEG sensors with PSRAM run at VGA 640x480 (YOLO's input size)
     instead of QVGA, and grab the LATEST frame so a slow Wi-Fi link drops
     stale frames instead of streaming delayed ones.
   - Wi-Fi credentials are placeholders: edit them locally, never commit them.
   v1 still works with the recorder; it is only discovered via a board reset.

   Target : ESP32 Arduino core 3.x  (tested API surface: 3.0.0 - 3.3.12)
   Board  : "AI Thinker ESP32-CAM"
   Author : written for DARK, 2026-10-25

   WHAT THIS DOES DIFFERENTLY:
   This sketch does NOT assume the sensor is an OV2640. It tries JPEG
   first; if the driver reports ESP_ERR_NOT_SUPPORTED (0x106) it falls
   back to RGB565, then GRAYSCALE, and converts each frame to JPEG in
   software so the browser still gets a normal MJPEG stream.

   STATUS LED (red LED on GPIO33, next to the antenna, ACTIVE LOW):
     very fast blink (5 Hz) ...... camera init FAILED
     double-blink every 1.5 s .... camera OK, Wi-Fi NOT connected
     one short blink every 2 s ... camera OK + Wi-Fi OK, server running
     LED goes dark/steady ........ a browser is actively streaming
   This lets you tell whether the chip booted even with zero serial output.
   ===================================================================== */

#include "esp_camera.h"
#include "img_converters.h"
#include <WiFi.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

/* =====================================================================
   >>>>>>>>>>>>>  ENTER YOUR WI-FI CREDENTIALS HERE  <<<<<<<<<<<<<
   Edit ONLY these two lines. 2.4 GHz hotspot only (ESP32 has no 5 GHz).
   Keep this file local - do not commit real credentials to GitHub.
   ===================================================================== */
const char *WIFI_SSID     = "YOUR_HOTSPOT_NAME";
const char *WIFI_PASSWORD = "YOUR_PASSWORD";
/* ===================================================================== */
const char *NODE_ID = "vf_cam_01";   /* becomes the camera id in VISIONFORGE */

/* ---- AI-Thinker ESP32-CAM pin map (matches the board in your photos) -- */
#define PWDN_GPIO_NUM   32
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM    0
#define SIOD_GPIO_NUM   26
#define SIOC_GPIO_NUM   27
#define Y9_GPIO_NUM     35
#define Y8_GPIO_NUM     34
#define Y7_GPIO_NUM     39
#define Y6_GPIO_NUM     36
#define Y5_GPIO_NUM     21
#define Y4_GPIO_NUM     19
#define Y3_GPIO_NUM     18
#define Y2_GPIO_NUM      5
#define VSYNC_GPIO_NUM  25
#define HREF_GPIO_NUM   23
#define PCLK_GPIO_NUM   22

#define STATUS_LED_GPIO 33   /* small red LED, active LOW  */
/* GPIO4 is the big white flash LED - deliberately left alone. */

/* ---- globals --------------------------------------------------------- */
WiFiServer httpServer(80);

static bool       g_cameraOK  = false;
static bool       g_wifiOK    = false;
static bool       g_streaming = false;
static pixformat_t g_format   = PIXFORMAT_JPEG;
static const char *g_formatName = "none";
static esp_err_t   g_lastCamErr = ESP_FAIL;

/* ---- sensor id -> human name ----------------------------------------- */
static const char *sensorName(uint16_t pid) {
  switch (pid) {
    case 0x26:   return "OV2640  (hardware JPEG: YES)";
    case 0x3660: return "OV3660  (hardware JPEG: YES)";
    case 0x5640: return "OV5640  (hardware JPEG: YES)";
    case 0x76:   return "OV7670  (hardware JPEG: NO)";
    case 0x77:   return "OV7725  (hardware JPEG: NO)";
    case 0x96:   return "OV9650  (hardware JPEG: NO)";
    case 0x1410: return "NT99141 (hardware JPEG: YES)";
    case 0x2145: return "GC2145  (hardware JPEG: NO)";
    case 0x232a: return "GC032A  (hardware JPEG: NO)";
    case 0x9b:   return "GC0308  (hardware JPEG: NO)";
    case 0x30:   return "BF3005  (hardware JPEG: NO)";
    case 0x20a6: return "BF20A6  (hardware JPEG: NO)";
    case 0x0031: return "SC031GS (hardware JPEG: NO)";
    case 0x9a46: return "SC030IOT(hardware JPEG: NO)";
    case 0xda4a: return "SC101IOT(hardware JPEG: NO)";
    default:     return "UNKNOWN sensor id";
  }
}

/* ---- base camera config ---------------------------------------------- */
static camera_config_t baseConfig() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer   = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM;   c.pin_d1 = Y3_GPIO_NUM;
  c.pin_d2 = Y4_GPIO_NUM;   c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM;   c.pin_d5 = Y7_GPIO_NUM;
  c.pin_d6 = Y8_GPIO_NUM;   c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk  = XCLK_GPIO_NUM;
  c.pin_pclk  = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM;
  c.pin_href  = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM;
  c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn  = PWDN_GPIO_NUM;
  c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.fb_location  = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  c.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
  return c;
}

/* ---- one init attempt ------------------------------------------------- */
static esp_err_t tryInit(pixformat_t fmt, framesize_t size,
                         uint32_t xclk, uint8_t fbCount, const char *label) {
  camera_config_t c = baseConfig();
  c.pixel_format  = fmt;
  c.frame_size    = size;
  c.xclk_freq_hz  = xclk;
  c.fb_count      = fbCount;
  c.jpeg_quality  = 12;               /* ignored unless fmt == JPEG */
  /* with 2 buffers, always hand out the newest frame (lower latency) */
  c.grab_mode     = fbCount > 1 ? CAMERA_GRAB_LATEST : CAMERA_GRAB_WHEN_EMPTY;

  Serial.printf("[CAM] attempt: %s, XCLK %lu Hz, fb_count %u ... ",
                label, (unsigned long)xclk, fbCount);
  esp_err_t err = esp_camera_init(&c);
  if (err == ESP_OK) {
    Serial.println("OK");
  } else {
    Serial.printf("FAILED 0x%03X (%s)\n", err, esp_err_to_name(err));
    esp_camera_deinit();             /* release XCLK / SCCB before retry */
    delay(200);
  }
  return err;
}

/* ---- camera bring-up with format fallback ----------------------------- */
static bool initCamera() {
  Serial.println("[CAM] ---------- camera bring-up ----------");
  Serial.printf("[CAM] PSRAM: %s\n", psramFound() ? "found" : "NOT FOUND");

  /* 1) JPEG - the fast path, works on OV2640/OV3660/OV5640/NT99141 */
  esp_err_t err = psramFound()
    ? tryInit(PIXFORMAT_JPEG, FRAMESIZE_VGA, 20000000, 2, "JPEG VGA 640x480")
    : tryInit(PIXFORMAT_JPEG, FRAMESIZE_QVGA, 20000000, 1, "JPEG QVGA 320x240");
  if (err == ESP_OK) { g_format = PIXFORMAT_JPEG; g_formatName = "JPEG (hardware)"; }

  /* 2) Probe worked but the sensor has no JPEG engine -> 0x106 */
  if (err == ESP_ERR_NOT_SUPPORTED) {
    Serial.println("[CAM] 0x106 = sensor was detected over SCCB but has no JPEG engine.");
    Serial.println("[CAM] Falling back to raw capture + software JPEG encoding.");
    err = tryInit(PIXFORMAT_RGB565, FRAMESIZE_QVGA, 20000000, 1,
                  "RGB565 QVGA 320x240");
    if (err == ESP_OK) { g_format = PIXFORMAT_RGB565; g_formatName = "RGB565 -> software JPEG"; }

    if (err != ESP_OK) {
      err = tryInit(PIXFORMAT_GRAYSCALE, FRAMESIZE_QQVGA, 10000000, 1,
                    "GRAYSCALE QQVGA 160x120");
      if (err == ESP_OK) { g_format = PIXFORMAT_GRAYSCALE; g_formatName = "GRAYSCALE -> software JPEG"; }
    }
  }

  g_lastCamErr = err;

  if (err != ESP_OK) {
    Serial.println("[CAM] *** CAMERA INIT FAILED - no video is possible ***");
    Serial.printf("[CAM] last error 0x%03X (%s)\n", err, esp_err_to_name(err));
    if (err == 0x105) {
      Serial.println("[CAM] 0x105 = sensor NOT detected at all.");
      Serial.println("[CAM] -> reseat the ribbon cable (contacts face the PCB),");
      Serial.println("[CAM]    and power the board from a supply that holds 5V/500mA.");
    }
    return false;
  }

  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    Serial.printf("[CAM] sensor PID 0x%04X -> %s\n", s->id.PID, sensorName(s->id.PID));
    s->set_vflip(s, 0);
    s->set_hmirror(s, 0);
  }
  Serial.printf("[CAM] active format: %s\n", g_formatName);
  Serial.println("[CAM] camera ready.");
  return true;
}

/* ---- Wi-Fi ------------------------------------------------------------ */
static bool initWiFi() {
  if (strcmp(WIFI_SSID, "YOUR_HOTSPOT_NAME") == 0) {
    Serial.println("[WIFI] *** You have not edited WIFI_SSID / WIFI_PASSWORD yet. ***");
    return false;
  }
  Serial.printf("[WIFI] connecting to \"%s\" ", WIFI_SSID);
  WiFi.persistent(false);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 20000) {
    delay(400);
    Serial.print('.');
  }
  Serial.println();

  if (WiFi.status() != WL_CONNECTED) {
    Serial.printf("[WIFI] FAILED (status %d). Check: 2.4 GHz only, WPA2, SSID/password spelling.\n",
                  WiFi.status());
    return false;
  }
  Serial.printf("[WIFI] connected. RSSI %d dBm\n", WiFi.RSSI());
  Serial.println("================================================");
  Serial.print  ("  OPEN THIS IN YOUR BROWSER:  http://");
  Serial.println(WiFi.localIP());
  Serial.println("================================================");
  return true;
}

/* ---- machine-readable heartbeat for src/hardware_bridge.py ------------ */
static void jsonHeartbeat(uint32_t frames) {
  static uint32_t last = 0;
  if (!g_wifiOK || millis() - last < 2000) return;
  last = millis();
  String ip = WiFi.localIP().toString();
  Serial.printf("{\"type\":\"heartbeat\",\"node_id\":\"%s\",\"ip\":\"%s\","
                "\"stream\":\"http://%s/stream\",\"ms\":%lu,\"rssi\":%d,"
                "\"camera_ok\":%s,\"streaming\":%s,\"frames\":%lu}\n",
                NODE_ID, ip.c_str(), ip.c_str(), (unsigned long)millis(), WiFi.RSSI(),
                g_cameraOK ? "true" : "false", g_streaming ? "true" : "false",
                (unsigned long)frames);
}

/* ---- status LED ------------------------------------------------------- */
static void statusLed() {
  if (g_streaming) { digitalWrite(STATUS_LED_GPIO, HIGH); return; }
  uint32_t t = millis();
  bool on;
  if (!g_cameraOK)     on = (t % 200) < 100;                               /* 5 Hz   */
  else if (!g_wifiOK)  on = ((t % 1500) < 90) || (t % 1500 > 220 && t % 1500 < 310);
  else                 on = (t % 2000) < 80;                               /* 0.5 Hz */
  digitalWrite(STATUS_LED_GPIO, on ? LOW : HIGH);   /* active LOW */
}

/* ---- grab one frame, always hand back JPEG bytes ---------------------- */
static bool grabJpeg(uint8_t **out, size_t *outLen, camera_fb_t **fbOut, bool *mustFree) {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) { Serial.println("[CAM] frame grab failed"); return false; }
  *fbOut = fb;
  if (fb->format == PIXFORMAT_JPEG) {
    *out = fb->buf; *outLen = fb->len; *mustFree = false;
    return true;
  }
  uint8_t *jpg = NULL; size_t jpgLen = 0;
  if (!frame2jpg(fb, 80, &jpg, &jpgLen)) {
    Serial.println("[CAM] software JPEG conversion failed");
    esp_camera_fb_return(fb); *fbOut = NULL;
    return false;
  }
  *out = jpg; *outLen = jpgLen; *mustFree = true;
  return true;
}

/* ---- HTTP responses ---------------------------------------------------- */
static void sendIndex(WiFiClient &client) {
  String body;
  body.reserve(900);
  body  = F("<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>ESP32-CAM</title></head>"
            "<body style='margin:0;background:#101214;color:#e6e6e6;"
            "font-family:system-ui,sans-serif;text-align:center'>"
            "<h3 style='margin:14px 0 6px'>ESP32-CAM live stream</h3>"
            "<p style='margin:0 0 10px;font-size:13px;color:#8a929b'>format: ");
  body += g_formatName;
  body += F("</p>");
  if (g_cameraOK) {
    body += F("<img src='/stream' style='max-width:100%;border:1px solid #2a2f36'>"
              "<p style='font-size:13px'><a style='color:#5db0ff' href='/jpg'>single snapshot</a></p>");
  } else {
    body += F("<p style='color:#ff6b6b'>Camera initialisation failed. "
              "No video available. Check the serial log.</p>");
  }
  body += F("</body></html>");

  client.print(F("HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nConnection: close\r\nContent-Length: "));
  client.print(body.length());
  client.print(F("\r\n\r\n"));
  client.print(body);
}

static void sendSnapshot(WiFiClient &client) {
  uint8_t *jpg = NULL; size_t len = 0; camera_fb_t *fb = NULL; bool mustFree = false;
  if (!grabJpeg(&jpg, &len, &fb, &mustFree)) {
    client.print(F("HTTP/1.1 500 Internal Server Error\r\nConnection: close\r\n\r\ncapture failed"));
    return;
  }
  client.print(F("HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\n"
                 "Cache-Control: no-store\r\nConnection: close\r\nContent-Length: "));
  client.print(len);
  client.print(F("\r\n\r\n"));
  client.write(jpg, len);
  if (mustFree) free(jpg);
  if (fb) esp_camera_fb_return(fb);
}

static void sendStream(WiFiClient &client) {
  Serial.println("[HTTP] stream started");
  g_streaming = true;
  client.print(F("HTTP/1.1 200 OK\r\n"
                 "Content-Type: multipart/x-mixed-replace; boundary=espframe\r\n"
                 "Cache-Control: no-cache, private\r\n"
                 "Pragma: no-cache\r\n"
                 "Connection: close\r\n\r\n"));

  uint32_t frames = 0, t0 = millis();
  while (client.connected()) {
    uint8_t *jpg = NULL; size_t len = 0; camera_fb_t *fb = NULL; bool mustFree = false;
    if (!grabJpeg(&jpg, &len, &fb, &mustFree)) break;

    char hdr[96];
    int n = snprintf(hdr, sizeof(hdr),
                     "--espframe\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n",
                     (unsigned)len);
    client.write((const uint8_t *)hdr, n);
    client.write(jpg, len);
    client.write((const uint8_t *)"\r\n", 2);

    if (mustFree) free(jpg);
    if (fb) esp_camera_fb_return(fb);

    jsonHeartbeat(frames);
    if (++frames % 30 == 0) {
      Serial.printf("[HTTP] %lu frames, %.1f fps\n",
                    (unsigned long)frames, 30000.0f / (millis() - t0));
      t0 = millis();
    }
    delay(1);                  /* yield to Wi-Fi / watchdog */
  }
  g_streaming = false;
  client.stop();
  Serial.printf("[HTTP] stream ended after %lu frames\n", (unsigned long)frames);
}

/* ---- request dispatch -------------------------------------------------- */
static void handleClient(WiFiClient client) {
  client.setTimeout(3000);
  String reqLine = client.readStringUntil('\n');
  if (reqLine.length() == 0) { client.stop(); return; }
  reqLine.trim();

  /* drain the remaining request headers */
  while (client.connected()) {
    String line = client.readStringUntil('\n');
    if (line.length() <= 1) break;
  }

  Serial.printf("[HTTP] %s\n", reqLine.c_str());

  if (reqLine.startsWith("GET /stream") && g_cameraOK)      sendStream(client);
  else if (reqLine.startsWith("GET /jpg") && g_cameraOK)    { sendSnapshot(client); client.stop(); }
  else if (reqLine.startsWith("GET /favicon")) {
    client.print(F("HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n"));
    client.stop();
  } else { sendIndex(client); client.stop(); }
}

/* ======================================================================== */
void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);   /* stop brown-out reboot loops on weak USB power */

  pinMode(STATUS_LED_GPIO, OUTPUT);
  digitalWrite(STATUS_LED_GPIO, LOW);          /* LED ON during boot */

  Serial.begin(115200);
  delay(1500);                                 /* give the monitor time to attach */
  Serial.println();
  Serial.println("################################################");
  Serial.println("#  esp32cam_stream_v2  -  build 2026-10-25      #");
  Serial.println("#  If you can read this, the chip booted and    #");
  Serial.println("#  serial output is reaching your PC.           #");
  Serial.println("################################################");
  Serial.printf("[SYS] chip %s, %d core(s), %lu MHz, free heap %lu\n",
                ESP.getChipModel(), ESP.getChipCores(),
                (unsigned long)ESP.getCpuFreqMHz(),
                (unsigned long)ESP.getFreeHeap());

  g_cameraOK = initCamera();
  g_wifiOK   = initWiFi();

  if (g_wifiOK) {
    httpServer.begin();
    httpServer.setNoDelay(true);
    Serial.println("[HTTP] server listening on port 80");
  }
  if (!g_cameraOK) Serial.println("[SYS] running in DIAGNOSTIC mode - camera unavailable.");
}

void loop() {
  statusLed();

  if (g_wifiOK && WiFi.status() != WL_CONNECTED) {
    Serial.println("[WIFI] link lost, reconnecting...");
    g_wifiOK = false;
    WiFi.reconnect();
    uint32_t start = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - start < 10000) { delay(300); statusLed(); }
    if (WiFi.status() == WL_CONNECTED) {
      g_wifiOK = true;
      Serial.print("[WIFI] reconnected, IP http://");
      Serial.println(WiFi.localIP());
    }
  }

  jsonHeartbeat(0);

  if (g_wifiOK) {
    WiFiClient client = httpServer.available();
    if (client) handleClient(client);
  }
}
