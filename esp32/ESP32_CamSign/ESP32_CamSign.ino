// CAMERA-IN-THE-LOOP test of the final Moroccan traffic-sign model on the ESP32-CAM (revision, R1.5 / R2).
// Everything runs on the board: camera frame -> centre square -> 48x48 RGB -> int8 model -> class.
//
// Arduino IDE: board "AI Thinker ESP32-CAM" (esp32 core 3.x). Wi-Fi is not used.
// live_model.h = final model c48_relu40k_s42_maR (same file as the energy test).
//
// Two ways to use it (both at 921600 baud):
//  1. Serial Monitor: prints a LIVE line about once per second (top-3 classes, times). Good for aiming.
//  2. camera_test.py on the PC (close the Serial Monitor first): the PC sends 'c', the board takes
//     5 fresh frames, and for each prints
//        RES <k> <capture_ms> <preprocess_ms> <infer_ms> <c1> <p1> <c2> <p2> <c3> <p3>
//     followed by one line  IMG <hex of the last 48x48x3 RGB image>  so the PC can save what the model saw.

#include <Arduino.h>
#include <WiFi.h>
#include <math.h>
#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#if __has_include("tensorflow/lite/schema/schema_generated.h")
#include "tensorflow/lite/schema/schema_generated.h"
#else
#include "tensorflow/lite/micro/schema/schema_generated.h"
#endif
#include "live_model.h"
#include "class_names.h"

// ---- orientation: set to 1 if the image looks upside down / mirrored (same as V-Flip / H-Mirror on the web page)
#define CAM_VFLIP 0
#define CAM_HMIRROR 0

// ---- AI-Thinker ESP32-CAM pins
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22
#define FLASH_LED_PIN 4

namespace {
constexpr int kArenaSize = 64 * 1024;
constexpr int kFramesPerCapture = 5;
uint8_t* tensor_arena = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
float prob[N_CLASSES];
uint8_t rgb48[IMG_BYTES];          // last 48x48 RGB image (for the PC)
float in_scale; int in_zero;
bool live = true;
uint32_t last_live = 0;
float t_cap, t_pre, t_inf;
int top[3];

bool camera_init() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0; c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM; c.pin_d1 = Y3_GPIO_NUM; c.pin_d2 = Y4_GPIO_NUM; c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM; c.pin_d5 = Y7_GPIO_NUM; c.pin_d6 = Y8_GPIO_NUM; c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM; c.pin_pclk = PCLK_GPIO_NUM; c.pin_vsync = VSYNC_GPIO_NUM; c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM; c.pin_sccb_scl = SIOC_GPIO_NUM; c.pin_pwdn = PWDN_GPIO_NUM; c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_RGB565;   // raw pixels, no JPEG decoding needed
  c.frame_size = FRAMESIZE_QVGA;       // 320 x 240
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.fb_count = 2;
  c.grab_mode = CAMERA_GRAB_LATEST;
  if (esp_camera_init(&c) != ESP_OK) return false;
  sensor_t* s = esp_camera_sensor_get();
  s->set_vflip(s, CAM_VFLIP);
  s->set_hmirror(s, CAM_HMIRROR);
  return true;
}

// centre square of the frame (h x h), averaged down to 48 x 48 RGB (box filter), then quantized into the model input
bool grab_and_preprocess() {
  uint32_t t0 = micros();
  camera_fb_t* fb = esp_camera_fb_get();          // drop one frame so the image is fresh
  if (fb) esp_camera_fb_return(fb);
  fb = esp_camera_fb_get();
  if (!fb) return false;
  uint32_t t1 = micros();
  const int W = fb->width, H = fb->height, S = IMG_SIZE;
  const int side = min(W, H), x0 = (W - side) / 2, y0 = (H - side) / 2;
  const uint8_t* buf = fb->buf;
  for (int oy = 0; oy < S; oy++) {
    int ya = y0 + oy * side / S, yb = y0 + (oy + 1) * side / S;
    for (int ox = 0; ox < S; ox++) {
      int xa = x0 + ox * side / S, xb = x0 + (ox + 1) * side / S;
      uint32_t r = 0, g = 0, b = 0, n = 0;
      for (int y = ya; y < yb; y++) {
        const uint8_t* row = buf + 2 * (y * W);
        for (int x = xa; x < xb; x++) {
          uint16_t p = (row[2 * x] << 8) | row[2 * x + 1];   // RGB565, high byte first
          r += ((p >> 11) & 31) * 255 / 31;
          g += ((p >> 5) & 63) * 255 / 63;
          b += (p & 31) * 255 / 31;
          n++;
        }
      }
      uint8_t* o = rgb48 + 3 * (oy * S + ox);
      o[0] = r / n; o[1] = g / n; o[2] = b / n;
    }
  }
  esp_camera_fb_return(fb);
  for (int i = 0; i < IMG_BYTES; i++) {
    int q = (int)lroundf(rgb48[i] / 255.f / in_scale) + in_zero;
    input->data.int8[i] = (int8_t)max(-128, min(127, q));
  }
  t_cap = (t1 - t0) / 1000.f;
  t_pre = (micros() - t1) / 1000.f;
  return true;
}

void classify() {
  uint32_t t0 = micros();
  interpreter->Invoke();
  t_inf = (micros() - t0) / 1000.f;
  const float s = output->params.scale; const int z = output->params.zero_point;
  float mx = -1e30f, sum = 0.f;
  for (int c = 0; c < N_CLASSES; c++) { prob[c] = (output->data.int8[c] - z) * s; mx = max(mx, prob[c]); }
  for (int c = 0; c < N_CLASSES; c++) { prob[c] = expf(prob[c] - mx); sum += prob[c]; }
  for (int c = 0; c < N_CLASSES; c++) prob[c] /= sum;
  for (int k = 0; k < 3; k++) {
    int best = -1;
    for (int c = 0; c < N_CLASSES; c++) {
      if ((k > 0 && c == top[0]) || (k > 1 && c == top[1])) continue;
      if (best < 0 || prob[c] > prob[best]) best = c;
    }
    top[k] = best;
  }
}
}  // namespace

void setup() {
  WiFi.mode(WIFI_OFF);
  btStop();
  pinMode(FLASH_LED_PIN, OUTPUT); digitalWrite(FLASH_LED_PIN, LOW);   // flash LED off
  Serial.begin(921600);
  delay(300);
  if (!camera_init()) { Serial.println("ERR camera init failed (check the board selection and the ribbon cable)"); return; }
  uint8_t* raw = (uint8_t*)heap_caps_malloc(kArenaSize + 16, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!raw) { Serial.println("ERR not enough RAM"); return; }
  tensor_arena = (uint8_t*)(((uintptr_t)raw + 15) & ~(uintptr_t)15);
  static tflite::MicroMutableOpResolver<10> resolver;
  resolver.AddConv2D(); resolver.AddDepthwiseConv2D(); resolver.AddFullyConnected(); resolver.AddMean();
  resolver.AddReshape(); resolver.AddQuantize(); resolver.AddDequantize(); resolver.AddAveragePool2D();
  resolver.AddSoftmax(); resolver.AddPad();
  static tflite::MicroInterpreter static_interpreter(tflite::GetModel(g_model), resolver, tensor_arena, kArenaSize);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) { Serial.println("ERR AllocateTensors failed"); return; }
  input = interpreter->input(0);
  output = interpreter->output(0);
  in_scale = input->params.scale; in_zero = input->params.zero_point;
  Serial.printf("READY %s %dx%d, camera QVGA RGB565, centre square -> %dx%d\n", RUN_NAME, IMG_SIZE, IMG_SIZE, IMG_SIZE, IMG_SIZE);
}

void loop() {
  if (!interpreter) { delay(1000); return; }
  while (Serial.available()) {
    char c = Serial.read();
    if (c == 'l') live = !live;
    if (c == 'c') {
      for (int k = 0; k < kFramesPerCapture; k++) {
        if (!grab_and_preprocess()) { Serial.println("ERR capture"); continue; }
        classify();
        Serial.printf("RES %d %.1f %.1f %.1f %d %.4f %d %.4f %d %.4f\n", k, t_cap, t_pre, t_inf,
                      top[0], prob[top[0]], top[1], prob[top[1]], top[2], prob[top[2]]);
      }
      Serial.print("IMG ");
      for (int i = 0; i < IMG_BYTES; i++) Serial.printf("%02x", rgb48[i]);
      Serial.println();
      last_live = millis();
    }
  }
  if (live && millis() - last_live > 1000) {
    if (grab_and_preprocess()) {
      classify();
      Serial.printf("LIVE  %-32s %3.0f%%  | 2nd %-26s %3.0f%% | capture %.0f ms, prep %.0f ms, model %.0f ms\n",
                    CLASS_NAMES[top[0]] + 5, prob[top[0]] * 100, CLASS_NAMES[top[1]] + 5, prob[top[1]] * 100, t_cap, t_pre, t_inf);
    }
    last_live = millis();
  }
}
