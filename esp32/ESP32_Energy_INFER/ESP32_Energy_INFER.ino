// ENERGY TEST, long run, measured with an inline USB power meter (revision, Reviewer 1 point 6).
// This sketch runs ONE mode for as long as it is powered:  MODE = INFER
//   INFER : classifies a fixed image again and again, as fast as possible (final model c48_relu40k_s42_maR)
//   IDLE  : same model loaded, CPU only waits (delay)
// Wi-Fi and Bluetooth are OFF in both modes. Every 60 s the serial monitor (115200) prints the elapsed time
// and, in INFER mode, the number of images and the time per image.
// Procedure: upload, wait for the first line, press RESET on the USB tester, run 30 minutes, then write down
// the tester's mAh, Wh, time and V, and the last serial line.
//
// Arduino IDE: board "ESP32 Dev Module", Partition Scheme "Huge APP (3MB No OTA/1MB SPIFFS)".

#define MODE_INFER 1

#include <Arduino.h>
#include <WiFi.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#if __has_include("tensorflow/lite/schema/schema_generated.h")
#include "tensorflow/lite/schema/schema_generated.h"
#else
#include "tensorflow/lite/micro/schema/schema_generated.h"
#endif
#include "live_model.h"

namespace {
constexpr int kArenaSize = 64 * 1024;
uint8_t* tensor_arena = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
uint32_t start_ms = 0, last_report = 0;
uint64_t n_inf = 0, t_inf_us = 0;
}  // namespace

void setup() {
  WiFi.mode(WIFI_OFF);
  btStop();
  Serial.begin(115200);
  delay(300);
  uint8_t* raw = (uint8_t*)malloc(kArenaSize + 16);
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
  uint32_t seed = 12345;
  for (int i = 0; i < IMG_BYTES; i++) { seed = seed * 1103515245u + 12345u; input->data.int8[i] = (int8_t)(seed >> 24); }
  Serial.printf("ENERGY TEST, MODE = %s, model %s. Press RESET on the USB tester now.\n", MODE_INFER ? "INFER" : "IDLE", RUN_NAME);
  start_ms = last_report = millis();
}

void loop() {
  if (!interpreter) { delay(1000); return; }
#if MODE_INFER
  uint32_t t0 = micros();
  interpreter->Invoke();
  t_inf_us += micros() - t0; n_inf++;
#else
  delay(10);
#endif
  if (millis() - last_report >= 60000) {
    last_report = millis();
    float mins = (millis() - start_ms) / 60000.f;
#if MODE_INFER
    Serial.printf("%5.1f min | %llu images | %.1f ms per image\n", mins, (unsigned long long)n_inf, t_inf_us / 1000.0 / n_inf);
#else
    Serial.printf("%5.1f min | IDLE\n", mins);
#endif
  }
}
