// Live test of the Moroccan traffic-sign model on the ESP32: the PC app (sign_live.py) sends one
// int8 image over USB, the ESP32 classifies it and sends back its top 3 classes and the time taken.
//
// Arduino IDE: board "ESP32 Dev Module" (esp32 core 3.x), Partition Scheme "Huge APP (3MB No OTA/1MB SPIFFS)".
// Make live_model.h first:  python esp32\live_pack.py --run runs\c48_relu40k_s42_maT
//
// Protocol (921600 baud):
//   ESP32 -> PC at start:   "READY <size> <n_classes> <run name>\n"
//   PC -> ESP32:            'S' 'I' 'G' 'N' then IMG_BYTES bytes (int8, RGB, row by row, already quantized)
//   ESP32 -> PC:            "RES <ms> <class1> <prob1> <class2> <prob2> <class3> <prob3>\n"  (class = index 0..N-1)

#include <Arduino.h>
#include <math.h>
#include <string.h>

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
TfLiteTensor* output = nullptr;
float prob[N_CLASSES];

bool read_exact(uint8_t* dst, size_t n, uint32_t timeout_ms) {
  size_t got = 0;
  uint32_t t0 = millis();
  while (got < n) {
    int a = Serial.available();
    if (a > 0) {
      got += Serial.readBytes(dst + got, min((size_t)a, n - got));
      t0 = millis();
    } else if (millis() - t0 > timeout_ms) {
      return false;
    }
  }
  return true;
}
}  // namespace

void setup() {
  Serial.setRxBufferSize(IMG_BYTES + 512);
  Serial.begin(921600);
  delay(300);
  uint8_t* raw = (uint8_t*)malloc(kArenaSize + 16);
  if (!raw) { Serial.println("ERR not enough RAM"); return; }
  tensor_arena = (uint8_t*)(((uintptr_t)raw + 15) & ~(uintptr_t)15);
  const tflite::Model* model = tflite::GetModel(g_model);
  static tflite::MicroMutableOpResolver<10> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddMean();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddAveragePool2D();
  resolver.AddSoftmax();
  resolver.AddPad();
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kArenaSize);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) { Serial.println("ERR AllocateTensors failed"); return; }
  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.printf("READY %d %d %s\n", IMG_SIZE, N_CLASSES, RUN_NAME);
}

void loop() {
  if (!interpreter) { delay(1000); return; }
  // wait for the 4-byte start marker "SIGN"
  static const char kMagic[4] = {'S', 'I', 'G', 'N'};
  static int matched = 0;
  while (Serial.available()) {
    char c = Serial.read();
    matched = (c == kMagic[matched]) ? matched + 1 : (c == 'S' ? 1 : 0);
    if (matched == 4) break;
  }
  if (matched < 4) { delay(1); return; }
  matched = 0;
  if (!read_exact((uint8_t*)input->data.int8, IMG_BYTES, 2000)) { Serial.println("ERR timeout"); return; }

  uint32_t t0 = micros();
  interpreter->Invoke();
  float ms = (micros() - t0) / 1000.f;

  // softmax over the dequantized class scores
  const float s = output->params.scale;
  const int z = output->params.zero_point;
  float mx = -1e30f, sum = 0.f;
  for (int c = 0; c < N_CLASSES; c++) { prob[c] = (output->data.int8[c] - z) * s; mx = max(mx, prob[c]); }
  for (int c = 0; c < N_CLASSES; c++) { prob[c] = expf(prob[c] - mx); sum += prob[c]; }
  int top[3] = {0, 0, 0};
  for (int k = 0; k < 3; k++) {
    int best = -1;
    for (int c = 0; c < N_CLASSES; c++) {
      if ((k > 0 && c == top[0]) || (k > 1 && c == top[1])) continue;
      if (best < 0 || prob[c] > prob[best]) best = c;
    }
    top[k] = best;
  }
  Serial.printf("RES %.2f %d %.4f %d %.4f %d %.4f\n", ms, top[0], prob[top[0]] / sum, top[1], prob[top[1]] / sum,
                top[2], prob[top[2]] / sum);
}
