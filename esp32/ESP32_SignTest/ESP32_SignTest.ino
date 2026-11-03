// ESP32 test of the Moroccan traffic-sign model (32x32 or 48x48 input), without a camera.
// The test images (your YouTube crops) and the "local photos" are stored in flash.
//
//   Step 1  classify all test images with the plain model -> accuracy, ms per image
//   Step 2  learn on the chip: build one prototype per class from the local photos (no training)
//   Step 3  classify again with the prototypes -> new accuracy
//   Step 4  speed loop for 20 s (read the USB power meter during this step for energy)
//
// Arduino IDE: board "ESP32 Dev Module" (esp32 core by Espressif, version 3.x),
//              Tools > Partition Scheme > "Huge APP (3MB No OTA/1MB SPIFFS)", Serial Monitor 115200.
// The headers are made by esp32_pack.py.

#include <Arduino.h>
#include <math.h>
#include <string.h>
#if __has_include(<esp_heap_caps.h>)
#include <esp_heap_caps.h>
#endif

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#if __has_include("tensorflow/lite/schema/schema_generated.h")
#include "tensorflow/lite/schema/schema_generated.h"
#else
#include "tensorflow/lite/micro/schema/schema_generated.h"   // newer TFLite Micro versions
#endif

#include "model_data.h"
#include "proto_params.h"
#include "test_data.h"
#include "adapt_data.h"

namespace {
constexpr int kArenaSize = 64 * 1024;   // big buffers come from the heap: the ESP32 has little static RAM
uint8_t* tensor_arena = nullptr;

tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* out_feat = nullptr;     // 160 features
TfLiteTensor* out_logit = nullptr;    // 70 class scores

float (*protos)[N_FEAT] = nullptr;   // prototype directions (unit length), N_CLASSES x N_FEAT
bool has_proto[N_CLASSES];
float feat[N_FEAT], logit[N_CLASSES];

bool is_speed(int c) {
  int id = class_ids[c];
  return id == 11 || id == 17 || id == 23 || id == 28 || id == 36 || id == 41 || id == 46;
}

// run the model on one stored image; fills feat[] (centred, unit length) and logit[]
void run_model(const int8_t* img) {
  memcpy(input->data.int8, img, IMG_BYTES);
  interpreter->Invoke();
  const float fs = out_feat->params.scale;
  const int fz = out_feat->params.zero_point;
  float nrm = 0.f;
  for (int i = 0; i < N_FEAT; i++) {
    feat[i] = (out_feat->data.int8[i] - fz) * fs - feat_mean[i];
    nrm += feat[i] * feat[i];
  }
  nrm = 1.f / (sqrtf(nrm) + 1e-8f);
  for (int i = 0; i < N_FEAT; i++) feat[i] *= nrm;
  const float ls = out_logit->params.scale;
  const int lz = out_logit->params.zero_point;
  for (int c = 0; c < N_CLASSES; c++) logit[c] = (out_logit->data.int8[c] - lz) * ls;
}

// class with the highest score; with prototypes, a class also scores alpha*s*cos if cos > tau
int predict(bool use_protos) {
  int best = 0;
  float best_v = -1e30f;
  for (int c = 0; c < N_CLASSES; c++) {
    float v = logit[c];
    if (use_protos && has_proto[c]) {
      float cs = 0.f;
      for (int i = 0; i < N_FEAT; i++) cs += feat[i] * protos[c][i];
      if (cs > PROTO_TAU && PROTO_ALPHA * LOGIT_SCALE * cs > v) v = PROTO_ALPHA * LOGIT_SCALE * cs;
    }
    if (v > best_v) { best_v = v; best = c; }
  }
  return best;
}

void test_all(bool use_protos, const uint8_t* pc_pred) {
  int ok = 0, ok_speed = 0, n_speed = 0, same_as_pc = 0;
  uint32_t t0 = micros();
  for (int k = 0; k < N_TEST; k++) {
    run_model(test_images[k]);
    int p = predict(use_protos);
    bool good = (p == test_labels[k]) || (p == test_alt[k]);
    ok += good;
    if (is_speed(test_labels[k])) { n_speed++; ok_speed += good; }
    same_as_pc += (p == pc_pred[k]);
  }
  float ms = (micros() - t0) / 1000.f / N_TEST;
  Serial.printf("  accuracy %.1f%%  (speed limits %.1f%% of %d)\n", 100.f * ok / N_TEST,
                n_speed ? 100.f * ok_speed / n_speed : 0.f, n_speed);
  Serial.printf("  %.2f ms per image (model%s)\n", ms, use_protos ? " + prototypes" : "");
  Serial.printf("  same answer as the PC on %d / %d images (PC accuracy %.1f%%)\n", same_as_pc, N_TEST,
                use_protos ? PC_ACC_PROTO : PC_ACC_HEAD);
}
}  // namespace

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.printf("\n=== Moroccan traffic signs on ESP32: %s ===\n", RUN_NAME);
  Serial.printf("CPU %d MHz, free heap %u bytes\n", getCpuFrequencyMhz(), ESP.getFreeHeap());

  // 1) working memory (tensor arena) and prototype buffers first: they are required
  Serial.printf("largest free block %u bytes\n", (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
  uint8_t* raw = (uint8_t*)malloc(kArenaSize + 16);
  protos = (float (*)[N_FEAT])malloc(sizeof(float) * N_CLASSES * N_FEAT);
  float (*sums)[N_FEAT] = (float (*)[N_FEAT])calloc(N_CLASSES * N_FEAT, sizeof(float));
  if (!raw || !protos || !sums) {
    Serial.println("not enough RAM");
    return;
  }
  tensor_arena = (uint8_t*)(((uintptr_t)raw + 15) & ~(uintptr_t)15);
  // 2) then copy the model into RAM if a block is still free (a little faster than reading flash)
  uint8_t* model_ram = (uint8_t*)malloc(g_model_len + 16);
  const uint8_t* model_ptr = g_model;
  if (model_ram) {
    uint8_t* aligned = (uint8_t*)(((uintptr_t)model_ram + 15) & ~(uintptr_t)15);
    memcpy(aligned, g_model, g_model_len);
    model_ptr = aligned;
    Serial.println("model copied to RAM");
  } else {
    Serial.println("model stays in flash (not enough RAM)");
  }

  const tflite::Model* model = tflite::GetModel(model_ptr);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("model schema version mismatch");
    return;
  }
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
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kArenaSize);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed (arena too small or unsupported operator)");
    return;
  }
  input = interpreter->input(0);
  for (size_t i = 0; i < interpreter->outputs_size(); i++) {
    TfLiteTensor* t = interpreter->output(i);
    if (t->dims->data[t->dims->size - 1] == N_CLASSES) out_logit = t; else out_feat = t;
  }
  Serial.printf("model %u bytes in flash, tensor arena used %u bytes of RAM\n", g_model_len,
                (unsigned)interpreter->arena_used_bytes());
  Serial.printf("%d test images, %d local photos\n\n", N_TEST, N_ADAPT);

  Serial.println("Step 1: plain model");
  test_all(false, pc_pred_head);

  Serial.println("\nStep 2: learning on the chip (prototypes, no training)");
  uint32_t t0 = micros();
  for (int k = 0; k < N_ADAPT; k++) {
    run_model(adapt_images[k]);
    for (int i = 0; i < N_FEAT; i++) sums[adapt_labels[k]][i] += feat[i];
  }
  int n_cls = 0;
  for (int c = 0; c < N_CLASSES; c++) {
    float nrm = 0.f;
    for (int i = 0; i < N_FEAT; i++) nrm += sums[c][i] * sums[c][i];
    has_proto[c] = nrm > 0.f;
    if (!has_proto[c]) continue;
    n_cls++;
    nrm = 1.f / sqrtf(nrm);
    for (int i = 0; i < N_FEAT; i++) protos[c][i] = sums[c][i] * nrm;
  }
  Serial.printf("  %d prototypes from %d photos in %.1f ms (%.2f ms per photo)\n", n_cls, N_ADAPT,
                (micros() - t0) / 1000.f, (micros() - t0) / 1000.f / N_ADAPT);
  Serial.printf("  prototype memory: %u bytes\n", (unsigned)(n_cls * N_FEAT * sizeof(float)));
  free(sums);

  Serial.println("\nStep 3: model + prototypes");
  test_all(true, pc_pred_proto);

  Serial.println("\nStep 4: speed loop for 20 s - read the USB power meter now");
  uint32_t n = 0, t1 = millis();
  while (millis() - t1 < 20000) {
    run_model(test_images[n % N_TEST]);
    predict(true);
    n++;
  }
  Serial.printf("  %u images in 20 s = %.1f images per second (%.2f ms each)\n", n, n / 20.f, 20000.f / n);
  Serial.println("\nDone. Energy per image = (current in A x 5 V) x (ms per image) / 1000, in joules.");
}

void loop() { delay(1000); }
