"""ESP32 energy (R1.6) from the USB power meter readings (KWS-MX17, whole board, 30-minute runs).
Readings taken from the photos of the meter at 00:30:00 (2026-11-23)."""
V = 5.15                  # V, meter reading in both runs
T_H = 0.5                 # h (30:00)
WH_INFER, MAH_INFER = 0.140, 29     # ESP32_Energy_INFER
WH_IDLE, MAH_IDLE = 0.102, 19       # ESP32_Energy_IDLE
T_IMG = 0.1886            # s per image (serial log: 188.6 ms, 319 images per minute)
RES = 0.0005              # Wh, half of the meter's 0.001 Wh step
p_inf, p_idle = WH_INFER / T_H, WH_IDLE / T_H
n_img = T_H * 3600 / T_IMG
print(f"images in 30 min: {n_img:.0f}")
print(f"{'mode':<12}{'Wh':>8}{'mAh':>6}{'avg power W':>13}{'current mA (Wh)':>17}{'current mA (mAh)':>18}")
for m, wh, mah, p in (("inference", WH_INFER, MAH_INFER, p_inf), ("idle", WH_IDLE, MAH_IDLE, p_idle)):
    print(f"{m:<12}{wh:>8.3f}{mah:>6}{p:>13.3f}{p / V * 1000:>17.1f}{mah / T_H:>18.0f}")
d = p_inf - p_idle
print(f"difference (classifier): {d:.3f} W, {d / V * 1000:.1f} mA")
print(f"energy per image, whole board: {p_inf * T_IMG * 1000:.1f} mJ "
      f"(range {(WH_INFER - RES) / T_H * T_IMG * 1000:.1f}-{(WH_INFER + RES) / T_H * T_IMG * 1000:.1f})")
print(f"extra energy per image (classifier): {d * T_IMG * 1000:.1f} mJ "
      f"(range {((WH_INFER - RES) - (WH_IDLE + RES)) / T_H * T_IMG * 1000:.1f}-{((WH_INFER + RES) - (WH_IDLE - RES)) / T_H * T_IMG * 1000:.1f})")
