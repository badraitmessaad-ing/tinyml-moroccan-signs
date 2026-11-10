"""Matched baselines for the paper (same data, input, training recipe, int8 conversion as the
40k depthwise-separable model; only the network blocks change).

  --arch plain : standard 3x3 convolutions instead of depthwise-separable blocks
  --arch ir    : MobileNetV2-style inverted residual blocks (expand 1x1, depthwise 3x3,
                 linear 1x1 projection, residual when shapes match)

Same stem (3x3 stride 2), same strides (1, 2, 1, 2), same global pooling, dropout and linear head.
All other options and the whole training/evaluation/report pipeline are those of train_tiny.py.

Usage (project folder, tinyml env):
  python src\\train_baselines.py --arch plain --widths 16,16,32,40,48 --data ts1m_ma_48_clean --size 48 --head linear --out runs\\b48_plain_s42 --seed 42
  python src\\train_baselines.py --arch ir --expand 4 --widths 16,16,24,40,64 --data ts1m_ma_48_clean --size 48 --head linear --out runs\\b48_ir_s42 --seed 42
"""
import sys
import train_tiny as T
from tensorflow import keras
from tensorflow.keras import layers

ARCH, EXPAND = "plain", 4
argv = list(sys.argv)
for flag in ("--arch", "--expand"):
    if flag in argv:
        i = argv.index(flag)
        if flag == "--arch": ARCH = argv[i + 1]
        else: EXPAND = int(argv[i + 1])
        del argv[i:i + 2]
assert ARCH in ("plain", "ir"), "--arch must be plain or ir"
sys.argv = argv


def bn_relu(x, name, relu=True):
    x = layers.BatchNormalization(momentum=0.9, name=f"{name}_bn")(x)
    return layers.ReLU(6.0, name=f"{name}_relu")(x) if relu else x


def plain_block(x, filters, stride, name):
    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, name=f"{name}_conv")(x)
    return bn_relu(x, name)


def ir_block(x, filters, stride, name, t):
    cin = x.shape[-1]
    h = layers.Conv2D(cin * t, 1, use_bias=False, name=f"{name}_exp")(x)
    h = bn_relu(h, f"{name}_exp")
    h = layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False, name=f"{name}_dw")(h)
    h = bn_relu(h, f"{name}_dw")
    h = layers.Conv2D(filters, 1, use_bias=False, name=f"{name}_proj")(h)
    h = bn_relu(h, f"{name}_proj", relu=False)          # linear bottleneck
    if stride == 1 and cin == filters:
        h = layers.Add(name=f"{name}_add")([x, h])
    return h


def build_model(size, channels, n_classes, widths, head, head_dim, grid, basis,
                multiscale, dropout, pool="gap", pool_ch=32, act="relu6", act_grid=4, act_from=1):
    assert head == "linear" and pool == "gap" and not multiscale and act == "relu6", \
        "baselines use the configuration of the final model: linear head, GAP, ReLU6"
    inp = keras.Input((size, size, channels))
    x = layers.Conv2D(widths[0], 3, strides=2, padding="same", use_bias=False, name="stem")(inp)
    x = bn_relu(x, "stem")
    for i, (w, s) in enumerate(zip(widths[1:], [1, 2, 1, 2]), 1):
        x = plain_block(x, w, s, f"b{i}") if ARCH == "plain" else ir_block(x, w, s, f"b{i}", EXPAND)
    feat = layers.GlobalAveragePooling2D(name="gap")(x)
    feat = layers.Dropout(dropout, name="drop")(feat)
    out = layers.Dense(n_classes, name="logits")(feat)
    return keras.Model(inp, out, name=f"baseline_{ARCH}")


T.build_model = build_model
if __name__ == "__main__":
    T.main()
