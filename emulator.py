from math import sqrt

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
LAYERS      = 6
HEADS       = 8
MLP_SCALE   = 4
EMBED_SIZE  = 128
HEAD_SIZE   = EMBED_SIZE // HEADS
VOCAB_SIZE  = 361
OUTPUT_SIZE = 64  # ???

WEIGHTS_PATH = "quantized/weight_files"

# ---------------------------------------------------------------------------
# Fixed-point arithmetic
# ---------------------------------------------------------------------------
# All activations are 24-bit unsigned integers representing signed values in
# two's complement. The real value of a stored integer v is v / 2^18.
#
#   Positive:  0x000000 – 0x7FFFFF  →  0       to  +32767.99...
#   Negative:  0x800000 – 0xFFFFFF  →  -32768  to  -0.000004
#
FIXED_POINT_SIZE  = 24
FIXED_POINT_MASK  = (1 << FIXED_POINT_SIZE) - 1   # 0xFFFFFF
MATMUL_FIXED_POINT = 18                           # fractional bits

# Epsilon added to variance before sqrt to avoid division by zero.
# Scaled to match the variance accumulator's fixed-point space: eps * N * 2^36
EPS = int(1e-5 * EMBED_SIZE * (1 << (2 * MATMUL_FIXED_POINT)))

# ---------------------------------------------------------------------------
# Precomputed reciprocal constants
# ---------------------------------------------------------------------------
# Integer division is avoided by multiplying with a precomputed reciprocal
# and then right-shifting. Pattern: x / c  =  (x * round(2^N / c)) >> N
#
# LAYERNORM_CONST:   x / EMBED_SIZE  =  (x * LAYERNORM_CONST) >> 32
LAYERNORM_CONST   = int((1 << 32) / EMBED_SIZE)

# LAYERNORM_CONST_2: x / sqrt(EMBED_SIZE)  =  (x * LAYERNORM_CONST_2) >> 27
LAYERNORM_CONST_2 = int((1 << 27) / sqrt(EMBED_SIZE))


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------
class LayerNorm:
    """
    Layer normalization with a learned per-dimension scale (gamma), no bias.

    Given input x of length EMBED_SIZE, computes:
        output[i] = gamma[i] * (x[i] - mean(x)) / std(x)

    All arithmetic uses 24-bit fixed-point with 18 fractional bits.
    Division by N and sqrt are replaced by multiply-then-shift using
    precomputed constants. The sqrt itself would be a lookup table in
    Redstone; here we use math.sqrt to emulate that.

    Weight file format:
        EMBED_SIZE * 3 bytes, little-endian unsigned integers.
        Each value encodes gamma as:  file_value = round(gamma * 2^22)
        Dividing by 2 on load gives the internal representation gamma * 2^21,
        which aligns with the >> 21 shift applied in forward().
    """

    def __init__(self, index: int):
        self.weights: list[int] = []
        with open(f"{WEIGHTS_PATH}/layernorm/ln_{index}.bin", "rb") as f:
            for _ in range(EMBED_SIZE):
                raw = int.from_bytes(f.read(3), byteorder="little")
                self.weights.append(raw // 2)  # internal: gamma * 2^21

    def forward(self, x: list[int]) -> list[int]:
        # ------------------------------------------------------------------
        # Step 1: Compute the mean
        # Sign-extend each 24-bit value to 32 bits before accumulating so
        # that negative values don't inflate the sum.
        # ------------------------------------------------------------------
        total = 0
        for v in x:
            #255 = 0x7F
            #FIXED_POINT_MASK // 2 = 0x7FFFFF
            sign_extension = (255 << FIXED_POINT_SIZE) if v > FIXED_POINT_MASK // 2 else 0
            total += v + sign_extension
        total &= (1 << 32) - 1 # keep in 32 bit signed int

        # Divide by EMBED_SIZE via multiply-then-shift (no real division).
        # Check sign of the 32-bit sum using bit 31.
        negative_mean = total >= (1 << (FIXED_POINT_SIZE + 7)) # if total negative
        if negative_mean:
            total = (-total) & ((1 << (FIXED_POINT_SIZE + 7)) - 1)
        mean = (total * LAYERNORM_CONST) >> 32 # mean = total >> 7 will also work for us
        if negative_mean:
            mean = (-mean) & FIXED_POINT_MASK

        # ------------------------------------------------------------------
        # Step 2: Compute the standard deviation
        # Accumulate sum of squared deviations: sum((x[i] - mean)^2).
        # Diffs are in Q18 so their squares are in Q36; EPS is pre-scaled to Q36.
        # ------------------------------------------------------------------
        variance_acc = EPS
        for v in x:
            diff = (v - mean) & FIXED_POINT_MASK
            if diff > FIXED_POINT_MASK // 2: #if diff negative
                diff = (-diff) & (FIXED_POINT_MASK // 2)  # work with abs value
            variance_acc += diff * diff
            variance_acc &= (1 << 48) - 1

        # sqrt gives sigma in Q18 space (i.e. real_sigma * 2^18).
        # In Redstone this step is a lookup table; math.sqrt emulates it.
        sigma = int(sqrt(variance_acc))

        # Divide sigma by sqrt(EMBED_SIZE) via multiply-then-shift.
        sigma = (LAYERNORM_CONST_2 * sigma) >> 27
        sigma &= FIXED_POINT_MASK

        # Compute 1/sigma in Q36 so the later multiply replaces a divide.
        # inv_sigma * sigma ≈ 2^36  →  x * inv_sigma >> 18 ≈ x / sigma * 2^18
        inv_sigma = ((1 << (2 * MATMUL_FIXED_POINT)) // sigma) & FIXED_POINT_MASK

        # ------------------------------------------------------------------
        # Step 3: Normalize and apply the learned scale (gamma)
        # ------------------------------------------------------------------
        result: list[int] = []
        for i, v in enumerate(x):
            diff = (v - mean) & FIXED_POINT_MASK

            # Track sign separately and work with the absolute value.
            negative = diff > FIXED_POINT_MASK // 2
            if negative:
                diff = (-diff) & (FIXED_POINT_MASK // 2)

            # x_hat = |x[i] - mean| / sigma  (result in Q18)
            x_hat = ((diff * inv_sigma) >> MATMUL_FIXED_POINT) & (FIXED_POINT_MASK // 2)

            # Apply gamma: output = gamma * x_hat
            #   weights[i] encodes gamma * 2^21
            #   x_hat encodes x_hat * 2^18
            #   product >> 21 leaves gamma * x_hat * 2^18  (back in Q18)
            out = ((x_hat * self.weights[i]) >> (MATMUL_FIXED_POINT + 3)) & (FIXED_POINT_MASK // 2)

            if negative:
                out = (-out) & FIXED_POINT_MASK
            result.append(out)

        return result
