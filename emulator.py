"""
This emulator implements the forward pass of the custom transformed-based model implemented in the /models directory.
Its goal is to emulate the behavior of the model as closely as possible, including the specific fixed-point arithmetic 
and weight encoding used in the implementation of Sammyuri, which is possibly the most memory-efficient way to run 
such a model in Minecraft. Absolutely no floating-point arithmetic is used in the emulator.

It comprises of several classes that correspond to different components of the model, such as matrix multiplication, 
layer normalization, multi-head attention, and the feedforward network (MLP). It also utilizes a KV cache which does 
not exist in the original model, but is necessary for efficient model execution in Minecraft.

The 'run_model' function serves as the main entry point, allowing users to input a word and generate an image based on 
that word using the model.

A brief description of weights are as follows:
- The weights for the model are stored in binary files in the "quantized/weight_files" directory.
Each file corresponds to a specific component of the model (e.g., layer normalization, attention, MLP) 
and contains the weights for that component in a specific format.
- The weights are read from the files and processed to be used in the computations of the model. MatMul encodes
8-bit weights in a single byte, using a specific scheme to represent the weight values and the shifts to apply during multiplication. 
This is further explained in the MatMul class, which decodes the weights and performs the matrix multiplication using fixed-point arithmetic.
So the weight files are just flat arrays of raw bytes. All the variable-bit-shift interpretation — treating each byte as an asymmetric 
fixed-point multiplier — is done at load time in Python, not in the file format itself.
"""

from math import sqrt
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
LAYERS = 4
HEADS = 8
FFN_SCALING = 4
EMBED_SIZE = 128
HEAD_SIZE = EMBED_SIZE // HEADS
VOCAB_SIZE = 116
PIXEL_START_ID = 100   # 100 word tokens; patch tokens start here
PATCH_SIZE = 2
PIXELS_PER_PATCH = PATCH_SIZE * PATCH_SIZE   # 4 pixels per patch
NUM_PATCH_TOKENS = 1 << PIXELS_PER_PATCH     # 16 possible patch values
OUTPUT_SIZE = NUM_PATCH_TOKENS               # top-k covers all patch tokens
CONTEXT = 65           # 1 word token + 64 patch tokens (8x8 patch grid)
IMG_SIZE = 16          # 8 patches × patch_size 2 = 16 pixels per side

SAMPLE_TOP_K = 4       # number of top patch tokens to sample from
SAMPLE_TEMPERATURE = 1.0  # temperature applied before softmax (1.0 = no change)

WEIGHTS_PATH = "quantized/weight_files"

# ---------------------------------------------------------------------------
# Fixed-point arithmetic (for inputs and activation values)
# ---------------------------------------------------------------------------
# All activations are 24-bit unsigned integers representing signed values in
# two's complement. The real value of a stored integer v is v / 2^18.
#
#   Positive:  0x000000 – 0x7FFFFF  →  0       to  +32767.99...
#   Negative:  0x800000 – 0xFFFFFF  →  -32768  to  -0.000004
ACTIVATION_SIZE = 24
ACTIVATION_MASK = (1 << ACTIVATION_SIZE) - 1
MATMUL_FIXED_POINT = 18
MATMUL_EXTRA_PRECISION = 4
MATMUL_BIG_MASK = (1 << (ACTIVATION_SIZE + MATMUL_EXTRA_PRECISION)) - 1


# ---------------------------------------------------------------------------
# Precomputed reciprocal constants
# ---------------------------------------------------------------------------
# Integer division is avoided by multiplying with a precomputed reciprocal
# and then right-shifting. Pattern: x / c  =  (x * round(2^N / c)) >> N
#
# LAYERNORM_AVG_CONST:   x / EMBED_SIZE  =  (x * LAYERNORM_CONST) >> 32
# LAYERNORM_CONST_2: x / sqrt(EMBED_SIZE)  =  (x * LAYERNORM_CONST_2) >> 27
LAYERNORM_AVG_CONST = int((1 << 32) / EMBED_SIZE)
LAYERNORM_VAR_CONST = int((1 << 27) / sqrt(EMBED_SIZE))
ATT_CONST = int((1 << 26) / sqrt(HEAD_SIZE))

# Epsilon added to variance before sqrt to avoid division by zero.
# Scaled to match the variance accumulator's fixed-point space: eps * N * 2^36
EPS = int(1e-5 * EMBED_SIZE * (1 << (2 * MATMUL_FIXED_POINT)))


class MatMul:
    """
    Initialization decodes weights into (neg, shift, big, small) tuples using the table provided in the quantization documentation.

    Input is a vector of fixed-point values.

    Returns input @ weight_matrix
    """
    def __init__(self, weights, input_size, output_size, relu=False):
        self.weights = []
        for row in weights:
            self.weights.append([])
            for w in row:
                neg = (w >= 128)
                w %= 128
                if w < 64:
                    self.weights[-1].append((neg, 8, w // 8, w % 8))
                elif w < 96:
                    w -= 64
                    self.weights[-1].append((neg, 7, 4 + (w // 8), w % 8))
                elif w < 112:
                    w -= 96
                    self.weights[-1].append((neg, 5, 2 + (w // 8), w % 8))
                elif w < 120:
                    w -= 112
                    self.weights[-1].append((neg, 3, 1 + (w // 8), w % 8))
                else:
                    w -= 120
                    self.weights[-1].append((neg, 2, 1 + (w // 8), w % 8))
        self.input_size = input_size
        self.output_size = output_size
        self.relu = relu    

    def forward(self, input):
        # Output vector
        output = []

        # Copy of input vector to be sign-extended to 28 bits from 24 bits.
        extended_input = input[:]

        # Sign-extend input 24->28 bits to prevent multiplication overflow.
        for j in range(self.input_size):
            extended_input[j] &= ACTIVATION_MASK
            # If number is negative, add 1s to the left to preserve the sign when we later mask back down to 24 bits after multiplication.
            if extended_input[j] > ACTIVATION_MASK // 2:
                extended_input[j] += ((1 << MATMUL_EXTRA_PRECISION) - 1) << ACTIVATION_SIZE

        # Perform the matrix multiplication using the custom weight encoding and fixed-point arithmetic.
        for i in range(self.output_size):
            cur = 0
            for j in range(self.input_size):
                w = self.weights[i][j]
                big = (extended_input[j] * w[2]) & MATMUL_BIG_MASK
                if big > (MATMUL_BIG_MASK // 2):
                    big += 255 << (MATMUL_EXTRA_PRECISION + ACTIVATION_SIZE)
                small = (extended_input[j] * w[3]) & MATMUL_BIG_MASK
                if small > (MATMUL_BIG_MASK // 2):
                    small += 255 << (MATMUL_EXTRA_PRECISION + ACTIVATION_SIZE)
                cont = (big >> w[1]) + (small >> (w[1] + 3))    # apply the shifts indicated by the weight encoding
                cont &= ACTIVATION_MASK                        # mask back down to 24 bits
                if w[0]:    # apply sign
                    cont = (-cont) & ACTIVATION_MASK
                cur += cont             # accumulate the contributions from each entry of input vector.
                cur &= ACTIVATION_MASK
            if self.relu and cur > (ACTIVATION_MASK // 2):     # if relu is enabled, set negative outputs to 0
                output.append(0)
            else:
                output.append(cur)
        return output


class LayerNorm:
    """
    Layer normalization with a learned per-dimension scale (gamma) and shift (beta).

    Given input x of length EMBED_SIZE, computes:
        output[i] = gamma[i] * (x[i] - mean(x)) / std(x) + beta[i]

    All arithmetic uses 24-bit fixed-point with 18 fractional bits.
    Division by N and sqrt are replaced by multiply-then-shift using
    precomputed constants. The sqrt itself would be a lookup table in
    Redstone; here we use math.sqrt to emulate that.

    Weight file format:
        EMBED_SIZE * 3 bytes for gamma, then EMBED_SIZE * 3 bytes for beta,
        all little-endian unsigned integers.
        Each gamma value encodes:  file_value = round(gamma * 2^22)
        Dividing by 2 on load gives the internal representation gamma * 2^21,
        which aligns with the >> 21 shift applied in forward().
    """

    def __init__(self, index: int):
        self.weights: list[int] = []
        self.shift: list[int] = []
        with open(f"{WEIGHTS_PATH}/layernorm/ln_{index}.bin", "rb") as f:
            for _ in range(EMBED_SIZE):
                raw = int.from_bytes(f.read(3), byteorder="little")
                self.weights.append(raw // 2)  # internal: gamma * 2^21
            for _ in range(EMBED_SIZE):
                self.shift.append(int.from_bytes(f.read(3), byteorder="little"))

    def forward(self, input: list[int]) -> list[int]:
        # ------------------------------------------------------------------
        # Step 1: Compute the mean
        # Sign-extend each 24-bit value to 32 bits before accumulating so
        # that negative values don't inflate the sum.
        # ------------------------------------------------------------------
        total = 0
        for value in input:
            sign_extended = value + (255 << ACTIVATION_SIZE) if value > ACTIVATION_MASK // 2 else 0
            total += sign_extended
        total &= (1 << 32) - 1
        
        # If mean represents a negative number, take 2's complement (i.e. absolute value)
        negative_mean = total >= (1 << (ACTIVATION_SIZE + 7))
        if negative_mean:
            total = (-total) & ((1 << (ACTIVATION_SIZE + 7)) - 1)
        # Integer approximation of total / EMBED_SIZE (division is because we are taking the average)
        mean = (total * LAYERNORM_AVG_CONST) >> 32
        # Negate and restore back to 24 bits. If the total was positive then the mean is already fitting into 24 bits.
        if negative_mean:
            mean = (-mean) & ACTIVATION_MASK


        # ------------------------------------------------------------------
        # Step 2: Compute the standard deviation
        # Accumulate sum of squared deviations: sum((x[i] - mean)^2).
        # Diffs are in Q18 so their squares are in Q36; EPS is pre-scaled to Q36.
        # ------------------------------------------------------------------
        variance_acc = EPS
        for value in input:
            diff = (value - mean) & ACTIVATION_MASK
            if diff > ACTIVATION_MASK // 2:
                diff = (-diff) & (ACTIVATION_MASK // 2)
            variance_acc += diff * diff
            # Q36 is 48 bits in total (6*2)
            variance_acc &= (1 << 48) - 1

        # sqrt gives sigma in Q18 space (i.e. real_sigma * 2^18).
        # In Redstone this step is a lookup table; math.sqrt emulates it.
        sigma = int(sqrt(variance_acc))

        # 27 bit seems to not lose much precision.
        sigma = (LAYERNORM_VAR_CONST * sigma) >> 27
        sigma &= ACTIVATION_MASK

        # Calculation of 1/sigma in Q18
        inv_sigma = ((1 << (2 * MATMUL_FIXED_POINT)) // sigma) & ACTIVATION_MASK


        # ------------------------------------------------------------------
        # Step 3: Normalize, apply the learned scale (gamma), then shift (beta)
        # ------------------------------------------------------------------
        result: list[int] = []
        for i, value in enumerate(input):
            diff = (value - mean) & ACTIVATION_MASK

            negative = diff > ACTIVATION_MASK // 2
            if negative:
                diff = (-diff) & (ACTIVATION_MASK // 2)

            x_hat = ((diff * inv_sigma) >> MATMUL_FIXED_POINT) & (ACTIVATION_MASK // 2)

            out = ((x_hat * self.weights[i]) >> (MATMUL_FIXED_POINT + 3)) & (ACTIVATION_MASK // 2)

            if negative:
                out = (-out) & ACTIVATION_MASK
            out = (out + self.shift[i]) & ACTIVATION_MASK
            result.append(out)

        return result


class MLP:
    def __init__(self, block_num):
        weights_up = [[] for _ in range(FFN_SCALING * EMBED_SIZE)]
        weights_down = [[] for _ in range(EMBED_SIZE)]
        with open(f"{WEIGHTS_PATH}/mlp/mlp_{block_num}_up.bin", "rb") as f:
            for i in range(FFN_SCALING * EMBED_SIZE):
                weights_up[i] = list(f.read(EMBED_SIZE))
        with open(f"{WEIGHTS_PATH}/mlp/mlp_{block_num}_down.bin", "rb") as f:
            for i in range(EMBED_SIZE):
                weights_down[i] = list(f.read(FFN_SCALING * EMBED_SIZE))
        self.matmul_up = MatMul(weights_up, EMBED_SIZE, FFN_SCALING * EMBED_SIZE, relu=False)
        self.matmul_down = MatMul(weights_down, FFN_SCALING * EMBED_SIZE, EMBED_SIZE)
        self.bias_up = []
        with open(f"{WEIGHTS_PATH}/mlp/mlp_{block_num}_up.bias", "rb") as f:
            for _ in range(FFN_SCALING * EMBED_SIZE):
                self.bias_up.append(int.from_bytes(f.read(3), byteorder="little"))
        self.bias_down = []
        with open(f"{WEIGHTS_PATH}/mlp/mlp_{block_num}_down.bias", "rb") as f:
            for _ in range(EMBED_SIZE):
                self.bias_down.append(int.from_bytes(f.read(3), byteorder="little"))

    def forward(self, input):
        res = self.matmul_up.forward(input)
        for i in range(FFN_SCALING * EMBED_SIZE):
            res[i] = (res[i] + self.bias_up[i]) & ACTIVATION_MASK
            if res[i] > ACTIVATION_MASK // 2:  # ReLU after bias
                res[i] = 0
        res = self.matmul_down.forward(res)
        for i in range(EMBED_SIZE):
            res[i] = (res[i] + self.bias_down[i]) & ACTIVATION_MASK
        return res


class Attention:
    def __init__(self, block_num):
        self.block_num = block_num
        key   = [[[] for _ in range(HEAD_SIZE)] for _ in range(HEADS)]
        value = [[[] for _ in range(HEAD_SIZE)] for _ in range(HEADS)]
        query = [[[] for _ in range(HEAD_SIZE)] for _ in range(HEADS)]
        proj  = [[] for _ in range(EMBED_SIZE)]

        for head in range(HEADS):
            with open(f"{WEIGHTS_PATH}/attention/att_{block_num}_h{head}_key.bin", "rb") as f:
                for i in range(HEAD_SIZE):
                    key[head][i] = list(f.read(EMBED_SIZE))
            with open(f"{WEIGHTS_PATH}/attention/att_{block_num}_h{head}_value.bin", "rb") as f:
                for i in range(HEAD_SIZE):
                    value[head][i] = list(f.read(EMBED_SIZE))
            with open(f"{WEIGHTS_PATH}/attention/att_{block_num}_h{head}_query.bin", "rb") as f:
                for i in range(HEAD_SIZE):
                    query[head][i] = list(f.read(EMBED_SIZE))

        with open(f"{WEIGHTS_PATH}/attention/att_{block_num}_proj.bin", "rb") as f:
            for i in range(EMBED_SIZE):
                proj[i] = list(f.read(EMBED_SIZE))

        self.matmul_key   = [MatMul(key[h],   EMBED_SIZE, HEAD_SIZE) for h in range(HEADS)]
        self.matmul_value = [MatMul(value[h], EMBED_SIZE, HEAD_SIZE) for h in range(HEADS)]
        self.matmul_query = [MatMul(query[h], EMBED_SIZE, HEAD_SIZE) for h in range(HEADS)]
        self.matmul_proj  = MatMul(proj, EMBED_SIZE, EMBED_SIZE)

        self.bias_proj = []
        with open(f"{WEIGHTS_PATH}/attention/att_{block_num}_proj.bias", "rb") as f:
            for _ in range(EMBED_SIZE):
                self.bias_proj.append(int.from_bytes(f.read(3), byteorder="little"))

        self.softmax_exp = []
        with open(f"{WEIGHTS_PATH}/softmax.bin", "rb") as f:
            for _ in range(1024):
                self.softmax_exp.append(int.from_bytes(f.read(3), byteorder="little"))

        self.k_cache = [[] for _ in range(HEADS)]
        self.v_cache = [[] for _ in range(HEADS)]

    def to_float16(self, value, offset=0):
        neg = False
        if value > ACTIVATION_MASK // 2:
            neg = True
            value = (-value) & (ACTIVATION_MASK // 2)
        for i in range(ACTIVATION_SIZE - 1, -1, -1):
            if ((value >> i) & 1) > 0:
                res = ((value << (ACTIVATION_SIZE - i)) >> 14) & ((1 << 10) - 1)
                res += (i + 9 - offset) << 10
                res += int(neg) << 15
                return res
        return 0

    def float_mult(self, a, b, shift=0):
        neg = False
        offset = ((a >> 10) & 31) + ((b >> 10) & 31)
        if a >= (1 << 15):
            neg = not neg
            a -= (1 << 15)
        if b >= (1 << 15):
            neg = not neg
            b -= (1 << 15)
        if a > 0:
            a = (a & ((1 << 10) - 1)) + (1 << 10)
        if b > 0:
            b = (b & ((1 << 10) - 1)) + (1 << 10)
        res = ((a * b) << offset) >> (56 + shift)
        res = res & ACTIVATION_MASK
        if neg:
            res = (-res) & ACTIVATION_MASK
        return res

    def undo_last(self):
        for i in range(HEADS):
            self.k_cache[i].pop()
            self.v_cache[i].pop()

    def forward(self, input):
        proj_input = []
        for head in range(HEADS):
            keys = self.matmul_key[head].forward(input)
            keys = list(map(self.to_float16, keys))
            self.k_cache[head].append(keys)

            values = self.matmul_value[head].forward(input)
            values = list(map(self.to_float16, values))
            self.v_cache[head].append(values)

            queries = self.matmul_query[head].forward(input)
            queries = list(map(self.to_float16, queries))

            relevance = [0] * len(self.k_cache[head])
            for i, k in enumerate(self.k_cache[head]):
                for j, q in enumerate(queries):
                    relevance[i] += self.float_mult(k[j], q, 5)
                    relevance[i] &= ACTIVATION_MASK

            biggest = 0
            for i in range(len(relevance)):
                neg = False
                if relevance[i] > (ACTIVATION_MASK // 2):
                    neg = True
                    relevance[i] = (-relevance[i]) & (ACTIVATION_MASK // 2)
                relevance[i] = ((relevance[i] * ATT_CONST) >> 23) & (ACTIVATION_MASK // 2)
                if neg:
                    relevance[i] = (-relevance[i]) & ACTIVATION_MASK
                relevance[i] ^= (1 << (ACTIVATION_SIZE - 1))
                biggest = max(biggest, relevance[i])

            output = [0] * HEAD_SIZE
            softmax_sum = 0
            for i in range(len(relevance)):
                power = (biggest - relevance[i]) >> 10
                res = 0 if power >= 1024 else self.softmax_exp[power]
                softmax_sum += res
            softmax_sum &= ACTIVATION_MASK
            softmax_sum = (1 << 39) // softmax_sum

            for i in range(len(relevance)):
                power = (biggest - relevance[i]) >> 10
                res = 0 if power >= 1024 else self.softmax_exp[power]
                res = ((softmax_sum * res) >> 17) & (ACTIVATION_MASK // 2)
                res = self.to_float16(res, offset=4)
                for j, v in enumerate(self.v_cache[head][i]):
                    output[j] += self.float_mult(res, v)
                    output[j] &= ACTIVATION_MASK

            proj_input += output

        res = self.matmul_proj.forward(proj_input)
        for i in range(EMBED_SIZE):
            res[i] = (res[i] + self.bias_proj[i]) & ACTIVATION_MASK
        return res


class Block:
    """
    A single transformer block, consisting of layer normalization, multi-head attention, another layer normalization, and an MLP.
    The forward pass applies these components in sequence, with residual connections after the attention and MLP 
    (i.e. input[i] + att_diff[i])
    """
    def __init__(self, block_num):
        self.ln_1 = LayerNorm(2 * block_num + 1)
        self.att  = Attention(block_num)
        self.ln_2 = LayerNorm(2 * block_num + 2)
        self.mlp  = MLP(block_num)

    def forward(self, input):
        att_diff = self.att.forward(self.ln_1.forward(input))
        for i in range(EMBED_SIZE):
            input[i] = (input[i] + att_diff[i]) & ACTIVATION_MASK
        mlp_diff = self.mlp.forward(self.ln_2.forward(input))
        for i in range(EMBED_SIZE):
            input[i] = (input[i] + mlp_diff[i]) & ACTIVATION_MASK
        return input


class Embedding:
    def __init__(self):
        self.wte = []
        with open(f"{WEIGHTS_PATH}/embedding/wte.bin", "rb") as f:
            for _ in range(VOCAB_SIZE):
                row = []
                for _ in range(EMBED_SIZE):
                    cur = int.from_bytes(f.read(3), byteorder="little")
                    if cur >= (1 << 17):
                        cur |= (1 << 18) * ((1 << 6) - 1)
                    row.append(cur)
                self.wte.append(row)

        self.wpe = []
        with open(f"{WEIGHTS_PATH}/embedding/wpe.bin", "rb") as f:
            for _ in range(CONTEXT):
                row = []
                for _ in range(EMBED_SIZE):
                    cur = int.from_bytes(f.read(3), byteorder="little")
                    if cur >= (1 << 17):
                        cur |= (1 << 18) * ((1 << 6) - 1)
                    row.append(cur)
                self.wpe.append(row)

    def get_weights(self, token, pos=-1):
        weights = self.wte[token].copy()
        if pos != -1:
            assert 0 <= pos < CONTEXT
            for i in range(EMBED_SIZE):
                weights[i] = (weights[i] + self.wpe[pos][i]) & ACTIVATION_MASK
        return weights


class Unembedding:
    def __init__(self):
        weights = [[] for _ in range(VOCAB_SIZE)]
        with open(f"{WEIGHTS_PATH}/unembedding/lm_head.bin", "rb") as f:
            for i in range(VOCAB_SIZE):
                weights[i] = list(f.read(EMBED_SIZE))
        self.lm_head = MatMul(weights, EMBED_SIZE, VOCAB_SIZE)

        self.softmax_exp = []
        with open(f"{WEIGHTS_PATH}/softmax_2.bin", "rb") as f:
            for _ in range(1024):
                self.softmax_exp.append(int.from_bytes(f.read(3), byteorder="little"))

    def forward(self, input):
        logits = self.lm_head.forward(input)
        biggest = 0
        for i in range(VOCAB_SIZE):
            logits[i] ^= (1 << (ACTIVATION_SIZE - 1))
            biggest = max(biggest, logits[i])
        softmax_sum = 0
        for i in range(VOCAB_SIZE):
            power = (biggest - logits[i]) >> 12
            res = 0 if power >= 1024 else self.softmax_exp[power]
            softmax_sum += res
        softmax_sum &= ((1 << 32) - 1)
        softmax_sum = (1 << 46) // softmax_sum
        output = [0] * OUTPUT_SIZE
        for i in range(VOCAB_SIZE):
            power = (biggest - logits[i]) >> 12
            res = 0 if power >= 1024 else self.softmax_exp[power]
            res = ((softmax_sum * res) >> 23) & ACTIVATION_MASK
            res = (1 << 11) * res + i
            for j in range(OUTPUT_SIZE):
                if res > output[j]:
                    res, output[j] = output[j], res
        return output


class PRNG:
    def __init__(self, seed):
        self.seed = seed

    def next(self):
        for i in range(256):
            next_bit = ((self.seed >> 22) & 1) ^ ((self.seed >> 17) & 1)
            self.seed <<= 1
            self.seed &= ((1 << 23) - 1)
            self.seed += next_bit
        
        return self.seed


class Model:
    def __init__(self):
        self.embedding   = Embedding()
        self.transformer = [Block(i) for i in range(LAYERS)]
        self.ln_f        = LayerNorm(2 * LAYERS + 1)  # ln_13
        self.unembedding = Unembedding()
        print("Model loaded.")
        self.index = 0

    def process(self, token):
        value = self.embedding.get_weights(token, self.index)
        for block in self.transformer:
            value = block.forward(value)
        value = self.ln_f.forward(value)
        ans = self.unembedding.forward(value)
        self.index += 1
        return ans

    def undo_last(self):
        self.index -= 1
        for block in self.transformer:
            block.att.undo_last()


def sample_pixel(top_k, rng):
    """Sample a patch token from the top SAMPLE_TOP_K highest-probability entries.

    top_k is sorted ascending so the highest-probability tokens are at the end.
    """
    cur = rng.next()
    pixel_tokens = [j for j in range(OUTPUT_SIZE) if (top_k[j] & 2047) >= PIXEL_START_ID]
    for j in pixel_tokens[-SAMPLE_TOP_K:][::-1]:
        cur -= (top_k[j] >> 11)
        if cur < 0:
            return top_k[j] & 2047
    # fallback: highest-probability pixel token
    return top_k[pixel_tokens[-1]] & 2047


def decode_patches(patch_values):
    """Convert a list of 64 patch token values (0-15) into a flat 16×16 pixel list (0/1)."""
    p = IMG_SIZE // PATCH_SIZE
    grid = [[0] * IMG_SIZE for _ in range(IMG_SIZE)]
    for idx, v in enumerate(patch_values):
        pr, pc = divmod(idx, p)
        for pi in range(PATCH_SIZE):
            for pj in range(PATCH_SIZE):
                bit_pos = PIXELS_PER_PATCH - 1 - (pi * PATCH_SIZE + pj)
                grid[pr * PATCH_SIZE + pi][pc * PATCH_SIZE + pj] = (v >> bit_pos) & 1
    return [pixel for row in grid for pixel in row]


def print_image(pixels):
    """Print a flat list of 256 pixel values (0/1) as 16×16 ASCII art."""
    for row in range(IMG_SIZE):
        print("".join("#" if pixels[row * IMG_SIZE + col] else "." for col in range(IMG_SIZE)))


def run_model():
    vocab = []
    with open("models/vocab.txt", "r") as f:
        for line in f.readlines():
            vocab.append(line.strip())
    word_to_id = {w: i for i, w in enumerate(vocab)}

    model = Model()
    seed = int(input("Enter RNG seed: "))
    rng = PRNG(seed)

    while True:
        word = input("Enter a word: ").strip().lower()
        if word not in word_to_id:
            print(f"Unknown word '{word}'. Available: {', '.join(vocab[:10])}...")
            continue

        word_id = word_to_id[word]
        model.index = 0
        for block in model.transformer:
            block.att.k_cache = [[] for _ in range(HEADS)]
            block.att.v_cache = [[] for _ in range(HEADS)]

        print(f"Generating image for '{word}'...")
        model.process(word_id)

        patches = []
        nxt = PIXEL_START_ID  # start by predicting first patch token
        num_patches = (IMG_SIZE // PATCH_SIZE) ** 2
        for _ in tqdm(range(num_patches), desc="Generating", unit="patch"):
            top_k = model.process(nxt)
            nxt = sample_pixel(top_k, rng)
            patches.append(nxt - PIXEL_START_ID)

        print_image(decode_patches(patches))


if __name__ == "__main__":
    run_model()
