from math import sqrt

LAYERS = 6
HEADS = 8
MLP_SCALE = 4
EMBED_SIZE = 256
HEAD_SIZE = EMBED_SIZE // HEADS  # 32
VOCAB_SIZE = 347
PIXEL_START_ID = 345   # token IDs 345 and 346 are the two pixel values (0 and 1)
OUTPUT_SIZE = 16
CONTEXT = 785          # 1 word token + 784 pixel tokens (28x28)
IMG_SIZE = 28

FIXED_POINT_SIZE = 24
FIXED_POINT_MASK = (1 << FIXED_POINT_SIZE) - 1
MATMUL_FIXED_POINT = 18
MATMUL_EXTRA_PRECISION = 4
MATMUL_BIG_MASK = (1 << (FIXED_POINT_SIZE + MATMUL_EXTRA_PRECISION)) - 1

LAYERNORM_CONST = int((1 << 32) / EMBED_SIZE)
LAYERNORM_CONST_2 = int((1 << 27) / sqrt(EMBED_SIZE))
ATT_CONST = int((1 << 26) / sqrt(HEAD_SIZE))

EPS = int(1e-5 * EMBED_SIZE * (1 << (2 * MATMUL_FIXED_POINT)))


class MatMul:
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
        output = []
        normed = input[:]
        for j in range(self.input_size):
            normed[j] &= FIXED_POINT_MASK
            if normed[j] > FIXED_POINT_MASK // 2:
                normed[j] += ((1 << MATMUL_EXTRA_PRECISION) - 1) << FIXED_POINT_SIZE
        for i in range(self.output_size):
            cur = 0
            for j in range(self.input_size):
                w = self.weights[i][j]
                big = (normed[j] * w[2]) & MATMUL_BIG_MASK
                if big > (MATMUL_BIG_MASK // 2):
                    big += 255 << (MATMUL_EXTRA_PRECISION + FIXED_POINT_SIZE)
                small = (normed[j] * w[3]) & MATMUL_BIG_MASK
                if small > (MATMUL_BIG_MASK // 2):
                    small += 255 << (MATMUL_EXTRA_PRECISION + FIXED_POINT_SIZE)
                cont = (big >> w[1]) + (small >> (w[1] + 3))
                cont &= FIXED_POINT_MASK
                if w[0]:
                    cont = (-cont) & FIXED_POINT_MASK
                cur += cont
                cur &= FIXED_POINT_MASK
            if self.relu and cur > (FIXED_POINT_MASK // 2):
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
        with open(f"weights2/weight_files/layernorm/ln_{index}.bin", "rb") as f:
            for _ in range(EMBED_SIZE):
                raw = int.from_bytes(f.read(3), byteorder="little")
                self.weights.append(raw // 2)  # internal: gamma * 2^21
            for _ in range(EMBED_SIZE):
                self.shift.append(int.from_bytes(f.read(3), byteorder="little"))

    def forward(self, x: list[int]) -> list[int]:
        # ------------------------------------------------------------------
        # Step 1: Compute the mean
        # Sign-extend each 24-bit value to 32 bits before accumulating so
        # that negative values don't inflate the sum.
        # ------------------------------------------------------------------
        total = 0
        for v in x:
            sign_extension = (255 << FIXED_POINT_SIZE) if v > FIXED_POINT_MASK // 2 else 0
            total += v + sign_extension
        total &= (1 << 32) - 1

        negative_mean = total >= (1 << (FIXED_POINT_SIZE + 7))
        if negative_mean:
            total = (-total) & ((1 << (FIXED_POINT_SIZE + 7)) - 1)
        mean = (total * LAYERNORM_CONST) >> 32
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
            if diff > FIXED_POINT_MASK // 2:
                diff = (-diff) & (FIXED_POINT_MASK // 2)
            variance_acc += diff * diff
            variance_acc &= (1 << 48) - 1

        # sqrt gives sigma in Q18 space (i.e. real_sigma * 2^18).
        # In Redstone this step is a lookup table; math.sqrt emulates it.
        sigma = int(sqrt(variance_acc))

        sigma = (LAYERNORM_CONST_2 * sigma) >> 27
        sigma &= FIXED_POINT_MASK

        inv_sigma = ((1 << (2 * MATMUL_FIXED_POINT)) // sigma) & FIXED_POINT_MASK

        # ------------------------------------------------------------------
        # Step 3: Normalize, apply the learned scale (gamma), then shift (beta)
        # ------------------------------------------------------------------
        result: list[int] = []
        for i, v in enumerate(x):
            diff = (v - mean) & FIXED_POINT_MASK

            negative = diff > FIXED_POINT_MASK // 2
            if negative:
                diff = (-diff) & (FIXED_POINT_MASK // 2)

            x_hat = ((diff * inv_sigma) >> MATMUL_FIXED_POINT) & (FIXED_POINT_MASK // 2)

            out = ((x_hat * self.weights[i]) >> (MATMUL_FIXED_POINT + 3)) & (FIXED_POINT_MASK // 2)

            if negative:
                out = (-out) & FIXED_POINT_MASK
            out = (out + self.shift[i]) & FIXED_POINT_MASK
            result.append(out)

        return result


class MLP:
    def __init__(self, block_num):
        weights_up = [[] for _ in range(MLP_SCALE * EMBED_SIZE)]
        weights_down = [[] for _ in range(EMBED_SIZE)]
        with open(f"weights2/weight_files/mlp/mlp_{block_num}_up.bin", "rb") as f:
            for i in range(MLP_SCALE * EMBED_SIZE):
                weights_up[i] = list(f.read(EMBED_SIZE))
        with open(f"weights2/weight_files/mlp/mlp_{block_num}_down.bin", "rb") as f:
            for i in range(EMBED_SIZE):
                weights_down[i] = list(f.read(MLP_SCALE * EMBED_SIZE))
        self.matmul_up = MatMul(weights_up, EMBED_SIZE, MLP_SCALE * EMBED_SIZE, relu=False)
        self.matmul_down = MatMul(weights_down, MLP_SCALE * EMBED_SIZE, EMBED_SIZE)
        self.bias_up = []
        with open(f"weights2/weight_files/mlp/mlp_{block_num}_up.bias", "rb") as f:
            for _ in range(MLP_SCALE * EMBED_SIZE):
                self.bias_up.append(int.from_bytes(f.read(3), byteorder="little"))
        self.bias_down = []
        with open(f"weights2/weight_files/mlp/mlp_{block_num}_down.bias", "rb") as f:
            for _ in range(EMBED_SIZE):
                self.bias_down.append(int.from_bytes(f.read(3), byteorder="little"))

    def forward(self, input):
        res = self.matmul_up.forward(input)
        for i in range(MLP_SCALE * EMBED_SIZE):
            res[i] = (res[i] + self.bias_up[i]) & FIXED_POINT_MASK
            if res[i] > FIXED_POINT_MASK // 2:  # ReLU after bias
                res[i] = 0
        res = self.matmul_down.forward(res)
        for i in range(EMBED_SIZE):
            res[i] = (res[i] + self.bias_down[i]) & FIXED_POINT_MASK
        return res


class Attention:
    def __init__(self, block_num):
        self.block_num = block_num
        key   = [[[] for _ in range(HEAD_SIZE)] for _ in range(HEADS)]
        value = [[[] for _ in range(HEAD_SIZE)] for _ in range(HEADS)]
        query = [[[] for _ in range(HEAD_SIZE)] for _ in range(HEADS)]
        proj  = [[] for _ in range(EMBED_SIZE)]

        for head in range(HEADS):
            with open(f"weights2/weight_files/attention/att_{block_num}_h{head}_key.bin", "rb") as f:
                for i in range(HEAD_SIZE):
                    key[head][i] = list(f.read(EMBED_SIZE))
            with open(f"weights2/weight_files/attention/att_{block_num}_h{head}_value.bin", "rb") as f:
                for i in range(HEAD_SIZE):
                    value[head][i] = list(f.read(EMBED_SIZE))
            with open(f"weights2/weight_files/attention/att_{block_num}_h{head}_query.bin", "rb") as f:
                for i in range(HEAD_SIZE):
                    query[head][i] = list(f.read(EMBED_SIZE))

        with open(f"weights2/weight_files/attention/att_{block_num}_proj.bin", "rb") as f:
            for i in range(EMBED_SIZE):
                proj[i] = list(f.read(EMBED_SIZE))

        self.matmul_key   = [MatMul(key[h],   EMBED_SIZE, HEAD_SIZE) for h in range(HEADS)]
        self.matmul_value = [MatMul(value[h], EMBED_SIZE, HEAD_SIZE) for h in range(HEADS)]
        self.matmul_query = [MatMul(query[h], EMBED_SIZE, HEAD_SIZE) for h in range(HEADS)]
        self.matmul_proj  = MatMul(proj, EMBED_SIZE, EMBED_SIZE)

        self.bias_proj = []
        with open(f"weights2/weight_files/attention/att_{block_num}_proj.bias", "rb") as f:
            for _ in range(EMBED_SIZE):
                self.bias_proj.append(int.from_bytes(f.read(3), byteorder="little"))

        self.softmax_exp = []
        with open("weights2/weight_files/softmax.bin", "rb") as f:
            for _ in range(1024):
                self.softmax_exp.append(int.from_bytes(f.read(3), byteorder="little"))

        self.k_cache = [[] for _ in range(HEADS)]
        self.v_cache = [[] for _ in range(HEADS)]

    def to_float16(self, value, offset=0):
        neg = False
        if value > FIXED_POINT_MASK // 2:
            neg = True
            value = (-value) & (FIXED_POINT_MASK // 2)
        for i in range(FIXED_POINT_SIZE - 1, -1, -1):
            if ((value >> i) & 1) > 0:
                res = ((value << (FIXED_POINT_SIZE - i)) >> 14) & ((1 << 10) - 1)
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
        res = res & FIXED_POINT_MASK
        if neg:
            res = (-res) & FIXED_POINT_MASK
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
                    relevance[i] &= FIXED_POINT_MASK

            biggest = 0
            for i in range(len(relevance)):
                neg = False
                if relevance[i] > (FIXED_POINT_MASK // 2):
                    neg = True
                    relevance[i] = (-relevance[i]) & (FIXED_POINT_MASK // 2)
                relevance[i] = ((relevance[i] * ATT_CONST) >> 23) & (FIXED_POINT_MASK // 2)
                if neg:
                    relevance[i] = (-relevance[i]) & FIXED_POINT_MASK
                relevance[i] ^= (1 << (FIXED_POINT_SIZE - 1))
                biggest = max(biggest, relevance[i])

            output = [0] * HEAD_SIZE
            softmax_sum = 0
            for i in range(len(relevance)):
                power = (biggest - relevance[i]) >> 10
                res = 0 if power >= 1024 else self.softmax_exp[power]
                softmax_sum += res
            softmax_sum &= FIXED_POINT_MASK
            softmax_sum = (1 << 39) // softmax_sum

            for i in range(len(relevance)):
                power = (biggest - relevance[i]) >> 10
                res = 0 if power >= 1024 else self.softmax_exp[power]
                res = ((softmax_sum * res) >> 17) & (FIXED_POINT_MASK // 2)
                res = self.to_float16(res, offset=4)
                for j, v in enumerate(self.v_cache[head][i]):
                    output[j] += self.float_mult(res, v)
                    output[j] &= FIXED_POINT_MASK

            proj_input += output

        res = self.matmul_proj.forward(proj_input)
        for i in range(EMBED_SIZE):
            res[i] = (res[i] + self.bias_proj[i]) & FIXED_POINT_MASK
        return res


class Block:
    def __init__(self, block_num):
        self.ln_1 = LayerNorm(2 * block_num + 1)
        self.att  = Attention(block_num)
        self.ln_2 = LayerNorm(2 * block_num + 2)
        self.mlp  = MLP(block_num)

    def forward(self, input):
        att_diff = self.att.forward(self.ln_1.forward(input))
        for i in range(EMBED_SIZE):
            input[i] = (input[i] + att_diff[i]) & FIXED_POINT_MASK
        mlp_diff = self.mlp.forward(self.ln_2.forward(input))
        for i in range(EMBED_SIZE):
            input[i] = (input[i] + mlp_diff[i]) & FIXED_POINT_MASK
        return input


class Embedding:
    def __init__(self):
        self.wte = []
        with open("weights2/weight_files/embedding/wte.bin", "rb") as f:
            for _ in range(VOCAB_SIZE):
                row = []
                for _ in range(EMBED_SIZE):
                    cur = int.from_bytes(f.read(3), byteorder="little")
                    if cur >= (1 << 17):
                        cur |= (1 << 18) * ((1 << 6) - 1)
                    row.append(cur)
                self.wte.append(row)

        self.wpe = []
        with open("weights2/weight_files/embedding/wpe.bin", "rb") as f:
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
                weights[i] = (weights[i] + self.wpe[pos][i]) & FIXED_POINT_MASK
        return weights


class Unembedding:
    def __init__(self):
        weights = [[] for _ in range(VOCAB_SIZE)]
        with open("weights2/weight_files/unembedding/lm_head.bin", "rb") as f:
            for i in range(VOCAB_SIZE):
                weights[i] = list(f.read(EMBED_SIZE))
        self.lm_head = MatMul(weights, EMBED_SIZE, VOCAB_SIZE)

        self.softmax_exp = []
        with open("weights2/weight_files/softmax_2.bin", "rb") as f:
            for _ in range(1024):
                self.softmax_exp.append(int.from_bytes(f.read(3), byteorder="little"))

    def forward(self, input):
        logits = self.lm_head.forward(input)
        biggest = 0
        for i in range(VOCAB_SIZE):
            logits[i] ^= (1 << (FIXED_POINT_SIZE - 1))
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
            res = ((softmax_sum * res) >> 23) & FIXED_POINT_MASK
            res = (1 << 11) * res + i
            for j in range(OUTPUT_SIZE):
                if res > output[j]:
                    res, output[j] = output[j], res
        return output


class PRNG:
    def __init__(self, seed):
        self.seed = seed

    def next(self):
        for _ in range(256):
            next_bit = ((self.seed >> 22) & 1) ^ ((self.seed >> 17) & 1)
            self.seed = ((self.seed << 1) & ((1 << 23) - 1)) + next_bit
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
    """Sample a pixel token (PIXEL_START_ID or PIXEL_START_ID+1) from top-k output."""
    cur = rng.next()
    for j in range(OUTPUT_SIZE - 1, -1, -1):
        token_id = top_k[j] & 2047
        if token_id < PIXEL_START_ID:
            continue
        cur -= (top_k[j] >> 11)
        if cur < 0:
            return token_id
    # fallback: return whichever pixel token has the highest probability
    best = max(
        (top_k[j] for j in range(OUTPUT_SIZE) if (top_k[j] & 2047) >= PIXEL_START_ID),
        default=top_k[0],
    )
    return best & 2047


def print_image(pixels):
    """Print a flat list of 784 pixel values (0/1) as 28x28 ASCII art."""
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

        pixels = []
        nxt = PIXEL_START_ID  # start by predicting first pixel
        for _ in range(IMG_SIZE * IMG_SIZE):
            top_k = model.process(nxt)
            nxt = sample_pixel(top_k, rng)
            pixels.append(nxt - PIXEL_START_ID)

        print_image(pixels)


if __name__ == "__main__":
    run_model()
