// baseline JPEG のエンコーダ（4:4:4）。
//
// **canvas の `toBlob('image/jpeg', q)` を使えない。** あれは色差を 4:2:0 に間引く実装が普通で、
// 止める手段が無い。デスクトップ側（1-10/2608_Obayashi_GNMHeadExporter の
// `infrastructure/imaging.py`）は `subsampling=0` を明示していて、理由も書いてある:
//
// > 既定の 4:2:0 は色差を半分に落とすので、アトラスの chart の境界や髪の縁で色がにじむ。にじんだ色は
// > Unity 側で継ぎ目に見えるが、原因がベイクなのか JPEG なのかを後から切り分けられない。
//
// つまり 4:2:0 で出すと**あちらが消した不具合を出力に戻す**。自分で書く方を採る。
//
// ## どこまで揃うか
//
// 揃うもの: 色差を間引かない（3 成分すべて 1x1 標本）/ 量子化表（Annex K を libjpeg と同じ式で品質
// スケール）/ Huffman 表（Annex K の標準表）/ マーカーの構成。
//
// 揃わないもの: DCT の実装。libjpeg は既定でスケール付き整数 DCT を使うが、こちらは浮動小数の
// 分離型 DCT。**同じ入力でもバイト列は一致しない**（画素値の差は丸めの 1 未満）。バイト一致は
// もともと 4:2:0 の時点で失われていたので、ここで戻すのは「色差の解像度」であって同一性ではない。
//
// baseline（progressive でない）・8bit・成分 3 つだけを書く。それ以外の JPEG は書かないので、
// 汎用エンコーダとしては使わない。

/** 量子化表の元（ITU-T T.81 Annex K.1、輝度）。 */
const BASE_LUMINANCE_TABLE = [
  16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
  14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
  92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

/** 同 Annex K.2、色差。 */
const BASE_CHROMINANCE_TABLE = [
  17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
  47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

const ZIGZAG = [
  0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
  13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52,
  45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

// --- Annex K の標準 Huffman 表 ---------------------------------------------
const DC_LUMINANCE_BITS = [0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const DC_LUMINANCE_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const DC_CHROMINANCE_BITS = [0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
const DC_CHROMINANCE_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

const AC_LUMINANCE_BITS = [0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];
const AC_LUMINANCE_VALUES = [
  0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
  0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
  0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
  0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
  0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
  0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
  0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
  0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
  0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
  0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
  0xf9, 0xfa,
];

const AC_CHROMINANCE_BITS = [0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
const AC_CHROMINANCE_VALUES = [
  0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
  0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
  0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
  0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
  0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
  0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
  0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
  0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
  0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
  0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
  0xf9, 0xfa,
];

interface HuffmanTable {
  /** 符号語（値 → ビット列）。 */
  readonly codes: Uint32Array;
  /** 符号語の長さ。0 なら未定義。 */
  readonly lengths: Uint8Array;
}

/** BITS / HUFFVAL から符号語を作る（T.81 Annex C の手順そのまま）。 */
function buildHuffmanTable(bits: readonly number[], values: readonly number[]): HuffmanTable {
  const codes = new Uint32Array(256);
  const lengths = new Uint8Array(256);
  let code = 0;
  let position = 0;
  for (let length = 1; length <= 16; length++) {
    for (let count = 0; count < bits[length]; count++) {
      const value = values[position++];
      codes[value] = code;
      lengths[value] = length;
      code++;
    }
    code <<= 1;
  }
  return { codes, lengths };
}

/**
 * 品質から量子化表を作る（libjpeg の `jpeg_set_quality` と同じ式）。
 *
 *     scale = quality < 50 ? 5000 / quality : 200 - quality * 2
 *     value = clamp((base * scale + 50) / 100, 1, 255)
 */
export function quantizationTable(base: readonly number[], quality: number): Uint8Array {
  const clampedQuality = Math.min(100, Math.max(1, Math.round(quality)));
  const scale =
    clampedQuality < 50 ? Math.floor(5000 / clampedQuality) : 200 - clampedQuality * 2;
  const table = new Uint8Array(64);
  for (let index = 0; index < 64; index++) {
    const value = Math.floor((base[index] * scale + 50) / 100);
    table[index] = value < 1 ? 1 : value > 255 ? 255 : value;
  }
  return table;
}

/** ビット単位で書き出す（0xFF の後に 0x00 を挟むのは JPEG の規約）。 */
class BitWriter {
  private readonly bytes: number[] = [];
  private accumulator = 0;
  private used = 0;

  writeByte(value: number): void {
    this.bytes.push(value & 0xff);
  }

  writeWord(value: number): void {
    this.writeByte(value >> 8);
    this.writeByte(value);
  }

  writeBits(value: number, length: number): void {
    for (let bit = length - 1; bit >= 0; bit--) {
      this.accumulator = (this.accumulator << 1) | ((value >> bit) & 1);
      this.used++;
      if (this.used === 8) {
        const byte = this.accumulator & 0xff;
        this.bytes.push(byte);
        // マーカーと見分けが付かなくなるので 0xFF の後には 0x00 を置く。
        if (byte === 0xff) this.bytes.push(0x00);
        this.accumulator = 0;
        this.used = 0;
      }
    }
  }

  /** 残りのビットを 1 で埋めて閉じる。 */
  flushBits(): void {
    while (this.used !== 0) this.writeBits(1, 1);
  }

  toUint8Array(): Uint8Array {
    return Uint8Array.from(this.bytes);
  }
}

/** 8x8 の分離型 DCT-II（レベルシフト済みの値を渡す）。 */
function forwardDct(block: Float64Array, out: Float64Array): void {
  const cosine = DCT_COSINE;
  const temporary = new Float64Array(64);
  for (let row = 0; row < 8; row++) {
    for (let u = 0; u < 8; u++) {
      let sum = 0;
      for (let x = 0; x < 8; x++) sum += block[row * 8 + x] * cosine[u * 8 + x];
      temporary[row * 8 + u] = sum;
    }
  }
  for (let column = 0; column < 8; column++) {
    for (let v = 0; v < 8; v++) {
      let sum = 0;
      for (let y = 0; y < 8; y++) sum += temporary[y * 8 + column] * cosine[v * 8 + y];
      out[v * 8 + column] = sum;
    }
  }
}

/** `cosine[u * 8 + x] = c(u) * cos((2x+1) u pi / 16)`、`c(0)=sqrt(1/8)` / `c(u)=1/2`。 */
const DCT_COSINE = (() => {
  const table = new Float64Array(64);
  for (let u = 0; u < 8; u++) {
    const normalization = u === 0 ? Math.sqrt(1 / 8) : 0.5;
    for (let x = 0; x < 8; x++) {
      table[u * 8 + x] = normalization * Math.cos(((2 * x + 1) * u * Math.PI) / 16);
    }
  }
  return table;
})();

/** 係数の大きさを表すのに要るビット数（DC の差分と AC の値に使う）。 */
function magnitudeCategory(value: number): number {
  let magnitude = Math.abs(value);
  let category = 0;
  while (magnitude !== 0) {
    magnitude >>= 1;
    category++;
  }
  return category;
}

interface Component {
  /** 成分ごとの標本（レベルシフト前の 0〜255）。 */
  readonly samples: Float64Array;
  readonly quantization: Uint8Array;
  readonly dcTable: HuffmanTable;
  readonly acTable: HuffmanTable;
  /** 量子化表の番号（0 = 輝度 / 1 = 色差）。 */
  readonly quantizationIndex: number;
}

/**
 * RGB を baseline JPEG（4:4:4）にする。
 *
 * @param quality 1〜100。デスクトップ側と同じ 90 を既定にする側は呼び出し元
 */
export function encodeJpeg444(
  data: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  quality: number,
): Uint8Array {
  if (width < 1 || height < 1) throw new Error(`大きさが ${width}x${height}`);
  if (data.length !== width * height * 3) {
    throw new Error(`RGB の要素数が ${data.length}（期待 ${width * height * 3}）`);
  }

  const luminanceQuantization = quantizationTable(BASE_LUMINANCE_TABLE, quality);
  const chrominanceQuantization = quantizationTable(BASE_CHROMINANCE_TABLE, quality);
  const dcLuminance = buildHuffmanTable(DC_LUMINANCE_BITS, DC_LUMINANCE_VALUES);
  const acLuminance = buildHuffmanTable(AC_LUMINANCE_BITS, AC_LUMINANCE_VALUES);
  const dcChrominance = buildHuffmanTable(DC_CHROMINANCE_BITS, DC_CHROMINANCE_VALUES);
  const acChrominance = buildHuffmanTable(AC_CHROMINANCE_BITS, AC_CHROMINANCE_VALUES);

  // JFIF の YCbCr（フルレンジ）。Pillow / libjpeg と同じ係数。
  const y = new Float64Array(width * height);
  const cb = new Float64Array(width * height);
  const cr = new Float64Array(width * height);
  for (let pixel = 0; pixel < width * height; pixel++) {
    const red = data[pixel * 3];
    const green = data[pixel * 3 + 1];
    const blue = data[pixel * 3 + 2];
    y[pixel] = 0.299 * red + 0.587 * green + 0.114 * blue;
    cb[pixel] = 128 - 0.168736 * red - 0.331264 * green + 0.5 * blue;
    cr[pixel] = 128 + 0.5 * red - 0.418688 * green - 0.081312 * blue;
  }

  const components: Component[] = [
    {
      samples: y,
      quantization: luminanceQuantization,
      dcTable: dcLuminance,
      acTable: acLuminance,
      quantizationIndex: 0,
    },
    {
      samples: cb,
      quantization: chrominanceQuantization,
      dcTable: dcChrominance,
      acTable: acChrominance,
      quantizationIndex: 1,
    },
    {
      samples: cr,
      quantization: chrominanceQuantization,
      dcTable: dcChrominance,
      acTable: acChrominance,
      quantizationIndex: 1,
    },
  ];

  const writer = new BitWriter();
  writer.writeWord(0xffd8); // SOI

  // APP0（JFIF）。密度は 1:1 の「単位なし」。
  writer.writeWord(0xffe0);
  writer.writeWord(16);
  for (const code of [0x4a, 0x46, 0x49, 0x46, 0x00]) writer.writeByte(code);
  writer.writeWord(0x0101); // version 1.1
  writer.writeByte(0); // units = none
  writer.writeWord(1);
  writer.writeWord(1);
  writer.writeByte(0);
  writer.writeByte(0);

  // DQT（zigzag 順で書く）。
  for (const [index, table] of [luminanceQuantization, chrominanceQuantization].entries()) {
    writer.writeWord(0xffdb);
    writer.writeWord(67);
    writer.writeByte(index);
    for (let position = 0; position < 64; position++) writer.writeByte(table[ZIGZAG[position]]);
  }

  // SOF0（baseline）。**標本比は 3 成分すべて 1x1 = 4:4:4。**
  writer.writeWord(0xffc0);
  writer.writeWord(8 + 3 * 3);
  writer.writeByte(8);
  writer.writeWord(height);
  writer.writeWord(width);
  writer.writeByte(3);
  components.forEach((component, index) => {
    writer.writeByte(index + 1);
    writer.writeByte(0x11);
    writer.writeByte(component.quantizationIndex);
  });

  // DHT。
  for (const [classAndId, bits, values] of [
    [0x00, DC_LUMINANCE_BITS, DC_LUMINANCE_VALUES],
    [0x10, AC_LUMINANCE_BITS, AC_LUMINANCE_VALUES],
    [0x01, DC_CHROMINANCE_BITS, DC_CHROMINANCE_VALUES],
    [0x11, AC_CHROMINANCE_BITS, AC_CHROMINANCE_VALUES],
  ] as const) {
    writer.writeWord(0xffc4);
    writer.writeWord(3 + 16 + values.length);
    writer.writeByte(classAndId);
    for (let length = 1; length <= 16; length++) writer.writeByte(bits[length]);
    for (const value of values) writer.writeByte(value);
  }

  // SOS。
  writer.writeWord(0xffda);
  writer.writeWord(6 + 2 * 3);
  writer.writeByte(3);
  components.forEach((component, index) => {
    writer.writeByte(index + 1);
    writer.writeByte(component.quantizationIndex === 0 ? 0x00 : 0x11);
  });
  writer.writeByte(0);
  writer.writeByte(63);
  writer.writeByte(0);

  // --- 走査 ---------------------------------------------------------------
  const previousDc = [0, 0, 0];
  const block = new Float64Array(64);
  const coefficients = new Float64Array(64);
  const quantized = new Int32Array(64);
  const blocksX = Math.ceil(width / 8);
  const blocksY = Math.ceil(height / 8);

  for (let blockY = 0; blockY < blocksY; blockY++) {
    for (let blockX = 0; blockX < blocksX; blockX++) {
      components.forEach((component, index) => {
        // 端の 8x8 に足りないところは最後の画素を伸ばす（libjpeg も端を複製する）。
        for (let row = 0; row < 8; row++) {
          const sourceRow = Math.min(height - 1, blockY * 8 + row);
          for (let column = 0; column < 8; column++) {
            const sourceColumn = Math.min(width - 1, blockX * 8 + column);
            block[row * 8 + column] =
              component.samples[sourceRow * width + sourceColumn] - 128;
          }
        }
        forwardDct(block, coefficients);
        for (let position = 0; position < 64; position++) {
          const natural = ZIGZAG[position];
          quantized[position] = Math.round(
            coefficients[natural] / component.quantization[natural],
          );
        }

        // DC は前のブロックとの差分。
        const difference = quantized[0] - previousDc[index];
        previousDc[index] = quantized[0];
        writeCoefficient(writer, component.dcTable, difference, 0);

        // AC は (連続ゼロ数, 大きさ) の組。
        let runLength = 0;
        for (let position = 1; position < 64; position++) {
          const value = quantized[position];
          if (value === 0) {
            runLength++;
            continue;
          }
          // 0 が 16 個以上続いたら ZRL（0xF0）で刻む。
          while (runLength >= 16) {
            writeSymbol(writer, component.acTable, 0xf0);
            runLength -= 16;
          }
          writeCoefficient(writer, component.acTable, value, runLength);
          runLength = 0;
        }
        // 末尾がゼロで終わるなら EOB（0x00）。
        if (runLength > 0) writeSymbol(writer, component.acTable, 0x00);
      });
    }
  }

  writer.flushBits();
  writer.writeWord(0xffd9); // EOI
  return writer.toUint8Array();
}

function writeSymbol(writer: BitWriter, table: HuffmanTable, symbol: number): void {
  const length = table.lengths[symbol];
  if (length === 0) throw new Error(`Huffman 表に記号 ${symbol} が無い`);
  writer.writeBits(table.codes[symbol], length);
}

/** (連続ゼロ数, 大きさ) の記号と、それに続く値のビットを書く。 */
function writeCoefficient(
  writer: BitWriter,
  table: HuffmanTable,
  value: number,
  runLength: number,
): void {
  const category = magnitudeCategory(value);
  writeSymbol(writer, table, (runLength << 4) | category);
  if (category === 0) return;
  // 負の値は 2 の補数ではなく「(2^category - 1) を足した値」で書く（T.81 の規約）。
  const encoded = value < 0 ? value + (1 << category) - 1 : value;
  writer.writeBits(encoded, category);
}
