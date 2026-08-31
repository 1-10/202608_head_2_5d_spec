// PNG エンコーダの検査。
//
// **見るのはカラータイプと可逆性。** デスクトップ側は配列の次元で mode を決めていて `hair_alpha` は
// mode "L"（カラータイプ 0・1 チャンネル）で出る。契約（`domain/contract`）も「単一チャンネルの
// uint8 画像」と言っているので、RGB へ膨らませて書くと申告と違う形のファイルを渡すことになる。
//
// 可逆性は IDAT を展開して行フィルタを戻すところまでやる（PNG は可逆なので**厳密に一致するはず**）。
//
// Pillow との突き合わせも実施済み: mode は RGB / L で一致、画素は完全一致、バイト数は
// 17,562 対 17,743（RGB）/ 194 対 191（グレースケール）。バイト列の一致は求めない（deflate の実装が
// 違う）。

import { describe, expect, it } from 'vitest';
import { unzlibSync } from 'fflate';
import { encodeGrayPng, encodeRgbPng } from '../src/infrastructure/png';

const SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

interface Chunk {
  readonly type: string;
  readonly payload: Uint8Array;
}

function readChunks(bytes: Uint8Array): Chunk[] {
  expect([...bytes.subarray(0, 8)]).toEqual(SIGNATURE);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const chunks: Chunk[] = [];
  let position = 8;
  while (position < bytes.length) {
    const length = view.getUint32(position);
    const type = String.fromCharCode(...bytes.subarray(position + 4, position + 8));
    chunks.push({ type, payload: bytes.subarray(position + 8, position + 8 + length) });
    position += 12 + length;
  }
  return chunks;
}

function paeth(left: number, up: number, upLeft: number): number {
  const estimate = left + up - upLeft;
  const distanceLeft = Math.abs(estimate - left);
  const distanceUp = Math.abs(estimate - up);
  const distanceUpLeft = Math.abs(estimate - upLeft);
  if (distanceLeft <= distanceUp && distanceLeft <= distanceUpLeft) return left;
  return distanceUp <= distanceUpLeft ? up : upLeft;
}

/** IDAT を展開して行フィルタを戻す（画素の可逆性を測るのに要る最小の復号）。 */
function decodePixels(bytes: Uint8Array): {
  width: number;
  height: number;
  channels: number;
  data: Uint8Array;
} {
  const chunks = readChunks(bytes);
  const header = chunks.find((item) => item.type === 'IHDR');
  if (header === undefined) throw new Error('IHDR が無い');
  const view = new DataView(header.payload.buffer, header.payload.byteOffset);
  const width = view.getUint32(0);
  const height = view.getUint32(4);
  const colorType = header.payload[9];
  const channels = colorType === 0 ? 1 : 3;
  const idat = chunks.filter((item) => item.type === 'IDAT');
  const compressed = new Uint8Array(idat.reduce((sum, item) => sum + item.payload.length, 0));
  let offset = 0;
  for (const part of idat) {
    compressed.set(part.payload, offset);
    offset += part.payload.length;
  }
  const raw = unzlibSync(compressed);
  const stride = width * channels;
  const data = new Uint8Array(stride * height);
  let previous = new Uint8Array(stride);
  for (let row = 0; row < height; row++) {
    const filter = raw[row * (stride + 1)];
    const line = raw.subarray(row * (stride + 1) + 1, (row + 1) * (stride + 1));
    const out = data.subarray(row * stride, (row + 1) * stride);
    for (let index = 0; index < stride; index++) {
      const left = index >= channels ? out[index - channels] : 0;
      const up = previous[index];
      const upLeft = index >= channels ? previous[index - channels] : 0;
      const value = line[index];
      out[index] =
        filter === 0
          ? value
          : filter === 1
            ? (value + left) & 0xff
            : filter === 2
              ? (value + up) & 0xff
              : filter === 3
                ? (value + ((left + up) >> 1)) & 0xff
                : (value + paeth(left, up, upLeft)) & 0xff;
    }
    previous = out;
  }
  return { width, height, channels, data };
}

function makeRgb(width: number, height: number): Uint8Array {
  const data = new Uint8Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = (y * width + x) * 3;
      data[index] = (x * 7) % 256;
      data[index + 1] = (y * 11) % 256;
      data[index + 2] = (x * y) % 256;
    }
  }
  return data;
}

function makeGray(width: number, height: number): Uint8Array {
  const data = new Uint8Array(width * height);
  for (let index = 0; index < data.length; index++) data[index] = (index * 3) % 256;
  return data;
}

describe('PNG', () => {
  it('チャンクは IHDR → IDAT → IEND', () => {
    const bytes = encodeRgbPng(makeRgb(17, 13), 17, 13);
    expect(readChunks(bytes).map((chunk) => chunk.type)).toEqual(['IHDR', 'IDAT', 'IEND']);
  });

  it('**単一チャンネルはカラータイプ 0（グレースケール）で書く**', () => {
    const bytes = encodeGrayPng(makeGray(19, 7), 19, 7);
    const header = readChunks(bytes)[0].payload;
    expect(header[8]).toBe(8); // bit depth
    expect(header[9]).toBe(0); // color type = grayscale
    expect(header[12]).toBe(0); // interlace = none
    const decoded = decodePixels(bytes);
    expect(decoded.channels).toBe(1);
  });

  it('RGB はカラータイプ 2（アルファを足さない）', () => {
    const header = readChunks(encodeRgbPng(makeRgb(9, 9), 9, 9))[0].payload;
    expect(header[9]).toBe(2);
  });

  it('可逆（画素が厳密に一致する）', () => {
    for (const [width, height] of [
      [1, 1],
      [3, 7],
      [64, 64],
      [197, 131],
    ] as const) {
      const rgb = makeRgb(width, height);
      const decodedRgb = decodePixels(encodeRgbPng(rgb, width, height));
      expect(decodedRgb.width, `${width}x${height}`).toBe(width);
      expect(decodedRgb.height, `${width}x${height}`).toBe(height);
      expect(decodedRgb.data, `RGB ${width}x${height}`).toEqual(rgb);

      const gray = makeGray(width, height);
      const decodedGray = decodePixels(encodeGrayPng(gray, width, height));
      expect(decodedGray.data, `グレー ${width}x${height}`).toEqual(gray);
    }
  });

  it('行フィルタは絵に合わせて選ぶ', () => {
    const size = 32;
    /** IDAT の行フィルタ番号を並べて返す。 */
    const filtersOf = (data: Uint8Array): number[] => {
      const chunks = readChunks(encodeRgbPng(data, size, size));
      const raw = unzlibSync(chunks.filter((chunk) => chunk.type === 'IDAT')[0].payload);
      const stride = size * 3;
      return Array.from({ length: size }, (_, row) => raw[row * (stride + 1)]);
    };

    // どの行も同じ絵（横方向だけに変化）→ 上の画素との差が 0 になる Up(2) が最小。
    const identicalRows = new Uint8Array(size * size * 3);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        identicalRows.fill(x * 8, (y * size + x) * 3, (y * size + x) * 3 + 3);
      }
    }
    expect(new Set(filtersOf(identicalRows).slice(1))).toEqual(new Set([2]));

    // 行ごとに一定の絵（縦方向だけに変化）→ 左の画素との差が 0 になる Sub(1) か Paeth(4)。
    // Up は行間の差ぶん（一定値）を毎画素払うので選ばれない。
    const constantRows = new Uint8Array(size * size * 3);
    for (let y = 0; y < size; y++) {
      constantRows.fill(y * 8, y * size * 3, (y + 1) * size * 3);
    }
    const rowFilters = filtersOf(constantRows).slice(1);
    for (const filter of rowFilters) expect([1, 4]).toContain(filter);
  });

  it('要素数と大きさが合わなければ落とす', () => {
    expect(() => encodeRgbPng(new Uint8Array(10), 4, 4)).toThrow(/要素数/);
    expect(() => encodeGrayPng(new Uint8Array(4), 0, 4)).toThrow(/大きさ/);
  });
});
