// PNG のエンコーダ（グレースケール / RGB）。
//
// **canvas の `toBlob('image/png')` を使わない。** あれは何を渡してもカラータイプ 6（RGBA）で書く。
// デスクトップ側（`infrastructure/imaging.encode_png`）は配列の次元で mode を決めていて、
// `hair_alpha` は **mode "L"（カラータイプ 0・1 チャンネル）**で出る。契約
// （`domain/contract`）も「単一チャンネルの uint8 画像」と言っているので、RGB へ膨らませて書くと
// **申告と違う形のファイルを渡す**（画素は同じでも、読む側が 3 チャンネルを受け取る）。
//
// 圧縮は fflate の zlib（deflate レベル 9 = Pillow の `optimize=True` と同じ側）。行フィルタは
// PNG 標準の 5 種から**行ごとに絶対値和が最小のものを選ぶ**（libpng / Pillow の適応フィルタと同じ
// 判断基準）。バイト列が Pillow と一致する保証はしない — deflate の実装が違うため。

import { zlibSync } from 'fflate';

const SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

/** カラータイプ 0（グレースケール）と 2（真の色）だけを書く。 */
const COLOR_TYPE_GRAYSCALE = 0;
const COLOR_TYPE_RGB = 2;

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let index = 0; index < 256; index++) {
    let value = index;
    for (let bit = 0; bit < 8; bit++) {
      value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
    }
    table[index] = value >>> 0;
  }
  return table;
})();

function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (const byte of bytes) crc = CRC_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function chunk(type: string, payload: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + payload.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, payload.length);
  for (let index = 0; index < 4; index++) out[4 + index] = type.charCodeAt(index);
  out.set(payload, 8);
  view.setUint32(8 + payload.length, crc32(out.subarray(4, 8 + payload.length)));
  return out;
}

/**
 * 行フィルタを 5 種試して絶対値和が最小のものを選ぶ。
 *
 * 判断基準は libpng の適応フィルタと同じ（絶対値和の最小）。**行ごとに選ぶ**のが PNG の規約で、
 * 画像全体で 1 つに固定すると圧縮が落ちる。
 */
function filterRow(
  row: Uint8Array,
  previous: Uint8Array,
  bytesPerPixel: number,
  out: Uint8Array,
): void {
  const width = row.length;
  const candidates = new Uint8Array(5 * width);
  const sums = new Float64Array(5);
  for (let index = 0; index < width; index++) {
    const raw = row[index];
    const left = index >= bytesPerPixel ? row[index - bytesPerPixel] : 0;
    const up = previous[index];
    const upLeft = index >= bytesPerPixel ? previous[index - bytesPerPixel] : 0;
    // 0=None 1=Sub 2=Up 3=Average 4=Paeth
    const values = [
      raw,
      (raw - left) & 0xff,
      (raw - up) & 0xff,
      (raw - ((left + up) >> 1)) & 0xff,
      (raw - paeth(left, up, upLeft)) & 0xff,
    ];
    for (let filter = 0; filter < 5; filter++) {
      candidates[filter * width + index] = values[filter];
      // 符号付きとして見た絶対値の和（128 以上は負として数える）。
      const value = values[filter];
      sums[filter] += value < 128 ? value : 256 - value;
    }
  }
  let best = 0;
  for (let filter = 1; filter < 5; filter++) if (sums[filter] < sums[best]) best = filter;
  out[0] = best;
  out.set(candidates.subarray(best * width, best * width + width), 1);
}

function paeth(left: number, up: number, upLeft: number): number {
  const estimate = left + up - upLeft;
  const distanceLeft = Math.abs(estimate - left);
  const distanceUp = Math.abs(estimate - up);
  const distanceUpLeft = Math.abs(estimate - upLeft);
  if (distanceLeft <= distanceUp && distanceLeft <= distanceUpLeft) return left;
  return distanceUp <= distanceUpLeft ? up : upLeft;
}

function encode(
  data: Uint8Array,
  width: number,
  height: number,
  channels: 1 | 3,
): Uint8Array {
  if (width < 1 || height < 1) throw new Error(`大きさが ${width}x${height}`);
  if (data.length !== width * height * channels) {
    throw new Error(`要素数が ${data.length}（期待 ${width * height * channels}）`);
  }
  const stride = width * channels;
  const raw = new Uint8Array((stride + 1) * height);
  let previous = new Uint8Array(stride);
  for (let row = 0; row < height; row++) {
    const source = data.subarray(row * stride, (row + 1) * stride);
    filterRow(source, previous, channels, raw.subarray(row * (stride + 1), (row + 1) * (stride + 1)));
    previous = source;
  }

  const header = new Uint8Array(13);
  const view = new DataView(header.buffer);
  view.setUint32(0, width);
  view.setUint32(4, height);
  header[8] = 8; // bit depth
  header[9] = channels === 1 ? COLOR_TYPE_GRAYSCALE : COLOR_TYPE_RGB;
  header[10] = 0; // compression = deflate
  header[11] = 0; // filter method = adaptive
  header[12] = 0; // interlace = none

  const parts = [
    Uint8Array.from(SIGNATURE),
    chunk('IHDR', header),
    chunk('IDAT', zlibSync(raw, { level: 9 })),
    chunk('IEND', new Uint8Array(0)),
  ];
  const total = parts.reduce((sum, part) => sum + part.length, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

/** RGB uint8 をカラータイプ 2 の PNG にする。 */
export function encodeRgbPng(data: Uint8Array, width: number, height: number): Uint8Array {
  return encode(data, width, height, 3);
}

/** 単一チャンネルの uint8 をカラータイプ 0（グレースケール）の PNG にする。 */
export function encodeGrayPng(data: Uint8Array, width: number, height: number): Uint8Array {
  return encode(data, width, height, 1);
}
