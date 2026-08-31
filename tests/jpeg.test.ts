// JPEG エンコーダの検査。
//
// **見るのは「色差を間引いていないこと」と表の中身。** デスクトップ側が `subsampling=0` を明示して
// いる理由（4:2:0 はアトラスの chart 境界と髪の縁で色をにじませ、Unity 側の継ぎ目の原因を切り分け
// られなくする）が、この 1 点に掛かっている。
//
// 画素そのものは Pillow との突き合わせで確認した（Node には JPEG のデコーダが無いので、ここでは
// 構造だけを見る）。再現手順:
//
//     1. `encodeJpeg444` の出力を .jpg として書き出す
//     2. Pillow で開き、`img.layer` の標本比・`img.quantization`・画素を元の配列と比べる
//     3. 同じ配列を Pillow の `save(quality=90, subsampling=0)` でも書き、画素とバイト数を比べる
//
// 実測（197x131 の色差が高周波な絵）: 標本比は 3 成分すべて (1,1) / 量子化表は輝度・色差ともに
// Pillow の q=90 と**完全一致** / 往復誤差 最大 10・平均 1.29 / Pillow 出力との画素差 最大 9・
// 平均 0.34 / バイト数 20,011 対 20,006。

import { describe, expect, it } from 'vitest';
import { encodeJpeg444, quantizationTable } from '../src/infrastructure/jpeg';

function makeImage(width: number, height: number): Uint8Array {
  const data = new Uint8Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = (y * width + x) * 3;
      data[index] = (x * 7) % 256;
      data[index + 1] = (y * 11) % 256;
      // 1 画素ごとに振れる色差（4:2:0 なら潰れる）。
      data[index + 2] = x % 2 === 0 ? 255 : 0;
    }
  }
  return data;
}

/** マーカーの並びを取り出す（走査データは読み飛ばす）。 */
function markers(bytes: Uint8Array): number[] {
  const found: number[] = [];
  let position = 0;
  while (position + 1 < bytes.length) {
    if (bytes[position] !== 0xff) {
      position++;
      continue;
    }
    const marker = bytes[position + 1];
    if (marker === 0x00 || marker === 0xff) {
      position += 2;
      continue;
    }
    found.push(marker);
    if (marker === 0xd8 || marker === 0xd9) {
      position += 2;
      continue;
    }
    if (marker === 0xda) break; // SOS 以降は走査データ
    const length = (bytes[position + 2] << 8) | bytes[position + 3];
    position += 2 + length;
  }
  return found;
}

function findSegment(bytes: Uint8Array, marker: number): Uint8Array {
  let position = 2;
  while (position + 3 < bytes.length) {
    if (bytes[position] !== 0xff) {
      position++;
      continue;
    }
    const current = bytes[position + 1];
    const length = (bytes[position + 2] << 8) | bytes[position + 3];
    if (current === marker) return bytes.subarray(position + 4, position + 2 + length);
    if (current === 0xda) break;
    position += 2 + length;
  }
  throw new Error(`マーカー 0x${marker.toString(16)} が無い`);
}

describe('量子化表', () => {
  it('libjpeg の品質スケールと同じ式（Pillow の表と一致することは実測済み）', () => {
    // q=50 は元の表そのまま、q=100 は全部 1。
    const base = [16, 11, 10, 16, 24, 40, 51, 61];
    const table50 = quantizationTable([...base, ...new Array(56).fill(99)], 50);
    expect([...table50.subarray(0, 8)]).toEqual(base);
    const table100 = quantizationTable([...base, ...new Array(56).fill(99)], 100);
    expect([...new Set(table100)]).toEqual([1]);
    // q=90 は scale=20 なので (base*20+50)/100 の切り捨て。
    const table90 = quantizationTable([...base, ...new Array(56).fill(99)], 90);
    expect([...table90.subarray(0, 8)]).toEqual(
      base.map((value) => Math.floor((value * 20 + 50) / 100)),
    );
  });

  it('1 未満と 255 超えを飽和させる', () => {
    const table = quantizationTable(new Array(64).fill(255), 1);
    expect([...new Set(table)]).toEqual([255]);
    const fine = quantizationTable(new Array(64).fill(1), 99);
    expect([...new Set(fine)]).toEqual([1]);
  });
});

describe('baseline JPEG（4:4:4）', () => {
  const width = 41;
  const height = 27;
  const bytes = encodeJpeg444(makeImage(width, height), width, height, 90);

  it('マーカーが baseline の並びで揃っている', () => {
    expect(markers(bytes)).toEqual([
      0xd8, // SOI
      0xe0, // APP0 (JFIF)
      0xdb, // DQT 輝度
      0xdb, // DQT 色差
      0xc0, // SOF0 (baseline)
      0xc4, // DHT x4
      0xc4,
      0xc4,
      0xc4,
      0xda, // SOS
    ]);
    // EOI で終わる。
    expect([bytes[bytes.length - 2], bytes[bytes.length - 1]]).toEqual([0xff, 0xd9]);
  });

  it('**色差を間引かない**（3 成分すべて標本比 1x1）', () => {
    const sof = findSegment(bytes, 0xc0);
    expect(sof[0]).toBe(8); // 8bit
    expect((sof[1] << 8) | sof[2]).toBe(height);
    expect((sof[3] << 8) | sof[4]).toBe(width);
    expect(sof[5]).toBe(3);
    for (let component = 0; component < 3; component++) {
      // [id, 標本比, 量子化表番号]
      expect(sof[6 + component * 3 + 1], `成分 ${component}`).toBe(0x11);
    }
  });

  it('量子化表を zigzag 順で 2 枚書く', () => {
    const dqt = findSegment(bytes, 0xdb);
    expect(dqt.length).toBe(1 + 64);
    expect(dqt[0]).toBe(0); // 表番号 0 = 輝度
    // zigzag の先頭は自然順の 0 番。q=90 で 16 → 4。
    expect(dqt[1]).toBe(Math.floor((16 * 20 + 50) / 100));
  });

  it('走査データの 0xFF に 0x00 を挟む（マーカーと衝突させない）', () => {
    // SOS の直後から EOI の直前までに、0xFF の次が 0x00 でも 0xD9 でもないバイトが無いこと。
    let scanStart = 2;
    while (!(bytes[scanStart] === 0xff && bytes[scanStart + 1] === 0xda)) scanStart++;
    scanStart += 2 + ((bytes[scanStart + 2] << 8) | bytes[scanStart + 3]);
    for (let position = scanStart; position < bytes.length - 2; position++) {
      if (bytes[position] !== 0xff) continue;
      expect(bytes[position + 1], `位置 ${position}`).toBe(0x00);
    }
  });

  it('品質を上げるとバイト数が増える', () => {
    const image = makeImage(64, 64);
    const low = encodeJpeg444(image, 64, 64, 40).length;
    const high = encodeJpeg444(image, 64, 64, 95).length;
    expect(high).toBeGreaterThan(low);
  });

  it('8 の倍数でない大きさも書ける（端は最後の画素を伸ばす）', () => {
    for (const [w, h] of [
      [1, 1],
      [7, 3],
      [9, 17],
      [197, 131],
    ] as const) {
      const encoded = encodeJpeg444(makeImage(w, h), w, h, 90);
      const sof = findSegment(encoded, 0xc0);
      expect((sof[3] << 8) | sof[4], `${w}x${h}`).toBe(w);
      expect((sof[1] << 8) | sof[2], `${w}x${h}`).toBe(h);
    }
  });

  it('要素数と大きさが合わなければ落とす', () => {
    expect(() => encodeJpeg444(new Uint8Array(10), 4, 4, 90)).toThrow(/要素数/);
    expect(() => encodeJpeg444(new Uint8Array(0), 0, 4, 90)).toThrow(/大きさ/);
  });
});
