// 二段検出の検査。
//
// 正本はデスクトップ側 `infrastructure/face_landmarks.py` の `detect_two_pass`。**検出器を持たずに
// 検証できる**のがこの層の要点。
//
// **偽の検出器は渡された画像を実際に走査する。** 「元画像のどこを切り出したか」を教えない — 教えると
// 座標の往復（縮小・クロップ・戻し）を検証したことにならず、フィクスチャ側の計算と実装が一致して
// いるかを見るだけになる。顔は明るい正方形として写真へ描き、偽検出器はそれを見つけて点を作る。
//
// 実写での症状（2160x3840 の写真で口の位置がずれる）を同じ形の合成写真で再現している。

import { describe, expect, it } from 'vitest';
import {
  LADDER_STEP,
  REFINE_CROP_SPAN_FACTOR,
  SCOUT_LADDER_FLOOR,
  cropBoxAroundFace,
  detectTwoPass,
  scaleLadder,
} from '../src/domain/faceLadder';
import { FaceNotDetectedError } from '../src/domain/errors';
import { PhotoRgb } from '../src/domain/photo';
import { LANCZOS3, resamplePilToLongSide } from '../src/domain/resample';
import { faceSquareOfLandmarks } from '../src/domain/faceSubject';

const MESH_COUNT = 468;
const POINT_COUNT = 478;

/** 顔として描く正方形（元画像の画素座標）。 */
interface Square {
  readonly centerX: number;
  readonly centerY: number;
  readonly span: number;
}

/** 暗い背景に、顔ぶんの明るい正方形を描いた写真。 */
function photoWithFaces(width: number, height: number, faces: readonly Square[]): PhotoRgb {
  const data = new Uint8Array(width * height * 3).fill(20);
  for (const face of faces) {
    const half = face.span / 2;
    const left = Math.max(0, Math.round(face.centerX - half));
    const right = Math.min(width, Math.round(face.centerX + half));
    const top = Math.max(0, Math.round(face.centerY - half));
    const bottom = Math.min(height, Math.round(face.centerY + half));
    for (let row = top; row < bottom; row++) {
      data.fill(250, (row * width + left) * 3, (row * width + right) * 3);
    }
  }
  return { data, width, height };
}

/** しきい値を超える画素の連結成分ごとに外接矩形を返す（偽検出器の「顔を見つける」段）。 */
function brightComponents(photo: PhotoRgb, threshold = 135): Square[] {
  const { width, height } = photo;
  const visited = new Uint8Array(width * height);
  const squares: Square[] = [];
  for (let start = 0; start < width * height; start++) {
    if (visited[start] !== 0 || photo.data[start * 3] <= threshold) continue;
    let minimumX = width;
    let maximumX = -1;
    let minimumY = height;
    let maximumY = -1;
    const stack = [start];
    visited[start] = 1;
    while (stack.length > 0) {
      const pixel = stack.pop() as number;
      const x = pixel % width;
      const y = (pixel - x) / width;
      if (x < minimumX) minimumX = x;
      if (x > maximumX) maximumX = x;
      if (y < minimumY) minimumY = y;
      if (y > maximumY) maximumY = y;
      for (const [dx, dy] of [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
      ] as const) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
        const next = ny * width + nx;
        if (visited[next] !== 0 || photo.data[next * 3] <= threshold) continue;
        visited[next] = 1;
        stack.push(next);
      }
    }
    squares.push({
      centerX: (minimumX + maximumX + 1) / 2,
      centerY: (minimumY + maximumY + 1) / 2,
      span: Math.max(maximumX - minimumX + 1, maximumY - minimumY + 1),
    });
  }
  return squares;
}

/**
 * 顔 1 つぶんの 478 点。468 点を正方形の縁へ並べ（外接正方形が (centerX, centerY, span) になる）、
 * 虹彩 10 点は中心へ置く。
 */
function faceLandmarks(square: Square): Float64Array {
  const points = new Float64Array(POINT_COUNT * 2);
  const half = square.span / 2;
  for (let point = 0; point < MESH_COUNT; point++) {
    const angle = (point / MESH_COUNT) * Math.PI * 2;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const scale = 1 / Math.max(Math.abs(cos), Math.abs(sin));
    points[point * 2] = square.centerX + cos * scale * half;
    points[point * 2 + 1] = square.centerY + sin * scale * half;
  }
  for (let point = MESH_COUNT; point < POINT_COUNT; point++) {
    points[point * 2] = square.centerX;
    points[point * 2 + 1] = square.centerY;
  }
  return points;
}

/**
 * 渡された画像を走査して顔を返す偽検出器。
 *
 * @param minimumSpanPixels 顔の一辺がこの画素数未満だと「小さすぎて位置が取れない」で落ちる
 * @param calls 渡された画像の大きさを記録する
 */
function scanningDetector(minimumSpanPixels: number, calls?: string[]) {
  return (photo: PhotoRgb): Float64Array[] => {
    calls?.push(`${photo.width}x${photo.height}`);
    const found = brightComponents(photo)
      .filter((square) => square.span >= minimumSpanPixels)
      .map(faceLandmarks);
    if (found.length === 0) {
      throw new FaceNotDetectedError(`${photo.width}x${photo.height} では顔が小さすぎる`);
    }
    return found;
  };
}

describe('解像度の階段', () => {
  it('下端から 2 倍で登り、最後に元の長辺を置く', () => {
    expect(scaleLadder(3840)).toEqual([256, 512, 1024, 2048, 3840]);
    expect(scaleLadder(2048)).toEqual([256, 512, 1024, 2048]);
    // 元の長辺が下端より小さければ 1 段だけ。
    expect(scaleLadder(200)).toEqual([200]);
  });

  it('刻みと下端は正本と同じ', () => {
    expect(SCOUT_LADDER_FLOOR).toBe(256);
    expect(LADDER_STEP).toBe(2);
    expect(REFINE_CROP_SPAN_FACTOR).toBe(2);
  });

  it('作れない階段は落とす', () => {
    expect(() => scaleLadder(0)).toThrow(/長辺/);
    expect(() => scaleLadder(1000, 0)).toThrow(/階段が作れない/);
    expect(() => scaleLadder(1000, 256, 1)).toThrow(/階段が作れない/);
  });
});

describe('段2 のクロップ', () => {
  it('顔の一辺の 2 倍を取り、画像の外へは出さない', () => {
    expect(cropBoxAroundFace({ centerX: 500, centerY: 700, span: 200 }, 2160, 3840, 2)).toEqual([
      300, 500, 700, 900,
    ]);
    // 端では切り詰める（正方形は崩れる）。
    expect(cropBoxAroundFace({ centerX: 50, centerY: 40, span: 200 }, 2160, 3840, 2)).toEqual([
      0, 0, 250, 240,
    ]);
  });
});

describe('二段検出', () => {
  it('大きな写真でも段2 が元解像度から測り直す（口がずれる症状そのもの）', () => {
    // 実写と同じ形: 2160x3840 で顔の一辺は 250px（長辺の 6.5%）。
    const face: Square = { centerX: 1080, centerY: 1900, span: 250 };
    const photo = photoWithFaces(2160, 3840, [face]);
    const calls: string[] = [];
    const detectFaces = scanningDetector(24, calls);

    const result = detectTwoPass({ detectFaces, photo, faceMeshCount: MESH_COUNT });

    // 段1 は全段回る（256 / 512 / 1024 / 2048 / 3840 の 5 段）。
    expect(calls.slice(0, 5)).toEqual([
      '144x256',
      '288x512',
      '576x1024',
      '1152x2048',
      '2160x3840',
    ]);
    // 段2 はクロップを大きい側から試すので、1 段目（クロップの元解像度）で通る。
    // クロップの一辺は段1 が測った顔の 2 倍なので、真値の 2 倍（500）の近くになる。
    expect(calls.length).toBe(6);
    const [cropWidth, cropHeight] = calls[5].split('x').map(Number);
    expect(Math.abs(cropWidth - face.span * REFINE_CROP_SPAN_FACTOR)).toBeLessThan(20);
    expect(Math.abs(cropHeight - face.span * REFINE_CROP_SPAN_FACTOR)).toBeLessThan(20);

    const square = faceSquareOfLandmarks(result.landmarks, MESH_COUNT);
    // 元解像度で測り直しているので 1 画素の内側に収まる。
    expect(Math.abs(square.centerX - face.centerX)).toBeLessThan(1);
    expect(Math.abs(square.centerY - face.centerY)).toBeLessThan(1);
    expect(Math.abs(square.span - face.span)).toBeLessThan(1);
    expect(result.refineNotes.at(-1)).toContain('成功');
  });

  it('段1 の点は段2 より粗い（だから段1 を最終出力にしない）', () => {
    // 同じ写真を段1 の一番粗い段だけで測ると、中心も一辺も画素単位でずれる。
    const face: Square = { centerX: 1080, centerY: 1900, span: 250 };
    const photo = photoWithFaces(2160, 3840, [face]);
    // 長辺 1024 の段（実装と同じ縮小を通す）。256 の段はこの写真では顔が 17px で検出できない。
    const coarse = scanningDetector(24)(resamplePilToLongSide(photo, 1024, LANCZOS3));
    const scale = 1024 / 3840;
    const coarseSquare = faceSquareOfLandmarks(coarse[0], MESH_COUNT);
    // 縮小画像の 1 画素は元画像の約 4 画素。段1 だけだと一辺の推定がこの単位でしか出ない。
    expect(coarseSquare.span / scale).not.toBeCloseTo(face.span, 0);
    expect(Math.abs(coarseSquare.span / scale - face.span)).toBeGreaterThan(1);
  });

  it('段2 が全滅したら段1 の粗い結果で妥協せずに落とす', () => {
    const photo = photoWithFaces(1000, 1000, [{ centerX: 500, centerY: 500, span: 300 }]);
    const detectFaces = scanningDetector(24);
    // クロップ（600x600 以下）を渡された段では必ず落ちる検出器。
    const failing = (image: PhotoRgb): Float64Array[] => {
      if (image.width < 1000) throw new FaceNotDetectedError('クロップでは見えない');
      return detectFaces(image);
    };
    expect(() =>
      detectTwoPass({ detectFaces: failing, photo, faceMeshCount: MESH_COUNT }),
    ).toThrow(/切り出しての再検出が通りません/);
  });

  it('段1 が全滅したらどの段を試したかを書いて落とす', () => {
    expect(() =>
      detectTwoPass({
        detectFaces: () => {
          throw new FaceNotDetectedError('見えない');
        },
        photo: photoWithFaces(1000, 1000, []),
        faceMeshCount: MESH_COUNT,
      }),
    ).toThrow(/段1: 長辺 256.*長辺 512.*長辺 1000/s);
  });

  it('主役は画像中心に近い大きな顔（複数写っていても段2 で同じ顔を追う）', () => {
    const subject: Square = { centerX: 600, centerY: 620, span: 300 };
    const bystander: Square = { centerX: 130, centerY: 130, span: 140 };
    const photo = photoWithFaces(1200, 1200, [subject, bystander]);
    const result = detectTwoPass({
      detectFaces: scanningDetector(24),
      photo,
      faceMeshCount: MESH_COUNT,
    });
    const square = faceSquareOfLandmarks(result.landmarks, MESH_COUNT);
    // 端の小さい顔ではなく中心の大きい顔を返す。
    expect(Math.abs(square.centerX - subject.centerX)).toBeLessThan(1);
    expect(Math.abs(square.centerY - subject.centerY)).toBeLessThan(1);
    expect(Math.abs(square.span - subject.span)).toBeLessThan(1);
  });

  it('同じ画素数になる段は 1 度しか検出しない', () => {
    const calls: string[] = [];
    detectTwoPass({
      detectFaces: scanningDetector(8, calls),
      photo: photoWithFaces(300, 300, [{ centerX: 150, centerY: 150, span: 100 }]),
      faceMeshCount: MESH_COUNT,
    });
    // 階段は [256, 300]。どちらも別の画素数なので 2 段（重複していれば 1 段に落ちる）。
    const scout = calls.slice(0, 2);
    expect(new Set(scout).size).toBe(scout.length);
  });
});
