// 画像UV空間でサンプルできる2Dスカラー場。
// セグメンテーションマスク・計測Depthなど「画像に張り付いたデータ」を
// メッシュ頂点から参照するための共通表現。

export interface ScalarField {
  width: number;
  height: number;
  data: Float32Array; // row-major, [0,0]=画像左上
  /** 場が画像全体のどの矩形に対応するか (画像UV, 0-1)。全体なら {0,0,1,1}。 */
  rect: { u0: number; v0: number; u1: number; v1: number };
}

export function fullImageRect(): ScalarField['rect'] {
  return { u0: 0, v0: 0, u1: 1, v1: 1 };
}

/**
 * 画像UV (u: 0-1 左→右, v: 0-1 下→上) でbilinearサンプルする。
 * rect外は最近傍縁の値へクランプせず0を返す (マスク・Depthとも「データ無し」扱い)。
 */
export function sampleField(field: ScalarField, u: number, v: number): number {
  const { u0, v0, u1, v1 } = field.rect;
  const tu = (u - u0) / Math.max(1e-9, u1 - u0);
  const tv = (v - v0) / Math.max(1e-9, v1 - v0);
  if (tu < 0 || tu > 1 || tv < 0 || tv > 1) return 0;

  // vは上が1、dataは上端がrow 0
  const fx = tu * (field.width - 1);
  const fy = (1 - tv) * (field.height - 1);
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const x1 = Math.min(field.width - 1, x0 + 1);
  const y1 = Math.min(field.height - 1, y0 + 1);
  const ax = fx - x0;
  const ay = fy - y0;

  const d = field.data;
  const w = field.width;
  const v00 = d[y0 * w + x0];
  const v10 = d[y0 * w + x1];
  const v01 = d[y1 * w + x0];
  const v11 = d[y1 * w + x1];
  return (v00 * (1 - ax) + v10 * ax) * (1 - ay) + (v01 * (1 - ax) + v11 * ax) * ay;
}

/** 場の値が threshold を超える領域の画像UV bounding box を返す。無ければ null。 */
export function fieldBoundsUv(
  field: ScalarField,
  threshold: number,
): { uMin: number; uMax: number; vMin: number; vMax: number } | null {
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;
  for (let y = 0; y < field.height; y++) {
    for (let x = 0; x < field.width; x++) {
      if (field.data[y * field.width + x] > threshold) {
        if (x < xMin) xMin = x;
        if (x > xMax) xMax = x;
        if (y < yMin) yMin = y;
        if (y > yMax) yMax = y;
      }
    }
  }
  if (xMin > xMax) return null;
  const { u0, v0, u1, v1 } = field.rect;
  const du = u1 - u0;
  const dv = v1 - v0;
  return {
    uMin: u0 + (xMin / (field.width - 1)) * du,
    uMax: u0 + (xMax / (field.width - 1)) * du,
    // row 0 = 画像上端 = v大 側
    vMin: v0 + (1 - yMax / (field.height - 1)) * dv,
    vMax: v0 + (1 - yMin / (field.height - 1)) * dv,
  };
}
