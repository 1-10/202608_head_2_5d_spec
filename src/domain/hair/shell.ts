// 段3〜6: 髪シェルのグリッド生成・厚み・縁の処理・三角形の採用。
//
// GNM は統計モデルなので髪を持たない（頭皮まで）。髪は写真から実測して、GNM の頭の手前に殻として
// 重ねる。段の役割分担:
//
//     段1 (depthFit)  DAViD の相対 Depth を GNM の z スケールへ合わせる
//     段2 (scalp)     GNM 表面の最前面 z。殻をどこに貼るかのアンカー
//     段3 (ここ)      髪マスクの範囲に格子を張り、格子点ごとに厚みを読む
//     段4 (ここ)      厚みの平滑化と縁の巻き込み
//     段5 (relief)    法線から毛束の起伏を足す
//     段6 (ここ)      旧方式と同じく、髪に触れる格子三角形だけを残す
//
// 最終 z（この式が正本）:
//
//     z = 頭皮z + lift + 厚み × edge − rolloff × (1 − edge) + 起伏 × edge
//
// 旧方式と同じく、頭皮が実在する場所も含めてこの式をそのまま使う。縁では `lift - rolloff` が負に
// なるため、殻が頭皮の内側へ入り、斜めから見える継ぎ目を隠す。
//
// **`lift < rolloff` を割らせない** — 割ると上の式の縁が正になり、殻が頭皮の外へ浮く。
//
// ジオメトリと alpha の輪郭分担
// -----------------------------
// 旧方式では、三角形の3頂点のどれかが髪マスクに触れれば三角形を丸ごと残す。精細な輪郭は alpha
// テクスチャと描画時の alpha test が担う。奥行きだけは平滑化した低周波マスクで巻き込み、生の房境界が
// メッシュをジグザグにしないようにする。
//
// 座標系
// ------
// 出力は GNM 空間のまま（右手系 / X=解剖学的左 / Y=上 / Z=前 / メートル）。左手系化は消費側の責務。
// 三角形の巻き順も GNM と同じ規約（外向き法線 = `(v1 − v0) × (v2 − v0)`）に揃える。

import { HairMask, ScalarField, insideRect, sampleField, uSpan, vSpan } from '../field';
import { Similarity2d } from '../gnm/fit';
import { smoothstep } from '../ramp';
import { DepthZFit, depthToZ } from './depthFit';
import {
  Grid2d,
  boxBlur3x3,
  gridCellSize,
  gridOverRect,
  gridPointsXy,
  passesForGrid,
} from './grid';
import { gnmXyToImageUv, imageUvToGnmXy } from './projection';
import { reliefFromNormal } from './relief';
import { ScalpSurface, buildScalpSurface } from './scalp';

/**
 * 既定の格子解像度（列 × 行）。DAViD の 512px crop の情報量を拾える密度。
 *
 * セルの寸法が GNM の三角形の辺長と同程度になる点数。頭皮 z バッファは三角形を走査するので、セルが
 * 辺長より大きいと格子点が三角形の間をすり抜ける。
 */
export const GRID_COLUMNS = 96;
export const GRID_ROWS = 120;

/** 髪マスクの bbox に足す余白（bbox の辺長に対する比）。縁の巻き込みと alpha の余地。 */
export const MARGIN_FRACTION = 0.05;

/**
 * 厚みのクランプ範囲（メートル）。
 *
 * 旧実装の `0.02..0.16`（faceWidth≈1 の正規化空間）を、当時の換算値 `1 unit ≈ 149 mm` でメートルへ
 * 戻した値。約 3..24 mm。
 */
export const THICKNESS_MIN_METERS = 0.003;
export const THICKNESS_MAX_METERS = 0.024;

/** 旧実装の `gnmHairRolloff=0.08` を当時の実寸へ戻した固定値。 */
export const LEGACY_ROLLOFF_METERS = 0.01193;

/** 旧実装の `gnmHairLift=0.02` を当時の実寸へ戻した固定値（約 3 mm）。 */
export const LEGACY_LIFT_METERS = 0.00298;

/**
 * 入口から調整できる `lift` と `rolloff` の範囲。
 *
 * 上限が旧値の数倍で止まるのは、この 2 つが殻の z を直接動かすため。桁を打ち間違えたまま書き出すと、
 * 殻が頭から離れた板になる。**`lift < rolloff` という関係のほうはここでは表せない**ので、両方を持つ
 * `HairShellParams` が検査する。
 */
export const MINIMUM_LIFT_METERS = 0.0;
export const MAXIMUM_LIFT_METERS = 0.02;
export const MINIMUM_ROLLOFF_METERS = 0.0;
export const MAXIMUM_ROLLOFF_METERS = 0.04;

/** 旧実装の 96x120 格子における厚み場の 3x3 blur 回数。 */
export const THICKNESS_BLUR_PASSES = 2;

/**
 * 旧実装の 96x120 格子における縁の 3x3 blur 回数。
 *
 * 厚み場より強くかけるのが要点。生のマスクを縁の重みに使うと房の切れ目ごとに z が上下してメッシュが
 * ジグザグになる。輪郭の精度は alpha テクスチャが担うので、ジオメトリ側の縁は低周波だけでよい。
 */
export const EDGE_BLUR_PASSES = 6;

/** 旧実装の `smoothstep(0.08, 0.50, 平滑マスク)` と同じ区間。 */
export const EDGE_SMOOTHSTEP_LOW = 0.08;
export const EDGE_SMOOTHSTEP_HIGH = 0.5;

/** 旧実装の範囲抽出値と三角形採用値。連続マスクに対して使う。 */
export const HAIR_BOUNDS_LEVEL = 0.08;
export const TRIANGLE_KEEP_LEVEL = 0.02;

/** 髪シェル生成のパラメータ。既定値はモジュール定数（そちらに根拠がある）。 */
export interface HairShellParams {
  readonly columns: number;
  readonly rows: number;
  readonly marginFraction: number;
  readonly rolloffMeters: number;
  readonly liftMeters: number;
  readonly thicknessMinMeters: number;
  readonly thicknessMaxMeters: number;
  readonly thicknessBlurPasses: number;
  readonly edgeBlurPasses: number;
  readonly edgeSmoothstepLow: number;
  readonly edgeSmoothstepHigh: number;
}

export const DEFAULT_HAIR_SHELL_PARAMS: HairShellParams = {
  columns: GRID_COLUMNS,
  rows: GRID_ROWS,
  marginFraction: MARGIN_FRACTION,
  rolloffMeters: LEGACY_ROLLOFF_METERS,
  liftMeters: LEGACY_LIFT_METERS,
  thicknessMinMeters: THICKNESS_MIN_METERS,
  thicknessMaxMeters: THICKNESS_MAX_METERS,
  thicknessBlurPasses: THICKNESS_BLUR_PASSES,
  edgeBlurPasses: EDGE_BLUR_PASSES,
  edgeSmoothstepLow: EDGE_SMOOTHSTEP_LOW,
  edgeSmoothstepHigh: EDGE_SMOOTHSTEP_HIGH,
};

/**
 * `lift < rolloff` を割らせない。
 *
 * 割ると `z = 頭皮z + lift + 厚み×edge − rolloff×(1−edge)` の縁（edge=0）が正になり、殻の外周が
 * 頭皮の外へ浮いて継ぎ目が見える。**両方の値を持つのはこの型だけ**なので、ここが正本。
 */
export function validateHairShellParams(params: HairShellParams): HairShellParams {
  if (!(params.liftMeters < params.rolloffMeters)) {
    throw new Error(
      `lift ${(params.liftMeters * 1000).toFixed(1)}mm が rolloff ` +
        `${(params.rolloffMeters * 1000).toFixed(1)}mm 以上（殻の外周が頭皮の外へ浮く）`,
    );
  }
  return params;
}

/** 髪シェルの出力と、各段の中間場（検査画像はこれをそのまま描く）。 */
export interface HairShellResult {
  /** (Nv, 3) GNM 空間・メートル。 */
  readonly positions: Float32Array;
  /** (Nv, 2) 写真の画像 UV 空間（テクスチャは元写真をそのまま貼る）。 */
  readonly uvs: Float32Array;
  /** (Nt, 3) GNM と同じ巻き順の規約。 */
  readonly triangles: Uint32Array;
  readonly grid: Grid2d;
  /** 段1 の結果。`residualRmsMeters` が検査の指標。 */
  readonly depthFit: DepthZFit;
  readonly scalp: ScalpSurface;
  readonly liftMeters: number;
  readonly rolloffMeters: number;
  /** (rows, columns) 段4 後の厚み（メートル）。 */
  readonly thickness: Float64Array;
  /** (rows, columns) 旧方式の三角形採用値 0.02 を超える格子点。 */
  readonly touchesHair: Uint8Array;
  /** (rows, columns) セル近傍の確信度。厚みの重みと縁の元（**判定ではなく重み**）。 */
  readonly presenceMean: Float64Array;
  /** (rows, columns) Depth の rect の内側にある格子点。 */
  readonly depthValid: Uint8Array;
  /** (rows, columns) 縁の重み 0..1。 */
  readonly edge: Float64Array;
  /** (rows, columns) 段5 の起伏（メートル）。 */
  readonly relief: Float64Array;
  /** (rows, columns) 全格子点の最終 z（採用されなかった点も含む）。 */
  readonly shellZ: Float64Array;
  /** (rows − 1, columns − 1) 輪郭内の三角形を持つセル。 */
  readonly adoptedCells: Uint8Array;
  readonly params: HairShellParams;
  readonly vertexCount: number;
  readonly triangleCount: number;
}

/** 髪の内側の格子点のうち、Depth の rect に入っていた割合。 */
export function depthCoverage(result: HairShellResult): number {
  let inside = 0;
  let valid = 0;
  for (let index = 0; index < result.touchesHair.length; index++) {
    if (result.touchesHair[index] === 0) continue;
    inside++;
    if (result.depthValid[index] !== 0) valid++;
  }
  return inside > 0 ? valid / inside : 0;
}

/**
 * 髪シェルを作る。髪が写っていなければ null。
 *
 * 「髪が無い」の判定は面積の門ではない
 * ------------------------------------
 * null になるのは**セグメンタが1画素も髪と判定しなかったとき**だけ（`confidence` が
 * `HAIR_BOUNDS_LEVEL` を超えず bbox が取れない）。面積の下限で切ると、刈り上げのような境界の写真が
 * 「殻が3枚まるごと消える」か「普通に出る」のどちらかに落ちる崖ができ、その境界の位置に根拠が無い。
 *
 * 髪が薄い写真では**厚みが自然に 0 へ落ちる**ので、「髪が無い」は門ではなく厚みが表現している。
 *
 * 残る退化は「髪と判定された範囲が頭部に1点も掛からない」場合で、殻はアンカーを失うので幾何が
 * 作れない。これは null（髪が無い）ではなく `EmptyScalpError` として上がる — 頭の上に無い髪は、
 * 相似変換かマスクが壊れている印であって「髪が無い写真」ではない。**黙って殻を落とさない**のが
 * この段の方針。
 */
export function buildHairShell(input: {
  vertices: Float64Array;
  triangles: Uint32Array;
  similarity: Similarity2d;
  hairMask: HairMask;
  depth: ScalarField;
  normal: Float32Array;
  imageSize: readonly [number, number];
  params?: HairShellParams;
  depthFit: DepthZFit;
}): HairShellResult | null {
  const settings = validateHairShellParams(input.params ?? DEFAULT_HAIR_SHELL_PARAMS);
  const { depth } = input;
  if (input.normal.length !== 3 * depth.height * depth.width) {
    throw new Error(
      `normal の形が depth と揃っていない: normal=${input.normal.length}` +
        ` depth=${depth.width}x${depth.height}`,
    );
  }

  const confidence = input.hairMask.confidence;
  const bounds = hairBounds(confidence, HAIR_BOUNDS_LEVEL);
  if (bounds === null) return null;

  const grid = gridOverMask(bounds, input.similarity, input.imageSize, settings);
  const scalp = buildScalpSurface(input.vertices, input.triangles, grid);

  const pointsXy = gridPointsXy(grid);
  const uv = gnmXyToImageUv(pointsXy, input.similarity, input.imageSize);
  const pointCount = grid.rows * grid.columns;
  const u = new Float64Array(pointCount);
  const v = new Float64Array(pointCount);
  for (let index = 0; index < pointCount; index++) {
    u[index] = uv[index * 2];
    v[index] = uv[index * 2 + 1];
  }

  const cellPixels = gridCellSize(grid) * input.similarity.scale;
  const presenceMean = new Float64Array(pointCount);
  const touchesHair = new Uint8Array(pointCount);
  for (let index = 0; index < pointCount; index++) {
    presenceMean[index] = Math.min(1, Math.max(0, sampleField(confidence, u[index], v[index])));
    touchesHair[index] = presenceMean[index] > TRIANGLE_KEEP_LEVEL ? 1 : 0;
  }
  const edgeField = boxBlur3x3(
    presenceMean,
    grid.rows,
    grid.columns,
    passesForGrid(settings.edgeBlurPasses, grid, GRID_COLUMNS, GRID_ROWS),
  );

  const depthValue = maskWeightedDepth(
    depth,
    confidence,
    u,
    v,
    (cellPixels / input.imageSize[0]) * 0.75,
    (cellPixels / input.imageSize[1]) * 0.75,
  );
  const rawThickness = new Float64Array(pointCount);
  for (let index = 0; index < pointCount; index++) {
    rawThickness[index] = depthToZ(input.depthFit, depthValue[index]) - scalp.z[index];
  }

  // Depth は写真全体ではなく頭部の切り出しに対して走ることがあり、rect の外は 0 が返る（「そこには
  // 何も無い」を 0 で表す規約）。0 をそのまま Depth として読むと厚みが `offset − 頭皮z` という
  // 無関係な値になる。
  const depthValid = new Uint8Array(pointCount);
  for (let index = 0; index < pointCount; index++) {
    depthValid[index] = insideRect(depth.rect, u[index], v[index]) ? 1 : 0;
  }

  const blurredThickness = boxBlur3x3(
    rawThickness,
    grid.rows,
    grid.columns,
    passesForGrid(settings.thicknessBlurPasses, grid, GRID_COLUMNS, GRID_ROWS),
  );
  const thickness = new Float64Array(pointCount);
  const edge = new Float64Array(pointCount);
  const anchorZ = new Float64Array(pointCount);
  const lift = settings.liftMeters;
  const rolloff = settings.rolloffMeters;
  for (let index = 0; index < pointCount; index++) {
    thickness[index] = Math.min(
      settings.thicknessMaxMeters,
      Math.max(settings.thicknessMinMeters, blurredThickness[index]),
    );
    edge[index] = smoothstep(
      settings.edgeSmoothstepLow,
      settings.edgeSmoothstepHigh,
      edgeField[index],
    );
    anchorZ[index] =
      scalp.z[index] + lift + thickness[index] * edge[index] - rolloff * (1 - edge[index]);
  }

  const relief = reliefFromNormal({
    normal: input.normal,
    normalWidth: depth.width,
    normalHeight: depth.height,
    rect: depth.rect,
    u,
    v,
    grid,
    weight: presenceMean,
    anchorZ,
  });

  const shellZ = new Float64Array(pointCount);
  for (let index = 0; index < pointCount; index++) shellZ[index] = anchorZ[index] + relief[index];

  const { kept, adoptedCells } = trianglesOverMask(presenceMean, grid, TRIANGLE_KEEP_LEVEL);
  if (kept.length === 0) return null;

  // 使った格子点だけを詰め直す（並びは昇順。`np.unique` と同じ）。
  const used = [...new Set(kept)].sort((first, second) => first - second);
  const remap = new Map<number, number>();
  used.forEach((point, index) => remap.set(point, index));
  const positions = new Float32Array(used.length * 3);
  const uvs = new Float32Array(used.length * 2);
  used.forEach((point, index) => {
    positions[index * 3] = pointsXy[point * 2];
    positions[index * 3 + 1] = pointsXy[point * 2 + 1];
    positions[index * 3 + 2] = shellZ[point];
    uvs[index * 2] = u[point];
    uvs[index * 2 + 1] = v[point];
  });
  const triangles = new Uint32Array(kept.length);
  kept.forEach((point, index) => {
    triangles[index] = remap.get(point) as number;
  });

  return {
    positions,
    uvs,
    triangles,
    grid,
    depthFit: input.depthFit,
    scalp,
    liftMeters: lift,
    rolloffMeters: rolloff,
    thickness,
    touchesHair,
    presenceMean,
    depthValid,
    edge,
    relief,
    shellZ,
    adoptedCells,
    params: settings,
    vertexCount: used.length,
    triangleCount: kept.length / 3,
  };
}

/**
 * 確信度が `level` を超える画素の bbox。無ければ null。
 *
 * bbox は画素の外縁で取る（画素中心ではなく、その画素が覆う範囲の端）。
 */
function hairBounds(
  confidence: ScalarField,
  level: number,
): [number, number, number, number] | null {
  let firstRow = -1;
  let lastRow = -1;
  let firstColumn = -1;
  let lastColumn = -1;
  for (let row = 0; row < confidence.height; row++) {
    for (let column = 0; column < confidence.width; column++) {
      if (confidence.values[row * confidence.width + column] <= level) continue;
      if (firstRow < 0) firstRow = row;
      lastRow = row;
      if (firstColumn < 0 || column < firstColumn) firstColumn = column;
      if (column > lastColumn) lastColumn = column;
    }
  }
  if (firstRow < 0) return null;
  const rect = confidence.rect;
  const pixelU = uSpan(rect) / confidence.width;
  const pixelV = vSpan(rect) / confidence.height;
  return [
    rect.uMin + firstColumn * pixelU,
    rect.vMin + firstRow * pixelV,
    rect.uMin + (lastColumn + 1) * pixelU,
    rect.vMin + (lastRow + 1) * pixelV,
  ];
}

/** 旧方式と同じく、3頂点のどれかがマスクに触れる三角形を残す。 */
function trianglesOverMask(
  mask: Float64Array,
  grid: Grid2d,
  level: number,
): { kept: number[]; adoptedCells: Uint8Array } {
  const cellRows = grid.rows - 1;
  const cellColumns = grid.columns - 1;
  const adoptedCells = new Uint8Array(cellRows * cellColumns);
  const kept: number[] = [];
  // 並びは Python の `np.concatenate([abd, adc])` と同じ（先に全セルの abd、次に adc）。
  const candidates: [number, number[]][] = [];
  for (let row = 0; row < cellRows; row++) {
    for (let column = 0; column < cellColumns; column++) {
      const a = row * grid.columns + column;
      const b = a + 1;
      const c = a + grid.columns;
      const d = c + 1;
      candidates.push([row * cellColumns + column, [a, b, d]]);
    }
  }
  for (let row = 0; row < cellRows; row++) {
    for (let column = 0; column < cellColumns; column++) {
      const a = row * grid.columns + column;
      const c = a + grid.columns;
      const d = c + 1;
      candidates.push([row * cellColumns + column, [a, d, c]]);
    }
  }
  for (const [cell, corners] of candidates) {
    const maximum = Math.max(mask[corners[0]], mask[corners[1]], mask[corners[2]]);
    if (!(maximum > level)) continue;
    kept.push(corners[0], corners[1], corners[2]);
    adoptedCells[cell] = 1;
  }
  return { kept, adoptedCells };
}

/**
 * マスクの bbox を GNM 空間 XY へ移し、余白を足した矩形に格子を張る。
 *
 * 相似変換は面内回転を含みうるので、bbox の 4 隅すべてを移してから軸並行の bbox を取り直す
 * （2 隅だけでは回転した矩形を覆えない）。
 */
function gridOverMask(
  bounds: readonly [number, number, number, number],
  similarity: Similarity2d,
  imageSize: readonly [number, number],
  settings: HairShellParams,
): Grid2d {
  const [uMin, vMin, uMax, vMax] = bounds;
  const cornersUv = Float64Array.from([uMin, vMin, uMax, vMin, uMin, vMax, uMax, vMax]);
  const cornersXy = imageUvToGnmXy(cornersUv, similarity, imageSize);
  let xMin = Infinity;
  let yMin = Infinity;
  let xMax = -Infinity;
  let yMax = -Infinity;
  for (let corner = 0; corner < 4; corner++) {
    xMin = Math.min(xMin, cornersXy[corner * 2]);
    xMax = Math.max(xMax, cornersXy[corner * 2]);
    yMin = Math.min(yMin, cornersXy[corner * 2 + 1]);
    yMax = Math.max(yMax, cornersXy[corner * 2 + 1]);
  }
  const marginX = (xMax - xMin) * settings.marginFraction;
  const marginY = (yMax - yMin) * settings.marginFraction;
  return gridOverRect(
    xMin - marginX,
    yMin - marginY,
    xMax + marginX,
    yMax + marginY,
    settings.columns,
    settings.rows,
  );
}

/**
 * Depth を「髪マスクで重み付けした 3x3 近傍平均」で読む。
 *
 * 前髪のようなまばらな髪帯では、毛の隙間から見える肌（奥）の Depth が混ざる。DAViD は解像度が高い
 * のでこの段差を実際に解像してしまい、格子がジグザグになる。近傍を「髪らしさ」で重み付けすると、
 * 隙間画素の肌 Depth が落ちる。
 *
 * 近傍の刻み幅はセルの半分（ただし最低 1 画素）にする。1 画素固定だとセルよりずっと狭い範囲しか
 * 見ず、セル内の変動を平均できない。
 */
function maskWeightedDepth(
  depth: ScalarField,
  confidence: ScalarField,
  u: Float64Array,
  v: Float64Array,
  stepU: number,
  stepV: number,
): Float64Array {
  const out = new Float64Array(u.length);
  const offsets = [-1, 0, 1];
  for (let index = 0; index < u.length; index++) {
    let total = 0;
    let weighted = 0;
    let center = 0;
    for (const dv of offsets) {
      for (const du of offsets) {
        const sampleU = u[index] + du * stepU;
        const sampleV = v[index] + dv * stepV;
        const depthTap = sampleField(depth, sampleU, sampleV);
        const weight = smoothstep(
          0.2,
          0.7,
          Math.max(sampleField(confidence, sampleU, sampleV), 0),
        );
        total += weight;
        weighted += weight * depthTap;
        if (du === 0 && dv === 0) center = depthTap;
      }
    }
    const hairDepth = total > 1e-3 ? weighted / Math.max(total, 1e-3) : center;
    out[index] = sampleField(confidence, u[index], v[index]) < 0.05 ? center : hairDepth;
  }
  return out;
}
