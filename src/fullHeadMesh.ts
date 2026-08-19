// FULL HEAD: Head Grid Mesh。Face Depth Field + Head Depth + Edge Rolloff +
// Hair Volume + Head Silhouette Maskを1枚の連続したメッシュへ合成する。
// Face MeshとHead Gridのtopologyは分離しており、Head Gridは常に一定間隔の格子。
//
// Head Depth / Silhouette / Hairの供給源は2系統あり、paramsで切替できる:
// - MEASURED: MediaPipe SelfieMulticlass (実シルエット・実髪マスク) + ARPortraitDepth (実測Depth)
// - ELLIPSE / HEURISTIC: 楕円近似 + Pseudo Head Depth (従来方式。比較用)

import * as THREE from 'three';
import { FACE_KEY_INDICES, type FaceTriangulation, type NormalizedFaceLandmark } from './faceTopology';
import { type FaceDepthField, sampleFaceDepthField } from './faceDepth';
import { fieldBoundsUv, sampleField, type ScalarField } from './fields';
import {
  estimateHeadMaskEllipse,
  rasterizeHeadMaskCanvas,
  sampleHeadMask,
  type HeadMaskEllipse,
} from './headMask';
import {
  computeFaceHeadBlendWeight,
  computeForeheadDepth,
  computeForeheadWeight,
  computeHairVolume,
  computeHeadDepthFinal,
  smoothstep,
} from './headDepth';
import { applyFlatNormals } from './meshUtils';
import { rasterizeMaskCanvas, type SegmentationResult } from './personSegmentation';
import type { FullHeadMode, Params } from './params';

/**
 * 実測/ニューラルソース一式。取得に失敗した(または未取得の)ものはnull。
 * NEURAL系がnullのままNEURALを選ぶとMEASURED系へ、それもnullなら
 * 楕円/ヒューリスティックへフォールバックする。
 */
export interface MeasuredHeadData {
  segmentation: SegmentationResult | null; // MediaPipe SelfieMulticlass
  depth: ScalarField | null; // ARPortraitDepth 相対Depth (0-1)
  depthFit: { scale: number; offset: number } | null; // 相対Depth→モデル空間Z
  neuralSegmentation: SegmentationResult | null; // BiRefNet×MediaPipe合成 (遅延取得)
  neuralDepth: ScalarField | null; // Depth Anything V2 (遅延取得)
  neuralDepthFit: { scale: number; offset: number } | null;
}

/** maskSourceに応じたセグメンテーションを選ぶ (NEURAL未取得時はMEASUREDへフォールバック)。 */
export function selectSegmentation(ctx: FullHeadBuildContext, params: Params): SegmentationResult | null {
  const m = ctx.measured;
  if (!m) return null;
  if (params.maskSource === 'NEURAL') return m.neuralSegmentation ?? m.segmentation;
  if (params.maskSource === 'MEASURED') return m.segmentation;
  return null;
}

function selectDepth(
  ctx: FullHeadBuildContext,
  params: Params,
): { depth: ScalarField; fit: { scale: number; offset: number } } | null {
  const m = ctx.measured;
  if (!m) return null;
  if (params.depthSource === 'NEURAL') {
    if (m.neuralDepth && m.neuralDepthFit) return { depth: m.neuralDepth, fit: m.neuralDepthFit };
    if (m.depth && m.depthFit) return { depth: m.depth, fit: m.depthFit };
    return null;
  }
  if (params.depthSource === 'MEASURED') {
    if (m.depth && m.depthFit) return { depth: m.depth, fit: m.depthFit };
    return null;
  }
  return null;
}

export interface FullHeadBuildContext {
  landmarks: NormalizedFaceLandmark[];
  triangulation: FaceTriangulation;
  faceZFinal: Float32Array; // FACE ONLYと共通のFace Depth (canonical+mediapipe混合済み)
  depthField: FaceDepthField;
  headCenterPx: { x: number; y: number };
  faceWidthPx: number;
  imageWidth: number;
  imageHeight: number;
  measured: MeasuredHeadData | null;
}

export interface FullHeadDebugChannels {
  faceDepth: Float32Array;
  headDepth: Float32Array;
  edgeRolloff: Float32Array;
  hairVolume: Float32Array;
  finalDepth: Float32Array;
}

export interface FullHeadBuild {
  group: THREE.Group;
  mesh: THREE.Mesh;
  geometry: THREE.BufferGeometry;
  ellipse: HeadMaskEllipse;
  cols: number;
  rows: number;
  bounds: { xMin: number; xMax: number; yMin: number; yMax: number };
  basePositions: Float32Array;
  maskValues: Float32Array; // 頭部マスク値 (blink/talk maskとの積算などに再利用可)
  hairMaskValues: Float32Array; // 髪マスク値 (デバッグ表示用)
  debug: FullHeadDebugChannels;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function landmarkAvgY(landmarks: NormalizedFaceLandmark[], indices: number[]): number {
  let sum = 0;
  for (const i of indices) sum += landmarks[i].y;
  return sum / indices.length;
}

function faceZAvg(zFinal: Float32Array, indices: number[]): number {
  let sum = 0;
  for (const i of indices) sum += zFinal[i];
  return sum / indices.length;
}

interface VertexChannels {
  maskVal: number;
  faceSideZ: number;
  zOutside: number; // 顔外領域のHead Depth (計測 or 楕円)
  zRolloff: number;
  hairMask: number;
  hairVolume: number;
  blended: number;
}

/**
 * Head Grid頂点1点のDepth各成分を評価するclosureを作る。
 * buildとrecomputeで同一ロジックを共有するための一本化ポイント。
 */
function createVertexEvaluator(ctx: FullHeadBuildContext, ellipse: HeadMaskEllipse, params: Params) {
  const hull = ctx.triangulation.hull.map((i) => ({ x: ctx.landmarks[i].x, y: ctx.landmarks[i].y }));
  const k = FACE_KEY_INDICES;
  const browIndices = [...k.eyebrowA, ...k.eyebrowB];
  const browY = landmarkAvgY(ctx.landmarks, browIndices);
  const zBrowConst = faceZAvg(ctx.faceZFinal, browIndices);
  const eyeLineY = landmarkAvgY(ctx.landmarks, [k.eyeA.outer, k.eyeA.inner, k.eyeB.outer, k.eyeB.inner]);
  const headTopY = ellipse.cy + ellipse.ry;
  const blendWidth = params.blendWidthRatio;

  const seg = selectSegmentation(ctx, params);
  const selected = selectDepth(ctx, params);
  const depth = selected?.depth ?? null;
  const depthFit = selected?.fit ?? null;
  const useMeasuredDepth = depth !== null && depthFit !== null;

  return (x: number, y: number, u: number, v: number): VertexChannels => {
    const maskVal = seg ? sampleField(seg.head, u, v) : sampleHeadMask(x, y, ellipse);
    const headResult = computeHeadDepthFinal(x, y, ellipse, params.headDepthScale, params.edgeStart, params.edgeDepth);
    const wFace = computeFaceHeadBlendWeight(x, y, hull, blendWidth);

    let zOutside: number;
    let zRolloff: number;
    if (useMeasuredDepth) {
      const d = sampleField(depth, u, v);
      const zMeasured = (d * depthFit.scale + depthFit.offset) * params.measuredDepthGain;
      // 楕円Head Depthへの正則化で計測ノイズと外れ値を抑える
      const reg = params.measuredRegularize;
      const zBase = zMeasured * (1 - reg) + headResult.zHead * reg;
      // Rolloffは実シルエット(マスクのfeather)に沿って外周を後方へ巻き込む
      zRolloff = -params.edgeDepth * (1 - smoothstep(0.1, 0.7, maskVal));
      zOutside = zBase + zRolloff;
    } else {
      zRolloff = headResult.zRolloff;
      zOutside = headResult.zHeadFinal;
    }

    const sample = sampleFaceDepthField(ctx.depthField, u, v);
    let faceSideZ = sample.coverage > 0.5 ? sample.depth : zOutside;
    // 額上部(hull外)の補間: landmarkは生え際までしか無いため、その上をzBrowからHead Depthへ繋ぐ。
    // hull内(眉〜生え際)はFace Depth Fieldの実値を信頼する (定数で上書きすると眉線に段差が出る)。
    // 計測Depthがあるときは実測に任せる。
    if (!useMeasuredDepth && sample.coverage <= 0.5 && y > browY) {
      const t = computeForeheadWeight(y, browY, headTopY);
      faceSideZ = computeForeheadDepth(zBrowConst, zOutside, t);
    }

    const blended = faceSideZ * wFace + zOutside * (1 - wFace);

    const hairMask = seg ? sampleField(seg.hair, u, v) : maskVal * (1 - wFace);
    const verticalT = (y - eyeLineY) / Math.max(1e-6, headTopY - eyeLineY);
    const hairVolume = computeHairVolume(hairMask, verticalT, params.hairVolumeMax);

    return { maskVal, faceSideZ, zOutside, zRolloff, hairMask, hairVolume, blended };
  };
}

/** Grid境界を決める。実測マスクがあればそのbboxから、無ければ楕円から。 */
function computeGridBounds(
  ctx: FullHeadBuildContext,
  ellipse: HeadMaskEllipse,
  params: Params,
): { xMin: number; xMax: number; yMin: number; yMax: number } {
  const seg = selectSegmentation(ctx, params);
  if (seg) {
    const uvBounds = fieldBoundsUv(seg.head, 0.1);
    if (uvBounds) {
      // 画像UV → モデル空間
      const toX = (u: number) => (u * ctx.imageWidth - ctx.headCenterPx.x) / ctx.faceWidthPx;
      const toY = (v: number) => (ctx.headCenterPx.y - (1 - v) * ctx.imageHeight) / ctx.faceWidthPx;
      const marginX = (toX(uvBounds.uMax) - toX(uvBounds.uMin)) * 0.06;
      const marginY = (toY(uvBounds.vMax) - toY(uvBounds.vMin)) * 0.06;
      return {
        xMin: toX(uvBounds.uMin) - marginX,
        xMax: toX(uvBounds.uMax) + marginX,
        yMin: toY(uvBounds.vMin) - marginY,
        yMax: toY(uvBounds.vMax) + marginY,
      };
    }
  }
  const margin = 1.18;
  return {
    xMin: ellipse.cx - ellipse.rx * margin,
    xMax: ellipse.cx + ellipse.rx * margin,
    yMin: -0.35, // 顎下あたりまで(首は含めない)
    yMax: ellipse.cy + ellipse.ry * margin,
  };
}

export function buildHeadGridGeometry(ctx: FullHeadBuildContext, texture: THREE.Texture, params: Params): FullHeadBuild {
  const ellipse = estimateHeadMaskEllipse(ctx.landmarks);
  const cols = params.headGridCols;
  const rows = params.headGridRows;
  const bounds = computeGridBounds(ctx, ellipse, params);
  const evaluate = createVertexEvaluator(ctx, ellipse, params);

  const count = cols * rows;
  const positions = new Float32Array(count * 3);
  const uvs = new Float32Array(count * 2);
  const maskValues = new Float32Array(count);
  const hairMaskValues = new Float32Array(count);
  const faceDepthDbg = new Float32Array(count);
  const headDepthDbg = new Float32Array(count);
  const edgeRolloffDbg = new Float32Array(count);
  const hairVolumeDbg = new Float32Array(count);
  const finalDepthDbg = new Float32Array(count);

  for (let row = 0; row < rows; row++) {
    const y = lerp(bounds.yMax, bounds.yMin, row / (rows - 1)); // row0=頭頂
    for (let col = 0; col < cols; col++) {
      const x = lerp(bounds.xMin, bounds.xMax, col / (cols - 1));
      const idx = row * cols + col;

      const px = x * ctx.faceWidthPx + ctx.headCenterPx.x;
      const py = ctx.headCenterPx.y - y * ctx.faceWidthPx;
      const u = px / ctx.imageWidth;
      const v = 1 - py / ctx.imageHeight;

      const ch = evaluate(x, y, u, v);

      positions[idx * 3 + 0] = x;
      positions[idx * 3 + 1] = y;
      uvs[idx * 2 + 0] = u;
      uvs[idx * 2 + 1] = v;
      maskValues[idx] = ch.maskVal;
      hairMaskValues[idx] = ch.hairMask;

      faceDepthDbg[idx] = ch.faceSideZ;
      headDepthDbg[idx] = ch.zOutside;
      edgeRolloffDbg[idx] = ch.zRolloff;
      hairVolumeDbg[idx] = ch.hairVolume;

      const zByMode = computeZForMode(params.fullHeadMode, ch.zOutside, ch.blended, ch.hairVolume);
      positions[idx * 3 + 2] = zByMode;
      finalDepthDbg[idx] = zByMode;
    }
  }

  const indices = buildGridIndices(cols, rows);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  applyFlatNormals(geometry);

  const seg = selectSegmentation(ctx, params);
  const maskCanvas = seg
    ? rasterizeMaskCanvas(seg.head, seg.head.width >= 512 ? 1024 : 512)
    : rasterizeHeadMaskCanvas(ellipse, ctx.headCenterPx, ctx.faceWidthPx, ctx.imageWidth, ctx.imageHeight);
  const alphaTexture = new THREE.CanvasTexture(maskCanvas);
  alphaTexture.wrapS = THREE.ClampToEdgeWrapping;
  alphaTexture.wrapT = THREE.ClampToEdgeWrapping;

  const material = new THREE.MeshStandardMaterial({
    map: texture,
    alphaMap: alphaTexture,
    transparent: true,
    roughness: 0.95,
    metalness: 0.0,
    side: THREE.DoubleSide,
    alphaTest: 0.02,
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.z = -params.pivotZRatio;

  const group = new THREE.Group();
  group.position.z = params.pivotZRatio;
  group.add(mesh);

  return {
    group,
    mesh,
    geometry,
    ellipse,
    cols,
    rows,
    bounds,
    basePositions: positions.slice(),
    maskValues,
    hairMaskValues,
    debug: {
      faceDepth: faceDepthDbg,
      headDepth: headDepthDbg,
      edgeRolloff: edgeRolloffDbg,
      hairVolume: hairVolumeDbg,
      finalDepth: finalDepthDbg,
    },
  };
}

function computeZForMode(mode: FullHeadMode, zHeadOutside: number, blended: number, hairVolume: number): number {
  switch (mode) {
    case 'HEAD_DEPTH_ONLY':
      return zHeadOutside;
    case 'FACE_HEAD':
      return blended;
    case 'FACE_HEAD_HAIR':
      return blended + hairVolume;
    default:
      return blended;
  }
}

function buildGridIndices(cols: number, rows: number): Uint32Array {
  const indices = new Uint32Array((cols - 1) * (rows - 1) * 6);
  let p = 0;
  for (let row = 0; row < rows - 1; row++) {
    for (let col = 0; col < cols - 1; col++) {
      const a = row * cols + col;
      const b = a + 1;
      const c = a + cols;
      const d = c + 1;
      indices[p++] = a;
      indices[p++] = c;
      indices[p++] = b;
      indices[p++] = b;
      indices[p++] = c;
      indices[p++] = d;
    }
  }
  return indices;
}

/**
 * GUIパラメータ変更時にDepthのみ再計算する (grid/UV/maskテクスチャは再利用)。
 * maskSource / depthSourceの切替はgrid境界とalphaMapが変わるため、
 * ここではなくbuildHeadGridGeometryからの再構築で行うこと (main.ts側の責務)。
 */
export function recomputeFullHeadDepth(build: FullHeadBuild, ctx: FullHeadBuildContext, params: Params): void {
  const evaluate = createVertexEvaluator(ctx, build.ellipse, params);
  const posAttr = build.geometry.getAttribute('position') as THREE.BufferAttribute;
  const uvAttr = build.geometry.getAttribute('uv') as THREE.BufferAttribute;

  for (let row = 0; row < build.rows; row++) {
    for (let col = 0; col < build.cols; col++) {
      const idx = row * build.cols + col;
      const x = build.basePositions[idx * 3 + 0];
      const y = build.basePositions[idx * 3 + 1];
      const u = uvAttr.getX(idx);
      const v = uvAttr.getY(idx);

      const ch = evaluate(x, y, u, v);

      build.hairMaskValues[idx] = ch.hairMask;
      build.debug.faceDepth[idx] = ch.faceSideZ;
      build.debug.headDepth[idx] = ch.zOutside;
      build.debug.edgeRolloff[idx] = ch.zRolloff;
      build.debug.hairVolume[idx] = ch.hairVolume;

      const zByMode = computeZForMode(params.fullHeadMode, ch.zOutside, ch.blended, ch.hairVolume);
      build.debug.finalDepth[idx] = zByMode;
      build.basePositions[idx * 3 + 2] = zByMode;
      posAttr.setZ(idx, zByMode);
    }
  }
  posAttr.needsUpdate = true;
  build.mesh.position.z = -params.pivotZRatio;
  build.group.position.z = params.pivotZRatio;
}
