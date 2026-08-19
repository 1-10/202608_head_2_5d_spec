// FULL HEAD: Head Grid Mesh。Face Depth Field + Pseudo Head Depth + Edge Rolloff +
// Hair Volume + Head Silhouette Maskを1枚の連続したメッシュへ合成する。
// Face MeshとHead Gridのtopologyは分離しており、Head Gridは常に一定間隔の格子。

import * as THREE from 'three';
import { FACE_KEY_INDICES, type FaceTriangulation, type NormalizedFaceLandmark } from './faceTopology';
import { type FaceDepthField, sampleFaceDepthField } from './faceDepth';
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
} from './headDepth';
import type { FullHeadMode, Params } from './params';

export interface FullHeadBuildContext {
  landmarks: NormalizedFaceLandmark[];
  triangulation: FaceTriangulation;
  faceZFinal: Float32Array; // FACE ONLYと共通のFace Depth (canonical+mediapipe混合済み)
  depthField: FaceDepthField;
  headCenterPx: { x: number; y: number };
  faceWidthPx: number;
  imageWidth: number;
  imageHeight: number;
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

export function buildHeadGridGeometry(ctx: FullHeadBuildContext, texture: THREE.Texture, params: Params): FullHeadBuild {
  const ellipse = estimateHeadMaskEllipse(ctx.landmarks);
  const cols = params.headGridCols;
  const rows = params.headGridRows;

  const margin = 1.18;
  const bounds = {
    xMin: ellipse.cx - ellipse.rx * margin,
    xMax: ellipse.cx + ellipse.rx * margin,
    yMin: -0.35, // 顎下あたりまで(首は含めない)
    yMax: ellipse.cy + ellipse.ry * margin,
  };

  const hull = ctx.triangulation.hull.map((i) => ({ x: ctx.landmarks[i].x, y: ctx.landmarks[i].y }));
  const k = FACE_KEY_INDICES;
  const browIndices = [...k.eyebrowA, ...k.eyebrowB];
  const browY = landmarkAvgY(ctx.landmarks, browIndices);
  const zBrowConst = faceZAvg(ctx.faceZFinal, browIndices);
  const eyeLineY = landmarkAvgY(ctx.landmarks, [k.eyeA.outer, k.eyeA.inner, k.eyeB.outer, k.eyeB.inner]);
  const headTopY = ellipse.cy + ellipse.ry;
  const blendWidth = params.blendWidthRatio;

  const count = cols * rows;
  const positions = new Float32Array(count * 3);
  const uvs = new Float32Array(count * 2);
  const maskValues = new Float32Array(count);
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

      const maskVal = sampleHeadMask(x, y, ellipse);
      const headResult = computeHeadDepthFinal(x, y, ellipse, params.headDepthScale, params.edgeStart, params.edgeDepth);
      const wFace = computeFaceHeadBlendWeight(x, y, hull, blendWidth);

      const sample = sampleFaceDepthField(ctx.depthField, u, v);
      let faceSideZ = sample.coverage > 0.5 ? sample.depth : headResult.zHeadFinal;
      if (y > browY) {
        const t = computeForeheadWeight(y, browY, headTopY);
        faceSideZ = computeForeheadDepth(zBrowConst, headResult.zHeadFinal, t);
      }

      const blended = faceSideZ * wFace + headResult.zHeadFinal * (1 - wFace);

      const hairMask = maskVal * (1 - wFace);
      const verticalT = (y - eyeLineY) / Math.max(1e-6, headTopY - eyeLineY);
      const hairVolume = computeHairVolume(hairMask, verticalT, params.hairVolumeMax);

      positions[idx * 3 + 0] = x;
      positions[idx * 3 + 1] = y;
      uvs[idx * 2 + 0] = u;
      uvs[idx * 2 + 1] = v;
      maskValues[idx] = maskVal;

      faceDepthDbg[idx] = faceSideZ;
      headDepthDbg[idx] = headResult.zHeadFinal;
      edgeRolloffDbg[idx] = headResult.zRolloff;
      hairVolumeDbg[idx] = hairVolume;

      const zByMode = computeZForMode(params.fullHeadMode, headResult.zHeadFinal, blended, hairVolume);
      positions[idx * 3 + 2] = zByMode;
      finalDepthDbg[idx] = zByMode;
    }
  }

  const indices = buildGridIndices(cols, rows);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeVertexNormals();

  const maskCanvas = rasterizeHeadMaskCanvas(ellipse, ctx.headCenterPx, ctx.faceWidthPx, ctx.imageWidth, ctx.imageHeight);
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
    debug: {
      faceDepth: faceDepthDbg,
      headDepth: headDepthDbg,
      edgeRolloff: edgeRolloffDbg,
      hairVolume: hairVolumeDbg,
      finalDepth: finalDepthDbg,
    },
  };
}

function computeZForMode(mode: FullHeadMode, zHeadFinal: number, blended: number, hairVolume: number): number {
  switch (mode) {
    case 'HEAD_DEPTH_ONLY':
      return zHeadFinal;
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

/** GUIパラメータ変更時にDepthのみ再計算する (grid/UV/maskは再利用)。 */
export function recomputeFullHeadDepth(build: FullHeadBuild, ctx: FullHeadBuildContext, params: Params): void {
  const hull = ctx.triangulation.hull.map((i) => ({ x: ctx.landmarks[i].x, y: ctx.landmarks[i].y }));
  const k = FACE_KEY_INDICES;
  const browIndices = [...k.eyebrowA, ...k.eyebrowB];
  const browY = landmarkAvgY(ctx.landmarks, browIndices);
  const zBrowConst = faceZAvg(ctx.faceZFinal, browIndices);
  const eyeLineY = landmarkAvgY(ctx.landmarks, [k.eyeA.outer, k.eyeA.inner, k.eyeB.outer, k.eyeB.inner]);
  const headTopY = build.ellipse.cy + build.ellipse.ry;
  const blendWidth = params.blendWidthRatio;

  const posAttr = build.geometry.getAttribute('position') as THREE.BufferAttribute;
  const uvAttr = build.geometry.getAttribute('uv') as THREE.BufferAttribute;

  for (let row = 0; row < build.rows; row++) {
    for (let col = 0; col < build.cols; col++) {
      const idx = row * build.cols + col;
      const x = build.basePositions[idx * 3 + 0];
      const y = build.basePositions[idx * 3 + 1];

      const headResult = computeHeadDepthFinal(x, y, build.ellipse, params.headDepthScale, params.edgeStart, params.edgeDepth);
      const wFace = computeFaceHeadBlendWeight(x, y, hull, blendWidth);

      const u = uvAttr.getX(idx);
      const v = uvAttr.getY(idx);
      const sample = sampleFaceDepthField(ctx.depthField, u, v);
      let faceSideZ = sample.coverage > 0.5 ? sample.depth : headResult.zHeadFinal;
      if (y > browY) {
        const t = computeForeheadWeight(y, browY, headTopY);
        faceSideZ = computeForeheadDepth(zBrowConst, headResult.zHeadFinal, t);
      }

      const blended = faceSideZ * wFace + headResult.zHeadFinal * (1 - wFace);
      const hairMask = build.maskValues[idx] * (1 - wFace);
      const verticalT = (y - eyeLineY) / Math.max(1e-6, headTopY - eyeLineY);
      const hairVolume = computeHairVolume(hairMask, verticalT, params.hairVolumeMax);

      build.debug.faceDepth[idx] = faceSideZ;
      build.debug.headDepth[idx] = headResult.zHeadFinal;
      build.debug.edgeRolloff[idx] = headResult.zRolloff;
      build.debug.hairVolume[idx] = hairVolume;

      const zByMode = computeZForMode(params.fullHeadMode, headResult.zHeadFinal, blended, hairVolume);
      build.debug.finalDepth[idx] = zByMode;
      build.basePositions[idx * 3 + 2] = zByMode;
      posAttr.setZ(idx, zByMode);
    }
  }
  posAttr.needsUpdate = true;
  build.geometry.computeVertexNormals();
  build.mesh.position.z = -params.pivotZRatio;
  build.group.position.z = params.pivotZRatio;
}
