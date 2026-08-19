// GNM Headバックエンドのメッシュ構築 (真3D頭部 + 実測髪シェルのハイブリッド)。
//
// 構成:
// - Head: GNMフィット結果の真3Dメッシュ。正面写真を頂点UVへ平行投影し、
//   背面/グレージング/シルエット外は「マスク内へクランプした写真色」(頂点色) へフェード。
// - Hair Shell: 実測髪マスク+実測Depthの前面シェルをGNMの手前に重ねる。
//   Depthのスケールは既存reliefのfaceZFinalではなく「フィット済GNM表面のz」へ
//   最小二乗で合わせる (頭部が実比率の奥行きを持つため)。
// - 髪で覆われた人は髪シェルのalphaがGNMの耳/頭皮を手前で隠し、
//   耳が出ている人はGNMの真3D耳が見える (Z順で自動解決。ケース分岐なし)。

import * as THREE from 'three';
import { sampleField, fieldBoundsUv, type ScalarField } from './fields';
import { MEDIAPIPE_IBUG68, fitGnmToLandmarks, type GnmFitResult, type GnmModel } from './gnmHead';
import { smoothstep } from './headDepth';
import { applyFlatNormals, buildGridIndices } from './meshUtils';
import { rasterizeMaskCanvas } from './personSegmentation';
import { selectDepth, selectSegmentation, type FullHeadBuildContext } from './fullHeadMesh';
import type { Params } from './params';

export interface GnmHeadBuild {
  group: THREE.Group;
  headMesh: THREE.Mesh;
  hairMesh: THREE.Mesh | null;
  fit: GnmFitResult;
  dispose(): void;
}

const UV_CLAMP_STEPS = 80; // シルエット外UVを頭部中心へ歩かせる最大ステップ数

export function buildGnmHead(
  model: GnmModel,
  ctx: FullHeadBuildContext,
  sourceCanvas: HTMLCanvasElement,
  texture: THREE.Texture,
  params: Params,
): GnmHeadBuild {
  const fit = fitGnmToLandmarks(model, ctx.landmarks, params.gnmIdentityReg);
  const seg = selectSegmentation(ctx, params);

  // --- Head geometry ---
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(fit.vertices, 3));
  geometry.setIndex(new THREE.BufferAttribute(model.triangles, 1));
  geometry.computeVertexNormals(); // 実法線 (投影重み計算用。描画前にflat化する)
  const realNormals = (geometry.getAttribute('normal') as THREE.BufferAttribute).array as Float32Array;

  const n = model.vertexCount;
  const uvs = new Float32Array(n * 2);
  const fallback = new Float32Array(n * 3);
  const photoW = new Float32Array(n);

  const img = sourceCanvas.getContext('2d')!.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height);
  const centerU = ctx.headCenterPx.x / ctx.imageWidth;
  const centerV = 1 - ctx.headCenterPx.y / ctx.imageHeight;
  const linear = new THREE.Color();

  for (let i = 0; i < n; i++) {
    const x = fit.vertices[i * 3];
    const y = fit.vertices[i * 3 + 1];
    const nz = realNormals[i * 3 + 2];

    // モデル空間 → 画像UV (平行投影)
    let u = (x * ctx.faceWidthPx + ctx.headCenterPx.x) / ctx.imageWidth;
    let v = 1 - (ctx.headCenterPx.y - y * ctx.faceWidthPx) / ctx.imageHeight;

    // シルエット外のUVは頭部中心方向へ歩かせてマスク内へクランプ (edge-extend)。
    // 歩幅は細かく取る — 頭頂では髪の帯が薄く、粗い歩幅だと帯を飛び越えて
    // 額の肌色を拾ってしまう (頭頂が禿げて見えるバグの原因)
    let maskAtUv = 1;
    if (seg) {
      maskAtUv = sampleField(seg.person, u, v);
      if (maskAtUv < 0.5) {
        for (let s = 0; s < UV_CLAMP_STEPS; s++) {
          u += (centerU - u) * 0.03;
          v += (centerV - v) * 0.03;
          if (sampleField(seg.person, u, v) >= 0.5) break;
        }
      }
    }
    uvs[i * 2] = u;
    uvs[i * 2 + 1] = v;

    // 前面かつシルエット内でのみ写真テクスチャを使い、それ以外は頂点色へフェード
    photoW[i] = smoothstep(0.08, 0.4, nz) * (seg ? smoothstep(0.2, 0.5, maskAtUv) : 1);

    // fallback頂点色: クランプ済みUVの3x3平均 (sRGB→linear)
    const px = Math.min(img.width - 2, Math.max(1, Math.round(u * img.width)));
    const py = Math.min(img.height - 2, Math.max(1, Math.round((1 - v) * img.height)));
    let r = 0;
    let g = 0;
    let b = 0;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const o = ((py + dy) * img.width + px + dx) * 4;
        r += img.data[o];
        g += img.data[o + 1];
        b += img.data[o + 2];
      }
    }
    linear.setRGB(r / 9 / 255, g / 9 / 255, b / 9 / 255, THREE.SRGBColorSpace);
    fallback[i * 3] = linear.r;
    fallback[i * 3 + 1] = linear.g;
    fallback[i * 3 + 2] = linear.b;
  }

  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geometry.setAttribute('aFallback', new THREE.BufferAttribute(fallback, 3));
  geometry.setAttribute('aPhotoW', new THREE.BufferAttribute(photoW, 1));
  applyFlatNormals(geometry); // reliefと同じ「写真の陰影のみ」のライティング方針に揃える

  const headMaterial = new THREE.MeshStandardMaterial({
    map: texture,
    roughness: 0.95,
    metalness: 0.0,
  });
  patchPhotoMixShader(headMaterial);

  const headMesh = new THREE.Mesh(geometry, headMaterial);

  // --- Hair shell (実測髪マスク+実測Depth) ---
  const hair = buildHairShell(ctx, texture, fit, params);

  // 回転pivotは頭部の重心z (真3Dなのでreliefのpivot比率ではなく実重心を使う)
  const pivotZ = fit.centerZ;
  headMesh.position.z = -pivotZ;
  if (hair) hair.mesh.position.z = -pivotZ;

  const group = new THREE.Group();
  group.position.z = pivotZ;
  group.add(headMesh);
  if (hair) group.add(hair.mesh);

  return {
    group,
    headMesh,
    hairMesh: hair?.mesh ?? null,
    fit,
    dispose() {
      geometry.dispose();
      headMaterial.dispose();
      if (hair) {
        hair.mesh.geometry.dispose();
        hair.alphaTexture.dispose();
        (hair.mesh.material as THREE.Material).dispose();
      }
    },
  };
}

/** map色とfallback頂点色をaPhotoWでmixするようMeshStandardMaterialへパッチする。 */
function patchPhotoMixShader(material: THREE.MeshStandardMaterial): void {
  material.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace(
        'void main() {',
        'attribute vec3 aFallback;\nattribute float aPhotoW;\nvarying vec3 vFallback;\nvarying float vPhotoW;\nvoid main() {\n\tvFallback = aFallback;\n\tvPhotoW = aPhotoW;',
      );
    shader.fragmentShader = shader.fragmentShader
      .replace('void main() {', 'varying vec3 vFallback;\nvarying float vPhotoW;\nvoid main() {')
      .replace(
        '#include <map_fragment>',
        `#ifdef USE_MAP
\tvec4 sampledDiffuseColor = texture2D( map, vMapUv );
\tdiffuseColor.rgb *= mix( vFallback, sampledDiffuseColor.rgb, vPhotoW );
#endif`,
      );
  };
  material.customProgramCacheKey = () => 'gnm-photo-mix';
}

interface HairShellBuild {
  mesh: THREE.Mesh;
  alphaTexture: THREE.CanvasTexture;
}

/**
 * 実測髪マスク+実測Depthの前面髪シェルを作る。
 * Depthの相対値はランドマーク位置の「フィット済GNM表面z」への最小二乗で
 * モデル空間zへ写像する (実比率スケール)。
 */
function buildHairShell(
  ctx: FullHeadBuildContext,
  texture: THREE.Texture,
  fit: GnmFitResult,
  params: Params,
): HairShellBuild | null {
  const seg = selectSegmentation(ctx, params);
  const selected = selectDepth(ctx, params);
  if (!seg || !selected) return null;
  const depth = selected.depth;

  const hairFit = fitDepthToGnmZ(depth, ctx, fit);
  if (!hairFit) return null;

  const uvBounds = fieldBoundsUv(seg.hair, 0.08);
  if (!uvBounds) return null; // 髪が写っていない (スキンヘッド等) → GNM単体で成立する

  const toX = (u: number) => (u * ctx.imageWidth - ctx.headCenterPx.x) / ctx.faceWidthPx;
  const toY = (v: number) => (ctx.headCenterPx.y - (1 - v) * ctx.imageHeight) / ctx.faceWidthPx;
  const marginX = (toX(uvBounds.uMax) - toX(uvBounds.uMin)) * 0.05;
  const marginY = (toY(uvBounds.vMax) - toY(uvBounds.vMin)) * 0.05;
  const xMin = toX(uvBounds.uMin) - marginX;
  const xMax = toX(uvBounds.uMax) + marginX;
  const yMin = toY(uvBounds.vMin) - marginY;
  const yMax = toY(uvBounds.vMax) + marginY;

  const cols = params.headGridCols;
  const rows = params.headGridRows;
  const positions = new Float32Array(cols * rows * 3);
  const uvs = new Float32Array(cols * rows * 2);
  const maskPerVertex = new Float32Array(cols * rows);

  // フィット済GNM表面のzバッファ (XYビンごとの最前面z)。
  // 髪シェルは「頭皮z + 実測髪厚」でアンカーする — Depthフィットの外挿を
  // そのままzに使うと頭頂で過大になり、シェルが頭蓋から浮くため。
  const scalp = buildScalpZBuffer(fit.vertices, { xMin, xMax, yMin, yMax });
  // 厚みを厚くしすぎるとピッチ回転時にシェルと頭皮の隙間が下から見える
  const maxThickness = 0.16; // モデル空間 (faceWidth≈1) での髪厚上限
  const minThickness = 0.02;

  for (let row = 0; row < rows; row++) {
    const y = yMax + (yMin - yMax) * (row / (rows - 1));
    for (let col = 0; col < cols; col++) {
      const x = xMin + (xMax - xMin) * (col / (cols - 1));
      const idx = row * cols + col;
      const u = (x * ctx.faceWidthPx + ctx.headCenterPx.x) / ctx.imageWidth;
      const v = 1 - (ctx.headCenterPx.y - y * ctx.faceWidthPx) / ctx.imageHeight;

      const hairMask = sampleField(seg.hair, u, v);
      maskPerVertex[idx] = hairMask;
      const d = sampleField(depth, u, v);
      const zMeasured = (d * hairFit.scale + hairFit.offset) * params.measuredDepthGain;
      const scalpZ = scalp(x, y);
      const thickness = Math.min(maxThickness, Math.max(minThickness, zMeasured - scalpZ));
      // feather帯では厚みを頭皮へ絞る (縁の浮き対策。rolloffは絞り増強として作用)
      const edge = smoothstep(0.08, 0.5, hairMask);
      const z = scalpZ + params.gnmHairLift + thickness * edge - params.gnmHairRolloff * (1 - edge);

      positions[idx * 3] = x;
      positions[idx * 3 + 1] = y;
      positions[idx * 3 + 2] = z;
      uvs[idx * 2] = u;
      uvs[idx * 2 + 1] = v;
    }
  }

  // Depthノイズ (GNM実スケールで増幅) をグリッド空間で平滑化する
  for (let pass = 0; pass < 2; pass++) {
    const src = new Float32Array(maskPerVertex.length);
    for (let i = 0; i < maskPerVertex.length; i++) src[i] = positions[i * 3 + 2];
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        let sum = 0;
        let count = 0;
        for (let dr = -1; dr <= 1; dr++) {
          const rr = row + dr;
          if (rr < 0 || rr >= rows) continue;
          for (let dc = -1; dc <= 1; dc++) {
            const cc = col + dc;
            if (cc < 0 || cc >= cols) continue;
            sum += src[rr * cols + cc];
            count++;
          }
        }
        positions[(row * cols + col) * 3 + 2] = sum / count;
      }
    }
  }

  // マスク外へはみ出すコーナーを含む三角形は張らない。境界セルの三角形が
  // グレージング視で横倒しになり「鋸歯状のスパイク」として見えるため、
  // 全コーナーがマスク内のセルだけ残す (縁のフェードはalphaMap+alphaTestに任せる)
  const gridIndices = buildGridIndices(cols, rows);
  const kept: number[] = [];
  for (let t = 0; t < gridIndices.length; t += 3) {
    const m = Math.min(
      maskPerVertex[gridIndices[t]],
      maskPerVertex[gridIndices[t + 1]],
      maskPerVertex[gridIndices[t + 2]],
    );
    if (m > 0.02) kept.push(gridIndices[t], gridIndices[t + 1], gridIndices[t + 2]);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(kept), 1));
  applyFlatNormals(geometry);

  const alphaTexture = new THREE.CanvasTexture(rasterizeMaskCanvas(seg.hair, 512));
  alphaTexture.wrapS = THREE.ClampToEdgeWrapping;
  alphaTexture.wrapT = THREE.ClampToEdgeWrapping;

  const material = new THREE.MeshStandardMaterial({
    map: texture,
    alphaMap: alphaTexture,
    transparent: true,
    roughness: 0.95,
    metalness: 0.0,
    side: THREE.DoubleSide,
    // GRIDより高め: シェル縁は頭蓋の外に浮くため、feather裾の薄い断面が
    // グレージング視で筋状に見える。裾を早めに切って背後のGNMに任せる
    alphaTest: 0.3,
  });

  return { mesh: new THREE.Mesh(geometry, material), alphaTexture };
}

const SCALP_BINS_X = 96;
const SCALP_BINS_Y = 112;

/**
 * フィット済GNM頂点をXYビンへ分配し、各ビンの最前面z (最大z) を持つ
 * 「頭皮zバッファ」を作る。空ビンはBFSで最寄りの値を伝播して埋めるため、
 * 髪がGNMシルエットの外へはみ出す画素でも連続したzが返る。
 */
function buildScalpZBuffer(
  verts: Float32Array,
  bounds: { xMin: number; xMax: number; yMin: number; yMax: number },
): (x: number, y: number) => number {
  const { xMin, xMax, yMin, yMax } = bounds;
  const w = SCALP_BINS_X;
  const h = SCALP_BINS_Y;
  const data = new Float32Array(w * h).fill(-Infinity);

  const spanX = Math.max(1e-6, xMax - xMin);
  const spanY = Math.max(1e-6, yMax - yMin);
  for (let i = 0; i < verts.length; i += 3) {
    const bx = Math.floor(((verts[i] - xMin) / spanX) * w);
    const by = Math.floor(((verts[i + 1] - yMin) / spanY) * h);
    if (bx < 0 || bx >= w || by < 0 || by >= h) continue;
    const idx = by * w + bx;
    if (verts[i + 2] > data[idx]) data[idx] = verts[i + 2];
  }

  // 空ビンをBFSで埋める (最寄りの既知zを伝播)
  const known = new Uint8Array(w * h);
  const queue = new Int32Array(w * h);
  let tail = 0;
  for (let i = 0; i < w * h; i++) {
    if (data[i] !== -Infinity) {
      known[i] = 1;
      queue[tail++] = i;
    }
  }
  if (tail === 0) return () => 0;
  let head = 0;
  while (head < tail) {
    const i = queue[head++];
    const bx = i % w;
    const by = (i / w) | 0;
    const z = data[i];
    if (bx > 0 && !known[i - 1]) {
      known[i - 1] = 1;
      data[i - 1] = z;
      queue[tail++] = i - 1;
    }
    if (bx < w - 1 && !known[i + 1]) {
      known[i + 1] = 1;
      data[i + 1] = z;
      queue[tail++] = i + 1;
    }
    if (by > 0 && !known[i - w]) {
      known[i - w] = 1;
      data[i - w] = z;
      queue[tail++] = i - w;
    }
    if (by < h - 1 && !known[i + w]) {
      known[i + w] = 1;
      data[i + w] = z;
      queue[tail++] = i + w;
    }
  }

  return (x: number, y: number) => {
    const bx = Math.min(w - 1, Math.max(0, Math.floor(((x - xMin) / spanX) * w)));
    const by = Math.min(h - 1, Math.max(0, Math.floor(((y - yMin) / spanY) * h)));
    return data[by * w + bx];
  };
}

/** 相対Depth→モデル空間z (GNM表面スケール) の線形フィット。68点ランドマークで解く。 */
function fitDepthToGnmZ(
  depth: ScalarField,
  ctx: FullHeadBuildContext,
  fit: GnmFitResult,
): { scale: number; offset: number } | null {
  let n = 0;
  let sumD = 0;
  let sumZ = 0;
  let sumDD = 0;
  let sumDZ = 0;
  for (let k = 0; k < MEDIAPIPE_IBUG68.length; k++) {
    const lm = ctx.landmarks[MEDIAPIPE_IBUG68[k]];
    const { u0, v0, u1, v1 } = depth.rect;
    if (lm.u < u0 || lm.u > u1 || lm.v < v0 || lm.v > v1) continue;
    const d = sampleField(depth, lm.u, lm.v);
    const z = fit.landmarkZ[k];
    n++;
    sumD += d;
    sumZ += z;
    sumDD += d * d;
    sumDZ += d * z;
  }
  if (n < 20) return null;
  const denom = n * sumDD - sumD * sumD;
  if (Math.abs(denom) < 1e-9) return null;
  const scale = (n * sumDZ - sumD * sumZ) / denom;
  const offset = (sumZ - scale * sumD) / n;
  if (!Number.isFinite(scale) || !Number.isFinite(offset)) return null;
  return { scale, offset };
}
