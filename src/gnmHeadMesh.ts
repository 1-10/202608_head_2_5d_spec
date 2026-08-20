// GNM Headのメッシュ構築 (真3D頭部 + 実測髪シェルのハイブリッド)。
//
// 構成:
// - Head: GNMフィット結果の真3Dメッシュ。正面写真を頂点UVへ平行投影し、
//   背面/グレージング/シルエット外は「マスク内へクランプした写真色」(頂点色) へフェード。
// - Hair Shell: 実測髪マスク+実測Depthの前面シェルをGNMの手前に重ねる。
//   Depthのスケールは「フィット済GNM表面のz」へ最小二乗で合わせる
//   (頭部が実比率の奥行きを持つため)。
// - 髪で覆われた人は髪シェルのalphaがGNMの耳/頭皮を手前で隠し、
//   耳が出ている人はGNMの真3D耳が見える (Z順で自動解決。ケース分岐なし)。

import * as THREE from 'three';
import { sampleField, fieldBoundsUv, type ScalarField } from './fields';
import type { NormalizedFaceLandmark } from './faceTopology';
import {
  MEDIAPIPE_IBUG68,
  applySimilarityInPlace,
  fitGnmToLandmarks,
  type GnmFitResult,
  type GnmModel,
  type SimilarityTransform,
} from './gnmHead';
import { GNM_EXPRESSION_PRESETS } from './gnmExpressions';
import { MP_EYES, MP_LIPS, applyResidualWarp, fillNostrils } from './gnmRefine';
import { buildHairFreeFaceCanvas } from './hairFill';
import { applyFlatNormals, buildGridIndices, smoothstep } from './meshUtils';
import { rasterizeMaskCanvas, type SegmentationResult } from './personSegmentation';
import type { Params } from './params';

/**
 * 実測/ニューラルソース一式。取得に失敗した(または未取得の)ものはnull。
 * NEURAL系がnullのままNEURALを選ぶとMEASURED系へフォールバックする。
 */
export interface MeasuredHeadData {
  segmentation: SegmentationResult | null; // MediaPipe SelfieMulticlass
  depth: ScalarField | null; // ARPortraitDepth 相対Depth (0-1)
  neuralSegmentation: SegmentationResult | null; // BiRefNet×MediaPipe合成 (遅延取得)
  neuralDepth: ScalarField | null; // Depth Anything V2 (遅延取得)
}

/** GNM頭部の構築に必要な入力一式 (画像1枚から導出される)。 */
export interface GnmBuildContext {
  landmarks: NormalizedFaceLandmark[];
  headCenterPx: { x: number; y: number };
  faceWidthPx: number;
  imageWidth: number;
  imageHeight: number;
  measured: MeasuredHeadData | null;
}

/** maskSourceに応じたセグメンテーションを選ぶ (NEURAL未取得時はMEASUREDへフォールバック)。 */
export function selectSegmentation(ctx: GnmBuildContext, params: Params): SegmentationResult | null {
  const m = ctx.measured;
  if (!m) return null;
  if (params.maskSource === 'NEURAL') return m.neuralSegmentation ?? m.segmentation;
  if (params.maskSource === 'MEASURED') return m.segmentation;
  return null;
}

/** depthSourceに応じたDepth場を選ぶ (NEURAL未取得時はMEASUREDへフォールバック)。 */
export function selectDepth(ctx: GnmBuildContext, params: Params): ScalarField | null {
  const m = ctx.measured;
  if (!m) return null;
  if (params.depthSource === 'NEURAL') return m.neuralDepth ?? m.depth;
  if (params.depthSource === 'MEASURED') return m.depth;
  return null;
}

export interface GnmHeadBuild {
  group: THREE.Group;
  headMesh: THREE.Mesh;
  hairMesh: THREE.Mesh | null;
  /** ランドマーク重畳デバッグ表示 (既定で非表示。Show Landmarksで切替)。 */
  landmarkOverlay: THREE.Object3D;
  fit: GnmFitResult;
  setNeutralExpression(): void;
  /** 表情係数を直接目標に設定する (長さexpressionCount。感情プリセット用)。 */
  setExpressionTarget(coeffs: ArrayLike<number>): void;
  /** 表情係数を即時適用する (遷移なし。成分ラベリングなどデバッグ用)。 */
  snapExpression(coeffs: ArrayLike<number>): void;
  /**
   * レンダーループから毎フレーム呼ぶ。目標表情へ滑らかに遷移する。
   * blinkAmount: 0-1のまばたき量 (公式WINK合成の閉眼ベクトルを表情に加算する)。
   */
  tickExpression(blinkAmount?: number): void;
  dispose(): void;
}

const UV_CLAMP_STEPS = 80; // シルエット外UVを頭部中心へ歩かせる最大ステップ数

export function buildGnmHead(
  model: GnmModel,
  ctx: GnmBuildContext,
  sourceCanvas: HTMLCanvasElement,
  texture: THREE.Texture,
  params: Params,
): GnmHeadBuild {
  const fit = fitGnmToLandmarks(model, ctx.landmarks, params.gnmIdentityReg, params.gnmDenseFit);
  const seg = selectSegmentation(ctx, params);

  // 額の髪焼き付き対策: headの投影テクスチャ・fallback色には「髪画素を肌色で
  // 埋めた写真」を使う。髪は髪シェル (元写真) だけに描かれ、視差が付いても
  // 「シェルの髪」と「肌に焼き付いた髪」が二重に見えない。
  // Show Hair off時は髪シェルを作らないため、全髪画素を置換したbald写真に切替
  let headCanvas = sourceCanvas;
  let headTexture = texture;
  let ownedHeadTexture: THREE.Texture | null = null;
  if ((params.gnmHairSkinFill || !params.gnmShowHair) && seg) {
    const filled = buildHairFreeFaceCanvas(
      sourceCanvas,
      seg,
      { landmarks: ctx.landmarks, faceWidthPx: ctx.faceWidthPx },
      params.gnmShowHair ? params.gnmHairFillStrength : 1,
      params.gnmShowHair ? 'overlay' : 'bald',
    );
    if (filled) {
      headCanvas = filled;
      const t = new THREE.CanvasTexture(filled);
      t.colorSpace = THREE.SRGBColorSpace;
      headTexture = t;
      ownedHeadTexture = t;
    }
  }

  // 残差ワープ: identity係数 (統計モデル) では張り切れない目・唇の位置残差を
  // neutral頂点へ焼き込む。まばたき・開口が「写真の目・口の位置」で起きる。
  // 返り値は目領域の表情成分の振幅スケール (ワープで瞼開口幅が変わった分の補正)
  const exprScales = applyResidualWarp(model, fit, ctx.landmarks, params.gnmWarpStrength);

  // 鼻孔: 内壁を平滑化で塞ぐ (穴のジオメトリは不要 — 写真の鼻孔の暗さで十分)
  fillNostrils(model, fit.vertices);

  // 口内側面 (唇のインナーロール) の重み。neutral時に他の面の背後に隠れており
  // 平行投影では正しい写真色が存在しない — シーム暗線が焼き付き、開口時に
  // 下唇の上へ「写真の合わせ目の線」として現れる。写真投影を切る判定に使う
  const mouthInteriorW = computeMouthInteriorWeights(model, fit.vertices, ctx);

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

  const img = headCanvas.getContext('2d')!.getImageData(0, 0, headCanvas.width, headCanvas.height);
  const centerU = ctx.headCenterPx.x / ctx.imageWidth;
  const centerV = 1 - ctx.headCenterPx.y / ctx.imageHeight;
  const linear = new THREE.Color();

  // 口内色: 下唇中央 (内唇14と下唇下端17の中点) の写真色を暗くした固定色。
  // 口内側面はこの頂点色で塗る (シーム暗線の縞ではなく一様な暗い口内に見せる)
  const lipCx = Math.min(img.width - 2, Math.max(1, Math.round((ctx.landmarks[14].px + ctx.landmarks[17].px) / 2)));
  const lipCy = Math.min(img.height - 2, Math.max(1, Math.round((ctx.landmarks[14].py + ctx.landmarks[17].py) / 2)));
  let lipR = 0;
  let lipG = 0;
  let lipB = 0;
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const o = ((lipCy + dy) * img.width + lipCx + dx) * 4;
      lipR += img.data[o];
      lipG += img.data[o + 1];
      lipB += img.data[o + 2];
    }
  }
  const interiorColor = new THREE.Color().setRGB(
    (lipR / 9 / 255) * 0.4,
    (lipG / 9 / 255) * 0.4,
    (lipB / 9 / 255) * 0.4,
    THREE.SRGBColorSpace,
  );

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

    // 前面かつシルエット内でのみ写真テクスチャを使い、それ以外は頂点色へフェード。
    // 口内側面は写真に正しい色が無いため投影を切る
    const iw = mouthInteriorW[i];
    photoW[i] = smoothstep(0.08, 0.4, nz) * (seg ? smoothstep(0.2, 0.5, maskAtUv) : 1) * (1 - iw);

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
    fallback[i * 3] = linear.r + (interiorColor.r - linear.r) * iw;
    fallback[i * 3 + 1] = linear.g + (interiorColor.g - linear.g) * iw;
    fallback[i * 3 + 2] = linear.b + (interiorColor.b - linear.b) * iw;

  }

  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geometry.setAttribute('aFallback', new THREE.BufferAttribute(fallback, 3));
  geometry.setAttribute('aPhotoW', new THREE.BufferAttribute(photoW, 1));
  applyFlatNormals(geometry); // ライティングは「写真の陰影のみ」の方針 (偽の影の帯を防ぐ)

  const headMaterial = new THREE.MeshStandardMaterial({
    map: headTexture,
    roughness: 0.95,
    metalness: 0.0,
  });
  patchPhotoMixShader(headMaterial);

  const headMesh = new THREE.Mesh(geometry, headMaterial);

  // --- Hair shell (実測髪マスク+実測Depth) ---
  const hair = params.gnmShowHair ? buildHairShell(ctx, texture, fit, params) : null;

  // 回転pivotは頭部の実重心z (真3Dのため固定比率ではなく実測で決める)
  const pivotZ = fit.centerZ;
  headMesh.position.z = -pivotZ;
  if (hair) hair.mesh.position.z = -pivotZ;

  const group = new THREE.Group();
  group.position.z = pivotZ;
  group.add(headMesh);
  if (hair) group.add(hair.mesh);

  // ランドマーク重畳デバッグ表示 (ワープ後のneutral頂点に対する残差を可視化)
  const landmarkOverlay = buildLandmarkOverlay(model, fit, ctx.landmarks);
  landmarkOverlay.object.visible = false;
  landmarkOverlay.object.position.z = -pivotZ;
  group.add(landmarkOverlay.object);

  // --- 表情 (GNM expression basis) ---
  // 未変換のneutral頂点を保持し、表情係数を足してから相似変換して差し替える。
  // UV/fallback/alphaはneutral時のまま使う (写真は表面に追従して動く)。
  // 注意: applyIdentityから作り直すと残差ワープなどの後処理が毎フレーム消えるため、
  // 最終頂点 (fit.vertices) を逆相似変換して作る
  const neutralUntransformed = invertSimilarity(fit.vertices, fit.sim);
  const exprCurrent = new Float32Array(model.expressionCount);
  const exprTarget = new Float32Array(model.expressionCount);
  const posAttr = geometry.getAttribute('position') as THREE.BufferAttribute;

  // まばたきベクトル: 公式ExpressionSamplerのWINK_LEFT+WINK_RIGHT合成 (目領域のみ)。
  // blinkAmountを乗算して感情表情へ加算する独立チャネル
  const blinkVec =
    GNM_EXPRESSION_PRESETS.blink?.length === model.expressionCount
      ? GNM_EXPRESSION_PRESETS.blink
      : new Array<number>(model.expressionCount).fill(0);
  let blinkNow = 0;

  const applyExpressionNow = (): void => {
    const out = new Float32Array(neutralUntransformed);
    for (let i = 0; i < model.expressionCount; i++) {
      const c = exprCurrent[i] + blinkVec[i] * blinkNow;
      if (c === 0) continue;
      // exprScales: 残差ワープで瞼開口幅が変わった分の目領域振幅補正
      const cs = (c * exprScales[i] * model.expressionScales[i]) / 32767;
      const base = i * model.vertexCount * 3;
      for (let j = 0; j < model.vertexCount * 3; j++) out[j] += model.expressionBasisQ[base + j] * cs;
    }
    applySimilarityInPlace(out, fit.sim);
    (posAttr.array as Float32Array).set(out);
    posAttr.needsUpdate = true;
  };

  return {
    group,
    headMesh,
    hairMesh: hair?.mesh ?? null,
    landmarkOverlay: landmarkOverlay.object,
    fit,
    setNeutralExpression() {
      exprTarget.fill(0);
    },
    setExpressionTarget(coeffs: ArrayLike<number>) {
      for (let i = 0; i < model.expressionCount; i++) exprTarget[i] = coeffs[i] ?? 0;
    },
    snapExpression(coeffs: ArrayLike<number>) {
      for (let i = 0; i < model.expressionCount; i++) {
        exprTarget[i] = coeffs[i] ?? 0;
        exprCurrent[i] = coeffs[i] ?? 0;
      }
      applyExpressionNow();
    },
    tickExpression(blinkAmount = 0) {
      let maxDiff = Math.abs(blinkAmount - blinkNow);
      for (let i = 0; i < model.expressionCount; i++) {
        const diff = exprTarget[i] - exprCurrent[i];
        if (Math.abs(diff) > maxDiff) maxDiff = Math.abs(diff);
      }
      if (maxDiff < 1e-3) return;
      const t = 0.06; // フレームごとの追従率 (指数的な滑らかな遷移。自動アニメーション向けに緩め)
      for (let i = 0; i < model.expressionCount; i++) {
        exprCurrent[i] += (exprTarget[i] - exprCurrent[i]) * t;
      }
      // まばたきは既にエンベロープ済みの値が来るため遅延なく反映する
      blinkNow = blinkAmount;
      applyExpressionNow();
    },
    dispose() {
      geometry.dispose();
      headMaterial.dispose();
      ownedHeadTexture?.dispose();
      landmarkOverlay.dispose();
      if (hair) {
        hair.mesh.geometry.dispose();
        hair.alphaTexture.dispose();
        (hair.mesh.material as THREE.Material).dispose();
      }
    },
  };
}

const MOUTH_INTERIOR_BINS_X = 96;
const MOUTH_INTERIOR_BINS_Y = 64;

/**
 * 口内側面 (唇のインナーロール) の重み (0-1) を求める。判定は2系統の合成 (max):
 * 1. 遮蔽判定: 口領域を(x,y)ビンへ分け、各ビンの最前面zより一定以上奥の頂点。
 *    唇の外側表面は各ビンで最前面になるため重み0、口の中へ巻き込む面だけが
 *    「同じ(x,y)で手前に別の面がある」状態になり重みが立つ。
 * 2. 開口縁判定: 口の開口境界リング (mouth sock除去の切断縁) と位相的に近い頂点。
 *    開口縁はneutral時に最前面 (シーム稜線上) なので遮蔽判定をすり抜けるが、
 *    UVはシーム暗帯を指しており、開口時に暗線として展開されるため口内扱いにする。
 */
function computeMouthInteriorWeights(
  model: GnmModel,
  verts: Float32Array,
  ctx: GnmBuildContext,
): Float32Array {
  const n = verts.length / 3;
  const w = new Float32Array(n);
  const lm = ctx.landmarks;
  // 口領域bbox: 61/291=口角, 0=上唇外上端, 17=下唇外下端 (MediaPipe index)。
  // 上マージンは小さく取る — 広げると鼻孔内壁 (これも遮蔽面) まで口内色になる
  const xMin = Math.min(lm[61].x, lm[291].x) - 0.02;
  const xMax = Math.max(lm[61].x, lm[291].x) + 0.02;
  const yMin = lm[17].y - 0.03;
  const yMax = lm[0].y + 0.02;
  const bx = MOUTH_INTERIOR_BINS_X;
  const by = MOUTH_INTERIOR_BINS_Y;
  const spanX = Math.max(1e-6, xMax - xMin);
  const spanY = Math.max(1e-6, yMax - yMin);

  const binOf = (i: number): number => {
    const ix = Math.floor(((verts[i * 3] - xMin) / spanX) * bx);
    const iy = Math.floor(((verts[i * 3 + 1] - yMin) / spanY) * by);
    if (ix < 0 || ix >= bx || iy < 0 || iy >= by) return -1;
    return iy * bx + ix;
  };

  const maxZ = new Float32Array(bx * by).fill(-Infinity);
  for (let i = 0; i < n; i++) {
    const b = binOf(i);
    if (b >= 0 && verts[i * 3 + 2] > maxZ[b]) maxZ[b] = verts[i * 3 + 2];
  }
  for (let i = 0; i < n; i++) {
    const b = binOf(i);
    if (b < 0) continue;
    // 閾値は唇表面の1ビン内z変化 (滑らかな面なら数mm相当) より大きく取り、
    // 曲率による誤検出を避ける
    w[i] = smoothstep(0.008, 0.02, maxZ[b] - verts[i * 3 + 2]);
  }

  // --- 開口縁判定: 口領域内の境界リングから位相BFSで近傍リングへ重みを立てる ---
  const inBox = (i: number): boolean => binOf(i) >= 0;
  const tris = model.triangles;
  const ekey = (a: number, b: number) => (a < b ? a * 65536 + b : b * 65536 + a);
  const edgeCount = new Map<number, number>();
  for (let t = 0; t < tris.length; t += 3) {
    for (let e = 0; e < 3; e++) {
      const k = ekey(tris[t + e], tris[t + ((e + 1) % 3)]);
      edgeCount.set(k, (edgeCount.get(k) ?? 0) + 1);
    }
  }
  const depth = new Int8Array(n).fill(-1);
  let frontier: number[] = [];
  for (const [k, c] of edgeCount) {
    if (c !== 1) continue;
    const a = Math.floor(k / 65536);
    const b = k % 65536;
    if (!inBox(a) || !inBox(b)) continue; // 口以外の境界 (メッシュ外周など) は除く
    if (depth[a] < 0) {
      depth[a] = 0;
      frontier.push(a);
    }
    if (depth[b] < 0) {
      depth[b] = 0;
      frontier.push(b);
    }
  }
  if (frontier.length > 0) {
    const neighbors = new Map<number, number[]>();
    for (let t = 0; t < tris.length; t += 3) {
      for (let e = 0; e < 3; e++) {
        const a = tris[t + e];
        const b = tris[t + ((e + 1) % 3)];
        if (!inBox(a) || !inBox(b)) continue;
        (neighbors.get(a) ?? neighbors.set(a, []).get(a)!).push(b);
        (neighbors.get(b) ?? neighbors.set(b, []).get(b)!).push(a);
      }
    }
    // リング=1.0, 1隣接=1.0, 2隣接=0.5 (縁の"濡れた縁"帯。以遠は写真の唇に任せる)
    const ringWeights = [1, 1, 0.5];
    for (let d = 1; d < ringWeights.length; d++) {
      const next: number[] = [];
      for (const i of frontier) {
        for (const nb of neighbors.get(i) ?? []) {
          if (depth[nb] < 0) {
            depth[nb] = d;
            next.push(nb);
          }
        }
      }
      frontier = next;
    }
    for (let i = 0; i < n; i++) {
      if (depth[i] >= 0) w[i] = Math.max(w[i], ringWeights[depth[i]]);
    }
  }

  return w;
}

/**
 * ランドマーク重畳デバッグ表示を作る。
 * 色点 = 写真のランドマーク位置 (残差ワープの目標。唇=赤 / 目=シアン / その他=黄)、
 * 白線 = フィット+ワープ後のGNM表面対応点から写真位置への残差ベクトル。
 * 開口シームが赤点 (内唇) から外れていれば、フィット/ワープの残差が
 * 「開口位置ズレ」の原因だと切り分けられる。
 * zはワープ後表面のzを使う (透視投影の視差で写真とXY比較が狂わないように)。
 */
function buildLandmarkOverlay(
  model: GnmModel,
  fit: GnmFitResult,
  landmarks: NormalizedFaceLandmark[],
): { object: THREE.Object3D; dispose(): void } {
  const useDense = model.denseCount > 0;
  const corrCount = useDense ? model.denseCount : MEDIAPIPE_IBUG68.length;
  const corrIdx = useDense ? model.denseTriIndices : model.landmarkIndices;
  const corrBary = useDense ? model.denseBaryWeights : model.landmarkWeights;
  const corrMp = (k: number) => (useDense ? model.denseMpIndices[k] : MEDIAPIPE_IBUG68[k]);

  const points: number[] = [];
  const colors: number[] = [];
  const lines: number[] = [];
  const zLift = 0.01; // 表面と重ならないよう僅かに手前へ

  for (let k = 0; k < corrCount; k++) {
    const mp = corrMp(k);
    const lm = landmarks[mp];
    if (!lm) continue;
    let sx = 0;
    let sy = 0;
    let sz = 0;
    for (let j = 0; j < 3; j++) {
      const vi = corrIdx[k * 3 + j];
      const bw = corrBary[k * 3 + j];
      sx += fit.vertices[vi * 3] * bw;
      sy += fit.vertices[vi * 3 + 1] * bw;
      sz += fit.vertices[vi * 3 + 2] * bw;
    }
    const z = sz + zLift;
    points.push(lm.x, lm.y, z);
    if (MP_LIPS.has(mp)) colors.push(1, 0.15, 0.3);
    else if (MP_EYES.has(mp)) colors.push(0.15, 0.9, 1);
    else colors.push(1, 0.85, 0.2);
    lines.push(sx, sy, z, lm.x, lm.y, z);
  }

  const pointsGeo = new THREE.BufferGeometry();
  pointsGeo.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
  pointsGeo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  const pointsMat = new THREE.PointsMaterial({
    size: 0.012,
    vertexColors: true,
    sizeAttenuation: true,
    depthTest: false,
    transparent: true,
  });
  const pointsObj = new THREE.Points(pointsGeo, pointsMat);
  pointsObj.renderOrder = 10;

  const linesGeo = new THREE.BufferGeometry();
  linesGeo.setAttribute('position', new THREE.Float32BufferAttribute(lines, 3));
  const linesMat = new THREE.LineBasicMaterial({
    color: 0xffffff,
    depthTest: false,
    transparent: true,
    opacity: 0.7,
  });
  const linesObj = new THREE.LineSegments(linesGeo, linesMat);
  linesObj.renderOrder = 9;

  const object = new THREE.Group();
  object.add(linesObj, pointsObj);
  return {
    object,
    dispose() {
      pointsGeo.dispose();
      pointsMat.dispose();
      linesGeo.dispose();
      linesMat.dispose();
    },
  };
}

/** 相似変換の逆変換 (新しい配列を返す)。 */
function invertSimilarity(verts: Float32Array, sim: SimilarityTransform): Float32Array {
  const out = new Float32Array(verts.length);
  for (let i = 0; i < verts.length; i += 3) {
    const dx = verts[i] - sim.tx;
    const dy = verts[i + 1] - sim.ty;
    out[i] = (sim.cos * dx + sim.sin * dy) / sim.s;
    out[i + 1] = (-sim.sin * dx + sim.cos * dy) / sim.s;
    out[i + 2] = (verts[i + 2] - sim.tz) / sim.s;
  }
  return out;
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
  ctx: GnmBuildContext,
  texture: THREE.Texture,
  fit: GnmFitResult,
  params: Params,
): HairShellBuild | null {
  const seg = selectSegmentation(ctx, params);
  const depth = selectDepth(ctx, params);
  if (!seg || !depth) return null;

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

  const cols = params.hairGridCols;
  const rows = params.hairGridRows;
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
    // シェル縁は頭蓋の外に浮くため、feather裾の薄い断面が
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
  ctx: GnmBuildContext,
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
