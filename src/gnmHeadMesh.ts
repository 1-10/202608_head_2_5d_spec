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
import { buildMouthInterior, buildTongueDriver } from './gnmMouthInterior';
import { MP_EYES, MP_LIPS, applyResidualWarp, buildEyeballContainment, fillNostrils } from './gnmRefine';
import { applyFlatNormals, buildGridIndices, smoothstep } from './meshUtils';
import { rasterizeMaskCanvas, type SegmentationResult } from './personSegmentation';
import type { Params } from './params';

/**
 * 実測ソース一式。取得に失敗した(または未取得の)ものはnull。
 * DAVID系がnullのままDAVIDを選ぶと該当のGoogle系ソースへフォールバックする。
 */
export interface MeasuredHeadData {
  segmentation: SegmentationResult | null; // MediaPipe SelfieMulticlass (生 256px)
  /** 髪系マスクをGuided Filterで写真エッジへ整合させた版 (768px)。失敗・未実行はnull */
  segmentationRefined: SegmentationResult | null;
  depth: ScalarField | null; // ARPortraitDepth 相対Depth (0-1)
  // --- DAViD multi-task (1回の推論で同時取得。遅延ロード。商用クリーン) ---
  davidDepth: ScalarField | null; // 人物相対Depth
  // 表面法線 (RGBエンコード済みObjectSpaceNormalMap, 画像全体UV空間)
  davidNormalCanvas: HTMLCanvasElement | null;
  davidPerson: ScalarField | null; // ソフト前景 (crop外はSelfieMulticlassで補完済み)
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

/** maskSourceに応じたセグメンテーションを選ぶ。 */
export function selectSegmentation(ctx: GnmBuildContext, params: Params): SegmentationResult | null {
  const m = ctx.measured;
  if (!m) return null;
  if (params.maskSource !== 'SELFIE_MULTICLASS') return null;
  // Mask Refine off時 (と精細化失敗時) は生マスクを使う
  let seg = (params.gnmMaskRefine ? m.segmentationRefined : null) ?? m.segmentation;
  if (!seg) return null;
  // 髪シェル用マスクに帽子・メガネ (accessories) を含める。
  // 下流 (髪シェル・alphaMap・レイヤー画像・均一髪色) はすべて seg.hair を見るので、
  // ここで差し替えるだけで一貫して帽子込みになる
  if (params.gnmHairIncludeAccessories) seg = { ...seg, hair: seg.hairWithAccessories };
  // 人物シルエットはDAViDソフト前景を優先 (境界精度が高い。意味分けはMediaPipeのまま)
  if (params.personSource === 'DAVID' && m.davidPerson) seg = { ...seg, person: m.davidPerson };
  return seg;
}

/** depthSourceに応じたDepth場を選ぶ (DAVID未取得時はARPortraitDepthへフォールバック)。 */
export function selectDepth(ctx: GnmBuildContext, params: Params): ScalarField | null {
  const m = ctx.measured;
  if (!m) return null;
  if (params.depthSource === 'DAVID') return m.davidDepth ?? m.depth;
  if (params.depthSource === 'ARPORTRAIT_DEPTH') return m.depth;
  return null;
}

/** レイヤー分離プレビューの1枚。 */
export interface PreviewLayer {
  label: string;
  canvas: HTMLCanvasElement;
  /**
   * キャンバスが元写真と同じアスペクト比か。falseなら頭部クロップを掛けても
   * 縦横比が合わない (髪alphaMapは正方形へ引き伸ばしたものをGPUへ渡している)。
   */
  photoAspect: boolean;
}

export interface GnmHeadBuild {
  group: THREE.Group;
  headMesh: THREE.Mesh;
  hairMesh: THREE.Mesh | null;
  /** 口腔内 (口腔壁・歯・歯茎・舌)。旧アセットはnull */
  mouthInteriorMesh: THREE.Mesh | null;
  /** ランドマーク重畳デバッグ表示 (既定で非表示。Show Landmarksで切替)。 */
  landmarkOverlay: THREE.Object3D;
  /**
   * ランドマーク重畳の対応点を現在の表情姿勢へ引き直す。
   * 表情が静止しているとtickExpressionが早期returnするため、
   * 表示をonにした直後は明示的に呼ぶ必要がある。
   */
  refreshLandmarkOverlay(): void;
  /**
   * レイヤー分離プレビュー。元写真から最終出力までの加工工程を順に並べる。
   * 「実際に使っているもの」だけを入れる — 表示用に作り直した近似は入れない
   * (GPUへ渡しているテクスチャはそのキャンバス自体を渡す)。
   */
  previewLayers: PreviewLayer[];
  /** レイヤー画像プレビューの頭部クロップ (画像に対する割合 0-1, yは上から)。 */
  layerPreviewCrop: { x: number; y: number; w: number; h: number };
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

/** 公式サンプラー由来の表情ベクトル (main.tsが gnmSampler から作って渡す)。 */
export interface GnmExpressionVectors {
  /** まばたき (左右ウインクの合成の目領域だけ)。長さ = expressionCount */
  blink: number[];
}

export function buildGnmHead(
  model: GnmModel,
  ctx: GnmBuildContext,
  sourceCanvas: HTMLCanvasElement,
  texture: THREE.Texture,
  params: Params,
  vectors: GnmExpressionVectors,
): GnmHeadBuild {
  const fit = fitGnmToLandmarks(model, ctx.landmarks, params.gnmIdentityReg, params.gnmDenseFit);
  const seg = selectSegmentation(ctx, params);

  const headCanvas = sourceCanvas;
  const headTexture = texture;

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
  const lipPhotoColor = new THREE.Color().setRGB(
    lipR / 9 / 255,
    lipG / 9 / 255,
    lipB / 9 / 255,
    THREE.SRGBColorSpace,
  );
  const interiorColor = new THREE.Color().setRGB(
    (lipR / 9 / 255) * 0.4,
    (lipG / 9 / 255) * 0.4,
    (lipB / 9 / 255) * 0.4,
    THREE.SRGBColorSpace,
  );

  // 背面・遠距離クランプ領域の均一色 (口内色と同じ発想)。写真に色情報が
  // 無い領域をクランプ済UVの引き伸ばしスメアで塗ると背面が縞・ピンク染みに
  // なるため、写真から測った髪/肌の平均色 (暗め=陰) で塗り潰す
  const uniformHairColor = seg ? maskedAverageColor(sourceCanvas, seg.hair, 0.85) : null;
  const uniformSkinColor = seg ? maskedAverageColor(sourceCanvas, seg.faceSkin, 0.85) : null;

  /**
   * テクスチャを引いてよい領域 = 人物マスク (境界精度の高いDAViD前景が入りうる)。
   *
   * 服 (SelfieMulticlassクラス4) を除かないのは意図的。GNMの首〜肩は写真では実際に
   * 服なので、服の色が「正しい」色。実測 (test_portrait.jpg): 服へ投影される頂点は
   * 2,183個 (モデル空間 y -1.47〜-0.42)、除外すると広い面が顎の縁の画素へ潰れて
   * 縦縞のスメアになり明確に悪化した (実装して比較した上で撤回した)。
   * 背景が入る写真があるとしたら真因はpersonマスク側。
   */
  const textureMask = (u: number, v: number): number =>
    seg ? sampleField(seg.person, u, v) : 1;

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
    const uProjected = u;
    const vProjected = v;
    let maskAtUv = 1;
    if (seg) {
      // maskAtUvは「投影先」の値のまま保つ (歩いた後の値にはしない) —
      // 下のphotoWがこれを使い、シルエット外へ落ちた頂点の写真重みを落としている
      maskAtUv = textureMask(u, v);
      if (maskAtUv < 0.5) {
        for (let s = 0; s < UV_CLAMP_STEPS; s++) {
          u += (centerU - u) * 0.03;
          v += (centerV - v) * 0.03;
          if (textureMask(u, v) >= 0.5) break;
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

    // 写真に色情報が無い度合い: 真の背面 (nz<0) と、クランプで長距離歩いたUV。
    // その分だけ3x3平均 (スメア) を捨て、髪/肌の均一色へ寄せる。
    // 閾値は背面側に寄せる — 側面 (nz≈0) はyaw回転で普通に見える領域で、
    // 写真の頬色の方が均一色より自然なため
    if (uniformHairColor && uniformSkinColor && seg) {
      const walked = Math.hypot(u - uProjected, v - vProjected);
      const invalidW = Math.max(1 - smoothstep(-0.25, -0.02, nz), smoothstep(0.08, 0.25, walked));
      if (invalidW > 0) {
        const hs = smoothstep(0.2, 0.6, sampleField(seg.hair, u, v));
        const ur = uniformSkinColor.r + (uniformHairColor.r - uniformSkinColor.r) * hs;
        const ug = uniformSkinColor.g + (uniformHairColor.g - uniformSkinColor.g) * hs;
        const ub = uniformSkinColor.b + (uniformHairColor.b - uniformSkinColor.b) * hs;
        linear.setRGB(
          linear.r + (ur - linear.r) * invalidW,
          linear.g + (ug - linear.g) * invalidW,
          linear.b + (ub - linear.b) * invalidW,
          THREE.LinearSRGBColorSpace,
        );
      }
    }

    fallback[i * 3] = linear.r + (interiorColor.r - linear.r) * iw;
    fallback[i * 3 + 1] = linear.g + (interiorColor.g - linear.g) * iw;
    fallback[i * 3 + 2] = linear.b + (interiorColor.b - linear.b) * iw;

  }

  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geometry.setAttribute('aFallback', new THREE.BufferAttribute(fallback, 3));
  geometry.setAttribute('aPhotoW', new THREE.BufferAttribute(photoW, 1));
  applyFlatNormals(geometry); // ライティングは「写真の陰影のみ」の方針 (偽の影の帯を防ぐ)

  // 実測法線 (DAViD): ObjectSpaceNormalMapとしてhead/髪に貼り、回転時の
  // 照明応答を与える。頂点法線は+Z固定のまま — 法線マップが全面的に置き換える
  let normalTexture: THREE.Texture | null = null;
  const useNormal = params.normalSource === 'DAVID' && ctx.measured?.davidNormalCanvas;
  if (useNormal) {
    normalTexture = new THREE.CanvasTexture(
      blendNormalCanvasToFlat(ctx.measured!.davidNormalCanvas!, params.gnmNormalStrength),
    );
    normalTexture.colorSpace = THREE.NoColorSpace;
  }

  const headMaterial = new THREE.MeshStandardMaterial({
    map: headTexture,
    roughness: 0.95,
    metalness: 0.0,
  });
  if (normalTexture) {
    headMaterial.normalMap = normalTexture;
    headMaterial.normalMapType = THREE.ObjectSpaceNormalMap;
  }
  patchPhotoMixShader(headMaterial);

  const headMesh = new THREE.Mesh(geometry, headMaterial);

  // --- Hair (実測髪マスク+実測Depth) ---
  // 前面1枚グリッドのシェル: 写真の髪シルエットと実測Depthの起伏に忠実。
  // 頂点は画像平面の自由グリッドなので、頭蓋の外へ垂れる髪 (耳下・ロング) も張れる
  const hair = buildHairShell(ctx, texture, fit, model.triangles, params, normalTexture);
  if (hair) hair.mesh.visible = params.gnmShowHair;

  // --- 口腔内 (口腔壁・歯・歯茎・舌) ---
  // 頭部と同じ頂点配列を共有するが、写真投影を持たず実法線+ライティングで描くため別メッシュ
  // 色の基準は公式の色式と同じ「肌色」。減光していない平均色を測り直して渡す
  const mouthSkinColor = seg ? maskedAverageColor(sourceCanvas, seg.faceSkin, 1) : null;
  const mouthInterior = buildMouthInterior(model, fit.vertices, mouthSkinColor, lipPhotoColor);
  if (mouthInterior) mouthInterior.mesh.visible = params.gnmShowMouthInterior;

  // 回転pivotは頭部の実重心z (真3Dのため固定比率ではなく実測で決める)
  const pivotZ = fit.centerZ;
  headMesh.position.z = -pivotZ;
  if (hair) hair.mesh.position.z = -pivotZ;
  if (mouthInterior) mouthInterior.mesh.position.z = -pivotZ;

  const group = new THREE.Group();
  group.position.z = pivotZ;
  group.add(headMesh);
  if (mouthInterior) group.add(mouthInterior.mesh);
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
  // 瞼が眼球へ潜り込むのを毎フレーム禁じる拘束 (neutralの形は変えない)
  const eyeballContainment = buildEyeballContainment(model, neutralUntransformed);
  const exprCurrent = new Float32Array(model.expressionCount);
  const exprTarget = new Float32Array(model.expressionCount);
  const posAttr = geometry.getAttribute('position') as THREE.BufferAttribute;

  // まばたきベクトル: 公式ExpressionSamplerのWINK_LEFT+WINK_RIGHT合成 (目領域のみ)。
  const blinkVec =
    vectors.blink.length === model.expressionCount
      ? vectors.blink
      : new Array<number>(model.expressionCount).fill(0);
  let blinkNow = 0;

  // 目領域の成分 (left_eye* / right_eye*)。まばたきはこれらを加算ではなく
  // 置き換える — 加算だと開瞼系の感情 (Surprise等) と打ち消し合い、
  // まばたき中も瞼が閉じ切らず眼球が瞼を貫通して見える
  const isEyeExpr = new Uint8Array(model.expressionCount);
  for (let i = 0; i < model.expressionCount; i++) {
    if (/^(left|right)_eye/.test(model.expressionNames[i] ?? '')) isEyeExpr[i] = 1;
  }

  // 舌: 公式の姿勢 (定数) + 顎の開きへの追随 (公式に無い連動)。gnmMouthInterior参照
  const tongueDriver = buildTongueDriver(model);
  const coeffs = new Float32Array(model.expressionCount);
  // 表情適用後の頂点 (相似変換済み)。毎フレーム作り直さず使い回す
  // (17k頂点 = 1フレームあたり200KBの割当を避ける)。初期値は無表情の最終頂点そのもの —
  // 表情が静止したままオーバーレイをonにしたときもここから現在の姿勢が取れるように
  const applied = new Float32Array(fit.vertices);

  const applyExpressionNow = (): void => {
    for (let i = 0; i < model.expressionCount; i++) {
      // 目領域はblinkNowでクロスフェード (閉眼時はまばたきが支配)。
      // それ以外 (下顔面) は感情表情のまま
      coeffs[i] = isEyeExpr[i]
        ? exprCurrent[i] * (1 - blinkNow) + blinkVec[i] * blinkNow
        : exprCurrent[i] + blinkVec[i] * blinkNow;
    }
    tongueDriver?.apply(coeffs, params.gnmTonguePose);

    const out = applied;
    out.set(neutralUntransformed);
    for (let i = 0; i < model.expressionCount; i++) {
      const c = coeffs[i];
      if (c === 0) continue;
      // exprScales: 残差ワープで瞼開口幅が変わった分の目領域振幅補正
      const cs = (c * exprScales[i] * model.expressionScales[i]) / 32767;
      const base = i * model.vertexCount * 3;
      for (let j = 0; j < model.vertexCount * 3; j++) out[j] += model.expressionBasisQ[base + j] * cs;
    }
    eyeballContainment?.apply(out);
    applySimilarityInPlace(out, fit.sim);
    (posAttr.array as Float32Array).set(out);
    posAttr.needsUpdate = true;
    mouthInterior?.update(out);
    // 非表示のときは対応点の引き直しを丸ごと省く (既定でoffなので通常はここで抜ける)
    if (landmarkOverlay.object.visible) landmarkOverlay.update(out);
  };

  // --- レイヤー分離プレビュー: 元写真 → 最終出力の加工工程を順に並べる ---
  // GPUへ渡しているテクスチャはそのキャンバス自体を入れる (作り直した近似は入れない)
  const previewLayers: PreviewLayer[] = [{ label: '1 元写真', canvas: sourceCanvas, photoAspect: true }];
  if (seg) {
    previewLayers.push({
      label: '2 人物マスク (person)',
      canvas: buildMaskedPhotoLayer(sourceCanvas, (u, v) => sampleField(seg.person, u, v)),
      photoAspect: true,
    });
  }
  previewLayers.push({
    label: '3 head参照画素 (UVクランプ後)',
    canvas: buildSampledPixelsLayer(sourceCanvas, uvs, photoW),
    photoAspect: true,
  });
  if (seg) {
    previewLayers.push({
      label: `4 髪マスク (帽子:${params.gnmHairIncludeAccessories ? 'on' : 'off'} / GF:${params.gnmMaskRefine ? 'on' : 'off'})`,
      canvas: buildMaskedPhotoLayer(sourceCanvas, (u, v) => sampleField(seg.hair, u, v)),
      photoAspect: true,
    });
  }
  if (hair) {
    previewLayers.push({
      // 正方形1024pxへ引き伸ばしたものをGPUへ渡しているので、そのまま出す
      label: '5 髪alphaMap (GPU実物1024²)',
      canvas: hair.alphaTexture.image as HTMLCanvasElement,
      photoAspect: false,
    });
  }
  const depthCanvas = buildDepthPreviewCanvas(ctx, params);
  if (depthCanvas) {
    previewLayers.push({ label: '6 深度 (使用中)', canvas: depthCanvas, photoAspect: true });
  }
  if (normalTexture) {
    previewLayers.push({
      // Normal Strengthを適用した後の実物 (生のDAViD出力ではない)
      label: `7 法線 (強度${params.gnmNormalStrength.toFixed(2)}適用後)`,
      canvas: normalTexture.image as HTMLCanvasElement,
      photoAspect: true,
    });
  }

  return {
    group,
    headMesh,
    hairMesh: hair?.mesh ?? null,
    mouthInteriorMesh: mouthInterior?.mesh ?? null,
    landmarkOverlay: landmarkOverlay.object,
    refreshLandmarkOverlay() {
      landmarkOverlay.update(applied);
    },
    previewLayers,
    layerPreviewCrop: computeHeadCropFraction(ctx),
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
      normalTexture?.dispose();
      landmarkOverlay.dispose();
      mouthInterior?.dispose();
      if (hair) {
        hair.mesh.geometry.dispose();
        hair.alphaTexture.dispose();
        (hair.mesh.material as THREE.Material).dispose();
      }
    },
  };
}

/**
 * 写真をマスク解像度へ縮小し、マスク重み付き平均色を返す (linear空間)。
 * darkenはsRGB空間での減光率 (背面=陰の暗さ。口内色の0.4と同じ発想)。
 * マスクがほぼ空なら null。
 */
function maskedAverageColor(
  sourceCanvas: HTMLCanvasElement,
  field: ScalarField,
  darken: number,
): THREE.Color | null {
  const c = document.createElement('canvas');
  c.width = field.width;
  c.height = field.height;
  const cc = c.getContext('2d')!;
  cc.drawImage(sourceCanvas, 0, 0, field.width, field.height);
  const data = cc.getImageData(0, 0, field.width, field.height).data;
  let r = 0;
  let g = 0;
  let b = 0;
  let wSum = 0;
  for (let i = 0; i < field.data.length; i++) {
    const w = field.data[i];
    if (w <= 0.2) continue;
    r += data[i * 4] * w;
    g += data[i * 4 + 1] * w;
    b += data[i * 4 + 2] * w;
    wSum += w;
  }
  if (wSum < 1) return null;
  return new THREE.Color().setRGB(
    (r / wSum / 255) * darken,
    (g / wSum / 255) * darken,
    (b / wSum / 255) * darken,
    THREE.SRGBColorSpace,
  );
}

/** 頭部まわりのクロップ矩形を画像割合 (0-1) で求める (プレビュー表示用)。 */
function computeHeadCropFraction(ctx: GnmBuildContext): { x: number; y: number; w: number; h: number } {
  const cx = ctx.headCenterPx.x / ctx.imageWidth;
  const cy = ctx.headCenterPx.y / ctx.imageHeight;
  const fw = ctx.faceWidthPx / ctx.imageWidth;
  const fh = ctx.faceWidthPx / ctx.imageHeight;
  // 髪を含む頭部全体が入る余裕 (横±1.5faceWidth, 上2.0 / 下1.6faceWidth)
  const x0 = Math.max(0, cx - fw * 1.5);
  const x1 = Math.min(1, cx + fw * 1.5);
  const y0 = Math.max(0, cy - fh * 2.0);
  const y1 = Math.min(1, cy + fh * 1.6);
  return { x: x0, y: y0, w: Math.max(1e-6, x1 - x0), h: Math.max(1e-6, y1 - y0) };
}

const LAYER_PREVIEW_MAX_DIM = 512;

/** 元写真と同じアスペクト比の作業キャンバスを作る (長辺512上限)。 */
function previewCanvas(sourceCanvas: HTMLCanvasElement): HTMLCanvasElement {
  const scale = Math.min(1, LAYER_PREVIEW_MAX_DIM / Math.max(sourceCanvas.width, sourceCanvas.height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(2, Math.round(sourceCanvas.width * scale));
  canvas.height = Math.max(2, Math.round(sourceCanvas.height * scale));
  return canvas;
}

/**
 * 写真を alpha(u,v) でマスクした画像を作る。
 * 「この工程が写真のどこを使っているか」を1枚で見せるための表現。
 */
function buildMaskedPhotoLayer(
  sourceCanvas: HTMLCanvasElement,
  alpha: (u: number, v: number) => number,
): HTMLCanvasElement {
  const canvas = previewCanvas(sourceCanvas);
  const { width: w, height: h } = canvas;
  const c = canvas.getContext('2d')!;
  c.drawImage(sourceCanvas, 0, 0, w, h);
  const img = c.getImageData(0, 0, w, h);
  for (let y = 0; y < h; y++) {
    const v = 1 - (y + 0.5) / h;
    for (let x = 0; x < w; x++) {
      const a = alpha((x + 0.5) / w, v);
      img.data[(y * w + x) * 4 + 3] = Math.round(Math.min(1, Math.max(0, a)) * 255);
    }
  }
  c.putImageData(img, 0, 0);
  return canvas;
}

/**
 * headメッシュが実際に読んだ写真画素の分布。クランプ済みUVを photoW 重みで
 * 3x3スプラットして密度にする。
 * シルエット外から歩かされた頂点は縁の同じ画素へ集まるので、
 * 「引き伸ばし (スメア) が起きている場所」が明るい線として見える。
 */
function buildSampledPixelsLayer(
  sourceCanvas: HTMLCanvasElement,
  uvs: Float32Array,
  photoW: Float32Array,
): HTMLCanvasElement {
  const canvas = previewCanvas(sourceCanvas);
  const { width: w, height: h } = canvas;
  const acc = new Float32Array(w * h);
  for (let i = 0; i < photoW.length; i++) {
    const weight = photoW[i];
    if (weight <= 0.01) continue;
    const cx = Math.round(uvs[i * 2] * w - 0.5);
    const cy = Math.round((1 - uvs[i * 2 + 1]) * h - 0.5);
    for (let dy = -1; dy <= 1; dy++) {
      const y = cy + dy;
      if (y < 0 || y >= h) continue;
      for (let dx = -1; dx <= 1; dx++) {
        const x = cx + dx;
        if (x < 0 || x >= w) continue;
        acc[y * w + x] += weight;
      }
    }
  }
  const c = canvas.getContext('2d')!;
  c.drawImage(sourceCanvas, 0, 0, w, h);
  const img = c.getImageData(0, 0, w, h);
  for (let i = 0; i < acc.length; i++) {
    // 1頂点でもほぼ不透明、重なるほど飽和。密度の絶対値に依存しない形にする
    img.data[i * 4 + 3] = Math.round((1 - Math.exp(-acc[i] * 1.5)) * 255);
  }
  c.putImageData(img, 0, 0);
  return canvas;
}

/**
 * 法線canvasを平坦 (+Z) へ向けてstrengthで弱めたコピーを返す (1ならそのまま)。
 * RGB混合でベクトル長は縮むが、three.js側で正規化されるため問題ない。
 */
function blendNormalCanvasToFlat(src: HTMLCanvasElement, strength: number): HTMLCanvasElement {
  const s = Math.min(1, Math.max(0, strength));
  if (s >= 1) return src;
  const out = document.createElement('canvas');
  out.width = src.width;
  out.height = src.height;
  const c = out.getContext('2d')!;
  c.drawImage(src, 0, 0);
  c.globalAlpha = 1 - s;
  c.fillStyle = 'rgb(128,128,255)';
  c.fillRect(0, 0, out.width, out.height);
  return out;
}

const DEPTH_PREVIEW_MAX_DIM = 512;

/** 使用中のDepth場を画像全体空間のグレースケールへ可視化する (白=手前、rect外=黒)。 */
function buildDepthPreviewCanvas(ctx: GnmBuildContext, params: Params): HTMLCanvasElement | null {
  const depth = selectDepth(ctx, params);
  if (!depth) return null;
  const scale = Math.min(1, DEPTH_PREVIEW_MAX_DIM / Math.max(ctx.imageWidth, ctx.imageHeight));
  const w = Math.max(2, Math.round(ctx.imageWidth * scale));
  const h = Math.max(2, Math.round(ctx.imageHeight * scale));
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const c = canvas.getContext('2d')!;
  const img = c.createImageData(w, h);
  // rect内のmin-maxで正規化して表示する
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < depth.data.length; i++) {
    if (depth.data[i] < min) min = depth.data[i];
    if (depth.data[i] > max) max = depth.data[i];
  }
  const span = Math.max(1e-9, max - min);
  for (let y = 0; y < h; y++) {
    const v = 1 - (y + 0.5) / h;
    for (let x = 0; x < w; x++) {
      const u = (x + 0.5) / w;
      const { u0, v0, u1, v1 } = depth.rect;
      const inside = u >= u0 && u <= u1 && v >= v0 && v <= v1;
      const g = inside ? Math.round(((sampleField(depth, u, v) - min) / span) * 255) : 0;
      const i = (y * w + x) * 4;
      img.data[i] = g;
      img.data[i + 1] = g;
      img.data[i + 2] = g;
      img.data[i + 3] = 255;
    }
  }
  c.putImageData(img, 0, 0);
  return canvas;
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
 * ランドマーク重畳デバッグ表示を作る。3種を重ねる:
 * - 色点 (大)  = 写真のランドマーク位置 (残差ワープの目標。唇=赤 / 目=シアン / その他=黄)。
 *                写真から測った固定値なので表情では動かない
 * - 暗点 (小)  = 現在の表情を適用したGNM表面の対応点。表情に追従して動く
 * - 白線       = 暗点→色点 の残差ベクトル
 *
 * 表情アニメーション中は「表情が対応点をどこへ運んだか」と「写真の目標」の差が
 * そのまま線の長さになる。無表情で線が短く、開口時に唇の線が伸びるなら、
 * 伸びた分が表情基底では張れていない差分。
 * zは常に現在の表面zを使う (透視投影の視差で写真とXY比較が狂わないように)。
 */
interface LandmarkOverlayBuild {
  object: THREE.Object3D;
  /** 表情適用後の頂点配列 (相似変換済み) から対応点を引き直す。 */
  update(vertices: Float32Array): void;
  dispose(): void;
}

function buildLandmarkOverlay(
  model: GnmModel,
  fit: GnmFitResult,
  landmarks: NormalizedFaceLandmark[],
): LandmarkOverlayBuild {
  const useDense = model.denseCount > 0;
  const corrCount = useDense ? model.denseCount : MEDIAPIPE_IBUG68.length;
  const corrIdx = useDense ? model.denseTriIndices : model.landmarkIndices;
  const corrBary = useDense ? model.denseBaryWeights : model.landmarkWeights;
  const corrMp = (k: number) => (useDense ? model.denseMpIndices[k] : MEDIAPIPE_IBUG68[k]);

  // 対応点を持つものだけを詰め直す (毎フレーム参照するので事前に平坦化しておく)
  const tri: number[] = [];
  const bary: number[] = [];
  const photoXy: number[] = [];
  const photoColors: number[] = [];
  const meshColors: number[] = [];
  for (let k = 0; k < corrCount; k++) {
    const mp = corrMp(k);
    const lm = landmarks[mp];
    if (!lm) continue;
    for (let j = 0; j < 3; j++) {
      tri.push(corrIdx[k * 3 + j]);
      bary.push(corrBary[k * 3 + j]);
    }
    photoXy.push(lm.x, lm.y);
    const c = MP_LIPS.has(mp) ? [1, 0.15, 0.3] : MP_EYES.has(mp) ? [0.15, 0.9, 1] : [1, 0.85, 0.2];
    photoColors.push(...c);
    // GNM側は同じ色相を暗くして「目標と現在」が対で読めるようにする
    meshColors.push(c[0] * 0.45, c[1] * 0.45, c[2] * 0.45);
  }
  const count = photoXy.length / 2;
  const triIdx = Int32Array.from(tri);
  const baryW = Float32Array.from(bary);
  const photo = Float32Array.from(photoXy);

  const zLift = 0.01; // 表面と重ならないよう僅かに手前へ
  const photoPos = new Float32Array(count * 3);
  const meshPos = new Float32Array(count * 3);
  const linePos = new Float32Array(count * 6);

  const makePoints = (pos: Float32Array, colors: number[], size: number): THREE.Points => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    const m = new THREE.PointsMaterial({
      size,
      vertexColors: true,
      sizeAttenuation: true,
      depthTest: false,
      transparent: true,
    });
    return new THREE.Points(g, m);
  };
  const photoObj = makePoints(photoPos, photoColors, 0.012);
  photoObj.renderOrder = 11;
  const meshObj = makePoints(meshPos, meshColors, 0.008);
  meshObj.renderOrder = 10;

  const linesGeo = new THREE.BufferGeometry();
  linesGeo.setAttribute('position', new THREE.BufferAttribute(linePos, 3));
  const linesMat = new THREE.LineBasicMaterial({
    color: 0xffffff,
    depthTest: false,
    transparent: true,
    opacity: 0.7,
  });
  const linesObj = new THREE.LineSegments(linesGeo, linesMat);
  linesObj.renderOrder = 9;

  const object = new THREE.Group();
  object.add(linesObj, meshObj, photoObj);

  const update = (verts: Float32Array): void => {
    for (let k = 0; k < count; k++) {
      let sx = 0;
      let sy = 0;
      let sz = 0;
      for (let j = 0; j < 3; j++) {
        const vi = triIdx[k * 3 + j] * 3;
        const bw = baryW[k * 3 + j];
        sx += verts[vi] * bw;
        sy += verts[vi + 1] * bw;
        sz += verts[vi + 2] * bw;
      }
      const z = sz + zLift;
      const px = photo[k * 2];
      const py = photo[k * 2 + 1];
      meshPos[k * 3] = sx;
      meshPos[k * 3 + 1] = sy;
      meshPos[k * 3 + 2] = z;
      photoPos[k * 3] = px;
      photoPos[k * 3 + 1] = py;
      photoPos[k * 3 + 2] = z;
      linePos[k * 6] = sx;
      linePos[k * 6 + 1] = sy;
      linePos[k * 6 + 2] = z;
      linePos[k * 6 + 3] = px;
      linePos[k * 6 + 4] = py;
      linePos[k * 6 + 5] = z;
    }
    (photoObj.geometry.getAttribute('position') as THREE.BufferAttribute).needsUpdate = true;
    (meshObj.geometry.getAttribute('position') as THREE.BufferAttribute).needsUpdate = true;
    (linesGeo.getAttribute('position') as THREE.BufferAttribute).needsUpdate = true;
  };
  update(fit.vertices); // 無表情の初期姿勢

  return {
    object,
    update,
    dispose() {
      photoObj.geometry.dispose();
      (photoObj.material as THREE.Material).dispose();
      meshObj.geometry.dispose();
      (meshObj.material as THREE.Material).dispose();
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

// 髪の厚み範囲 (モデル空間, faceWidth≈1)。厚すぎるとピッチ回転時に
// シェル/キャップと頭皮の隙間が下から見える
const HAIR_MAX_THICKNESS = 0.16;
const HAIR_MIN_THICKNESS = 0.02;

/** 法線キャンバス (RGBエンコード) からモデル空間法線を読むサンプラを作る。 */
function makeNormalSampler(canvas: HTMLCanvasElement): (u: number, v: number) => [number, number, number] {
  const w = canvas.width;
  const h = canvas.height;
  const data = canvas.getContext('2d')!.getImageData(0, 0, w, h).data;
  return (u, v) => {
    const px = Math.min(w - 1, Math.max(0, Math.round(u * w - 0.5)));
    const py = Math.min(h - 1, Math.max(0, Math.round((1 - v) * h - 0.5)));
    const o = (py * w + px) * 4;
    return [data[o] / 127.5 - 1, data[o + 1] / 127.5 - 1, data[o + 2] / 127.5 - 1];
  };
}

/**
 * 実測法線でシェル表面の起伏 (毛束の凹凸) を作る。
 * 高さ場の法線は n ∝ (-∂z/∂x, -∂z/∂y, 1) なので ∂z/∂x = -nx/nz で勾配が決まる。
 * その勾配に合う高さ場をJacobi反復で解く。Depth由来のzを初期値+データ項に使い
 * 反復を有界で打ち切るため、低周波はDepthのまま・高周波だけ法線由来になる。
 *
 * Depthは絶対位置は取れるが毛束スケールの凹凸は潰れており、逆に法線は
 * 高周波に強く絶対位置を持たない — 役割を分けて融合する。
 */
function applyNormalRelief(
  positions: Float32Array,
  uvs: Float32Array,
  maskPerVertex: Float32Array,
  cols: number,
  rows: number,
  sampleNormal: (u: number, v: number) => [number, number, number],
  dx: number,
  dy: number,
  strength: number,
): void {
  const total = cols * rows;
  const gx = new Float32Array(total);
  const gy = new Float32Array(total);
  const trust = new Float32Array(total);
  for (let i = 0; i < total; i++) {
    const [nx, ny, nz] = sampleNormal(uvs[i * 2], uvs[i * 2 + 1]);
    // グレージング (nz小) は勾配が発散するので信頼しない。髪の外も使わない
    const t = smoothstep(0.35, 0.6, nz) * smoothstep(0.25, 0.55, maskPerVertex[i]);
    trust[i] = t;
    if (t <= 0) continue;
    const nzSafe = Math.max(0.35, nz);
    gx[i] = -nx / nzSafe;
    gy[i] = -ny / nzSafe;
  }

  const anchor = new Float32Array(total);
  for (let i = 0; i < total; i++) anchor[i] = positions[i * 3 + 2];
  const z = new Float32Array(anchor);
  const next = new Float32Array(total);
  const LAMBDA = 0.08; // データ項 (Depth由来zへの引き戻し) の重み
  const MAX_STEP = 0.05; // 1セルあたりの許容起伏 (モデル空間)。法線ノイズの暴走止め
  const ITERATIONS = 120;
  const clampStep = (v: number) => Math.min(MAX_STEP, Math.max(-MAX_STEP, v));

  for (let iter = 0; iter < ITERATIONS; iter++) {
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const i = row * cols + col;
        let sum = LAMBDA * anchor[i];
        let wsum = LAMBDA;
        // 隣接4方向。row+1はyが小さくなる向き (グリッドは上から下へ張る)
        if (col > 0) {
          const j = i - 1;
          const e = Math.min(trust[i], trust[j]) * strength;
          sum += z[j] + clampStep(0.5 * (gx[i] + gx[j]) * dx * e);
          wsum += 1;
        }
        if (col < cols - 1) {
          const j = i + 1;
          const e = Math.min(trust[i], trust[j]) * strength;
          sum += z[j] - clampStep(0.5 * (gx[i] + gx[j]) * dx * e);
          wsum += 1;
        }
        if (row > 0) {
          const j = i - cols;
          const e = Math.min(trust[i], trust[j]) * strength;
          sum += z[j] - clampStep(0.5 * (gy[i] + gy[j]) * dy * e);
          wsum += 1;
        }
        if (row < rows - 1) {
          const j = i + cols;
          const e = Math.min(trust[i], trust[j]) * strength;
          sum += z[j] + clampStep(0.5 * (gy[i] + gy[j]) * dy * e);
          wsum += 1;
        }
        next[i] = sum / wsum;
      }
    }
    z.set(next);
  }

  for (let i = 0; i < total; i++) positions[i * 3 + 2] = z[i];
}

/** グリッド上のスカラー場を3x3近傍平均で平滑化する (in-place)。 */
function smoothGridField(field: Float32Array, cols: number, rows: number, passes: number): void {
  if (passes <= 0) return;
  const src = new Float32Array(field.length);
  for (let pass = 0; pass < passes; pass++) {
    src.set(field);
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
        field[row * cols + col] = sum / count;
      }
    }
  }
}

/**
 * 実測髪マスク+実測Depthの前面髪シェルを作る。
 * Depthの相対値はランドマーク位置の「フィット済GNM表面z」への最小二乗で
 * モデル空間zへ写像する (実比率スケール)。
 *
 * 役割分担: 髪マスクは「色 (RGB) と輪郭 (alphaMap)」専用で、形状には
 * 低周波成分だけを使う。マスクの高周波 (房の切れ目・GFの滲み) をzへ通すと
 * 房境界ごとにメッシュが折れ、グリッド解像度のジグザグになる。
 * 形状は実測Depth (絶対位置) が担う。
 */
function buildHairShell(
  ctx: GnmBuildContext,
  texture: THREE.Texture,
  fit: GnmFitResult,
  triangles: Uint32Array,
  params: Params,
  normalTexture: THREE.Texture | null = null,
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
  const scalp = buildScalpZBuffer(fit.vertices, triangles, { xMin, xMax, yMin, yMax });

  // 髪マスク重み付きの深度サンプル。前髪などまばらな髪帯では、毛の隙間から
  // 見える肌 (奥) の深度が混ざって格子zがジグザグになる (高解像度のDAViDは
  // この毛/肌の段差を実際に解像する。低解像度のARPortraitDepthでは潰れて
  // 起きない)。シェルは「髪の表面」を張るものなので、近傍3x3を髪らしさで
  // 重み付けし、隙間画素の肌深度を捨てて毛側の深度だけを拾う
  const cellU = ((xMax - xMin) / (cols - 1)) * (ctx.faceWidthPx / ctx.imageWidth);
  const cellV = ((yMax - yMin) / (rows - 1)) * (ctx.faceWidthPx / ctx.imageHeight);
  const sampleHairDepth = (u: number, v: number, centerMask: number): number => {
    if (centerMask < 0.05) return sampleField(depth, u, v); // 髪の外は素通し
    let sum = 0;
    let wsum = 0;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const su = u + dx * cellU * 0.75;
        const sv = v + dy * cellV * 0.75;
        const w = smoothstep(0.2, 0.7, sampleField(seg.hair, su, sv));
        if (w <= 0) continue;
        sum += sampleField(depth, su, sv) * w;
        wsum += w;
      }
    }
    return wsum > 1e-3 ? sum / wsum : sampleField(depth, u, v);
  };

  const thicknessRaw = new Float32Array(cols * rows); // クランプ前の実測髪厚
  const scalpZs = new Float32Array(cols * rows);
  for (let row = 0; row < rows; row++) {
    const y = yMax + (yMin - yMax) * (row / (rows - 1));
    for (let col = 0; col < cols; col++) {
      const x = xMin + (xMax - xMin) * (col / (cols - 1));
      const idx = row * cols + col;
      const u = (x * ctx.faceWidthPx + ctx.headCenterPx.x) / ctx.imageWidth;
      const v = 1 - (ctx.headCenterPx.y - y * ctx.faceWidthPx) / ctx.imageHeight;

      const hairMask = sampleField(seg.hair, u, v);
      maskPerVertex[idx] = hairMask;
      const d = sampleHairDepth(u, v, hairMask);
      const zMeasured = (d * hairFit.scale + hairFit.offset) * params.measuredDepthGain;
      scalpZs[idx] = scalp(x, y);
      thicknessRaw[idx] = zMeasured - scalpZs[idx];

      positions[idx * 3] = x;
      positions[idx * 3 + 1] = y;
      uvs[idx * 2] = u;
      uvs[idx * 2 + 1] = v;
    }
  }

  // 厚み場のノイズ (GNM実スケールで増幅される) をグリッド空間で平滑化する。
  // 3x3 blurの物理半径はセルサイズに比例して縮むため、パス数はグリッド密度の
  // 2乗でスケールして「見た目の平滑量」を解像度から独立させる。
  // DAViDはノイズが少ないため基準を弱め (旧64列グリッドで1パス相当) にして
  // 生え際・毛束の実起伏を残す
  const basePasses = params.depthSource === 'DAVID' && ctx.measured?.davidDepth ? 1 : 2;
  const densityScale = Math.max(1, Math.round((cols / 64) ** 2));
  smoothGridField(thicknessRaw, cols, rows, basePasses * densityScale);

  // 縁の巻き込みに使うマスクは強く平滑化して低周波成分だけにする。
  // 生のマスクを使うと房の切れ目ごとにzが (厚み+rolloff) 幅で上下し、
  // それがメッシュのジグザグの主因になる (輪郭の精度はalphaMapが担う)
  const edgeField = new Float32Array(maskPerVertex);
  smoothGridField(edgeField, cols, rows, 3 * densityScale);

  for (let i = 0; i < cols * rows; i++) {
    const edge = smoothstep(0.08, 0.5, edgeField[i]);
    const thickness = Math.min(HAIR_MAX_THICKNESS, Math.max(HAIR_MIN_THICKNESS, thicknessRaw[i]));
    positions[i * 3 + 2] =
      scalpZs[i] + params.gnmHairLift + thickness * edge - params.gnmHairRolloff * (1 - edge);
  }

  // 毛束スケールの起伏を実測法線から作る (Depthは厚みの低周波しか持たない)
  const normalCanvas = params.normalSource === 'DAVID' ? ctx.measured?.davidNormalCanvas : null;
  if (normalCanvas && params.gnmHairRelief > 0) {
    applyNormalRelief(
      positions,
      uvs,
      maskPerVertex,
      cols,
      rows,
      makeNormalSampler(normalCanvas),
      (xMax - xMin) / (cols - 1),
      (yMax - yMin) / (rows - 1),
      params.gnmHairRelief,
    );
  }

  // 三角形は「1つでも髪に触れていれば」張る。全コーナーがマスク内という条件だと
  // 境界がグリッド解像度の階段になり、マスクが薄い房の内部にも穴が空く
  // (実測: 96x120で28%の三角形がこれで落ちていた)。
  // 実際の輪郭はalphaMap (1024px) のper-pixelカットが担うので、
  // ジオメトリはマスクより1セル外まで張っておくのが正しい
  const gridIndices = buildGridIndices(cols, rows);
  const kept: number[] = [];
  for (let t = 0; t < gridIndices.length; t += 3) {
    const m = Math.max(
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

  // 精細化済みマスク (768px) の解像度を活かすため1024/blur1で焼く
  const alphaTexture = new THREE.CanvasTexture(rasterizeMaskCanvas(seg.hair, 1024, 1));
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
  if (normalTexture) {
    material.normalMap = normalTexture;
    material.normalMapType = THREE.ObjectSpaceNormalMap;
  }

  return { mesh: new THREE.Mesh(geometry, material), alphaTexture };
}

const SCALP_BINS_X = 96;
const SCALP_BINS_Y = 112;

/**
 * フィット済GNMの表面を平行投影でラスタライズし、各ビンの最前面z (最大z) を持つ
 * 「頭皮zバッファ」を作る。空ビン (シルエット外) はBFSで最寄りの値を伝播するため、
 * 髪がGNMシルエットの外へはみ出す画素でも連続したzが返る。
 *
 * 頂点splatではなく三角形ラスタライズで埋める: GNM頂点はビン格子より疎で、
 * splatだと内部にも空ビンが散り、BFS伝播の階段がそのまま髪シェルの段差
 * (クシャクシャ) として現れる。
 */
function buildScalpZBuffer(
  verts: Float32Array,
  triangles: Uint32Array,
  bounds: { xMin: number; xMax: number; yMin: number; yMax: number },
): (x: number, y: number) => number {
  const { xMin, xMax, yMin, yMax } = bounds;
  const w = SCALP_BINS_X;
  const h = SCALP_BINS_Y;
  const data = new Float32Array(w * h).fill(-Infinity);

  const spanX = Math.max(1e-6, xMax - xMin);
  const spanY = Math.max(1e-6, yMax - yMin);
  const toBx = (x: number) => ((x - xMin) / spanX) * w - 0.5;
  const toBy = (y: number) => ((y - yMin) / spanY) * h - 0.5;
  for (let t = 0; t < triangles.length; t += 3) {
    const a = triangles[t] * 3;
    const b = triangles[t + 1] * 3;
    const c = triangles[t + 2] * 3;
    const ax = toBx(verts[a]);
    const ay = toBy(verts[a + 1]);
    const bx = toBx(verts[b]);
    const by = toBy(verts[b + 1]);
    const cx = toBx(verts[c]);
    const cy = toBy(verts[c + 1]);
    const area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    if (Math.abs(area) < 1e-9) continue;
    const x0 = Math.max(0, Math.floor(Math.min(ax, bx, cx)));
    const x1 = Math.min(w - 1, Math.ceil(Math.max(ax, bx, cx)));
    const y0 = Math.max(0, Math.floor(Math.min(ay, by, cy)));
    const y1 = Math.min(h - 1, Math.ceil(Math.max(ay, by, cy)));
    for (let py = y0; py <= y1; py++) {
      for (let px = x0; px <= x1; px++) {
        // 重心座標 (符号付き面積比)。三角形の向きに関係なく内外判定できるよう
        // areaで正規化してから0-1範囲を見る
        const wa = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / area;
        const wb = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / area;
        const wc = 1 - wa - wb;
        if (wa < -1e-4 || wb < -1e-4 || wc < -1e-4) continue;
        const z = wa * verts[a + 2] + wb * verts[b + 2] + wc * verts[c + 2];
        const idx = py * w + px;
        if (z > data[idx]) data[idx] = z;
      }
    }
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

  // ビンは「疎なGNM頂点のmax」なのでビン単位のノイズを持つ。3x3 blurで均す
  // (thicknessクランプ時にzがこのバッファへ直接従うため、ここのノイズは
  // そのままメッシュのクシャクシャになる)
  for (let pass = 0; pass < 2; pass++) {
    const src = data.slice();
    for (let by = 0; by < h; by++) {
      for (let bx = 0; bx < w; bx++) {
        let sum = 0;
        let cnt = 0;
        for (let dy = -1; dy <= 1; dy++) {
          const yy = by + dy;
          if (yy < 0 || yy >= h) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const xx = bx + dx;
            if (xx < 0 || xx >= w) continue;
            sum += src[yy * w + xx];
            cnt++;
          }
        }
        data[by * w + bx] = sum / cnt;
      }
    }
  }

  // バイリニア補間で参照する — 最近傍参照だとビン境界の階段が
  // グリッド解像度と干渉し、クランプ帯のメッシュが列ごとに跳ねる
  return (x: number, y: number) => {
    const fx = Math.min(w - 1.001, Math.max(0, ((x - xMin) / spanX) * w - 0.5));
    const fy = Math.min(h - 1.001, Math.max(0, ((y - yMin) / spanY) * h - 0.5));
    const bx = Math.floor(fx);
    const by = Math.floor(fy);
    const ax = fx - bx;
    const ay = fy - by;
    const i00 = data[by * w + bx];
    const i10 = data[by * w + bx + 1];
    const i01 = data[(by + 1) * w + bx];
    const i11 = data[(by + 1) * w + bx + 1];
    return (i00 * (1 - ax) + i10 * ax) * (1 - ay) + (i01 * (1 - ax) + i11 * ax) * ay;
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
