// Unity向けエクスポート (本番構成)。
//
// 本番の役割分担 (docs/unity_integration.md が利用側の指示書):
// - **テンプレート (1回だけ)**: GNM Headの「型」をUnityへ常駐させる。
//   平均形状 + 表情44成分のblendshape (未変換GNM空間・振幅補正なし) +
//   口腔内テンプレート + 表情サンプラー重み。ゲストに依存しない。
// - **ゲスト (毎回)**: ゲスト固有データだけを送る。
//   neutral頂点 (未変換空間・ワープ焼き込み済み) + UV + COLOR_0(fallback+photoW) +
//   写真テクスチャ + 髪シェル。フィットの相似変換はGLBノードのTransformとして持たせ、
//   Unityはテンプレートメッシュへ頂点・UV・頂点色を差し替えて同じTransformを掛ける。
//
// この分離が成立する理由: ランタイムの頂点式は
//   final = sim( neutralU + Σ cᵢ·exprScalesᵢ·basisΔᵢ )
// で、simは線形 (等方スケールs + Z回転 + 平行移動)。Unityでは
//   メッシュのローカル頂点 = neutralU + Σ weightᵢ·basisΔᵢ (テンプレートblendshape)
//   GameObjectのTransform = sim / weightᵢ = cᵢ·exprScalesᵢ
// とすれば厳密に一致する。exprScales (残差ワープ由来の目領域振幅補正) は
// ゲスト固有なのでmeta.jsonで渡し、Unity側でweightに乗算する。
//
// 頭部の写真投影シェーダ (photoW mix) はGLB標準に無いため、
// COLOR_0 = (fallback.rgb, photoW) に詰めてUnity側 (URP Lit Shader Graph) で再現する。
// 眼球非貫通拘束 (buildEyeballContainment) は毎フレームの非線形処理で
// blendshapeに畳み込めないため含めない (既知差分として指示書に明記)。

import * as THREE from 'three';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';
import { strToU8, zipSync, type Zippable } from 'three/examples/jsm/libs/fflate.module.js';
import { applyIdentity, type GnmModel } from './gnmHead';
import { EXPR_FOLLOW_RATE, type GnmBuildContext, type GnmHeadBuild } from './gnmHeadMesh';
import {
  OFFICIAL_COLOR_MODIFIERS,
  OFFICIAL_TONGUE_POSE,
  SKIN_MODIFIER,
  TONGUE_COEFF_LIMIT,
  compactInteriorVertices,
} from './gnmMouthInterior';
import type { Params } from './params';

/** Emotion=AUTO/RANDOM巡回のタイミング定数 (正本はmain.tsのGNM_AUTO_CYCLE)。 */
export interface AutoCycleTiming {
  neutralMinMs: number;
  neutralRandMs: number;
  holdMinMs: number;
  holdRandMs: number;
}

const EXPORT_TEXTURE_MAX_DIM = 2048; // GLBへ埋めるテクスチャの長辺上限
const SAMPLER_BIN_URL = 'gnm/gnm_expression_decoder.bin';

export const TEMPLATE_FORMAT_VERSION = 2; // v2: joints+LBS (glTF skin) を追加
export const GUEST_FORMAT_VERSION = 2; // v2: meta.jointsにゲスト固有のbind位置を追加

// ---------------------------------------------------------------------------
// テンプレート (全ゲスト共通・1回だけ)
// ---------------------------------------------------------------------------

export interface UnityTemplateInputs {
  model: GnmModel;
  /** まばたきベクトル (モデル成分順。main.tsのbuildBlinkVectorが正本。ゲスト非依存)。 */
  blinkVector: number[];
  params: Params;
  autoCycle: AutoCycleTiming;
}

/**
 * GNM Headテンプレート一式を template.glb + template_meta.json +
 * gnm_expression_decoder.bin のzipにまとめる。
 */
export async function exportUnityTemplate(inputs: UnityTemplateInputs): Promise<Blob> {
  const { model } = inputs;
  const n = model.vertexCount;

  // blendshape delta (未変換GNM空間・振幅1)。exprScales/相似はゲスト側で掛ける
  const morphDeltas: Float32Array[] = [];
  for (let i = 0; i < model.expressionCount; i++) {
    const factor = model.expressionScales[i] / 32767;
    const base = i * n * 3;
    const out = new Float32Array(n * 3);
    for (let j = 0; j < n * 3; j++) out[j] = model.expressionBasisQ[base + j] * factor;
    morphDeltas.push(out);
  }

  // --- HeadTemplate (平均形状。頂点・UV・頂点色はゲストロード時に差し替えられる) ---
  const headGeometry = new THREE.BufferGeometry();
  headGeometry.setAttribute('position', new THREE.BufferAttribute(model.positions.slice(), 3));
  headGeometry.setAttribute('normal', flatNormalAttribute(n));
  headGeometry.setIndex(new THREE.BufferAttribute(model.triangles, 1));
  headGeometry.morphTargetsRelative = true;
  headGeometry.morphAttributes.position = morphDeltas.map((delta, i) =>
    namedAttribute(delta, 3, model.expressionNames[i] ?? `expr_${i}`),
  );
  // 公式LBS (neck/head/left_eye/right_eye) をglTF skinとして持たせる。
  // J=4なのでglTFの「頂点あたり4関節」に全関節がそのまま収まる
  const hasSkin = model.jointNames.length > 0 && model.skinWeights.length === n * model.jointNames.length;
  if (hasSkin) {
    headGeometry.setAttribute('skinIndex', sequentialSkinIndexAttribute(n, model.jointNames.length));
    headGeometry.setAttribute('skinWeight', new THREE.BufferAttribute(model.skinWeights.slice(), 4));
  }
  const headSkinned = hasSkin
    ? new THREE.SkinnedMesh(headGeometry, new THREE.MeshStandardMaterial({ roughness: 0.95, metalness: 0 }))
    : null;
  const headMesh =
    headSkinned ?? new THREE.Mesh(headGeometry, new THREE.MeshStandardMaterial({ roughness: 0.95, metalness: 0 }));
  headMesh.name = 'HeadTemplate';
  headMesh.updateMorphTargets(); // attribute名からmorphTargetDictionaryを作る (=GLBのtargetNames)

  // --- MouthInteriorTemplate (詰め直し。頂点順はcompactInteriorVerticesが正本) ---
  const compacted = compactInteriorVertices(model);
  let mouthMesh: THREE.Mesh | null = null;
  let mouthSkinned: THREE.SkinnedMesh | null = null;
  if (compacted) {
    const { vertexMap: map, index } = compacted;
    const m = map.length;
    const slice3 = (src: Float32Array): Float32Array => {
      const out = new Float32Array(m * 3);
      for (let i = 0; i < m; i++) {
        const g = map[i] * 3;
        out[i * 3] = src[g];
        out[i * 3 + 1] = src[g + 1];
        out[i * 3 + 2] = src[g + 2];
      }
      return out;
    };
    const basePos = slice3(model.positions);
    const baseNormals = computeAreaWeightedNormals(basePos, index);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(basePos, 3));
    geometry.setAttribute('normal', new THREE.BufferAttribute(baseNormals, 3));
    geometry.setIndex(new THREE.BufferAttribute(index, 1));
    geometry.morphTargetsRelative = true;
    const posTargets: THREE.BufferAttribute[] = [];
    const normalTargets: THREE.BufferAttribute[] = [];
    const posed = new Float32Array(m * 3);
    for (let ci = 0; ci < morphDeltas.length; ci++) {
      const delta = slice3(morphDeltas[ci]);
      for (let j = 0; j < m * 3; j++) posed[j] = basePos[j] + delta[j];
      // 顎の開閉で面の向きが変わる分の法線morph (平均形状基準の近似。
      // ゲストのneutralは残差ワープで少し違うが、変形は目唇局所の数pxで影響は小さい)
      const posedNormals = computeAreaWeightedNormals(posed, index);
      for (let j = 0; j < m * 3; j++) posedNormals[j] -= baseNormals[j];
      const name = model.expressionNames[ci] ?? `expr_${ci}`;
      posTargets.push(namedAttribute(delta, 3, name));
      normalTargets.push(namedAttribute(posedNormals, 3, name));
    }
    geometry.morphAttributes.position = posTargets;
    geometry.morphAttributes.normal = normalTargets;
    if (hasSkin) {
      const j = model.jointNames.length;
      const weights = new Float32Array(m * j);
      for (let i = 0; i < m; i++) {
        for (let k = 0; k < j; k++) weights[i * j + k] = model.skinWeights[map[i] * j + k];
      }
      geometry.setAttribute('skinIndex', sequentialSkinIndexAttribute(m, j));
      geometry.setAttribute('skinWeight', new THREE.BufferAttribute(weights, 4));
    }
    // 公式gnm_pyrenderと同じ MetallicRoughness(metallic=0, roughness=1)
    const material = new THREE.MeshStandardMaterial({ roughness: 1, metalness: 0 });
    mouthSkinned = hasSkin ? new THREE.SkinnedMesh(geometry, material) : null;
    mouthMesh = mouthSkinned ?? new THREE.Mesh(geometry, material);
    mouthMesh.name = 'MouthInteriorTemplate';
    mouthMesh.updateMorphTargets();
  }

  const root = new THREE.Group();
  root.name = 'GnmTemplate';
  root.add(headMesh);
  if (mouthMesh) root.add(mouthMesh);

  // 骨格 (テンプレートのbind位置)。ゲストロード時にUnity側でbind位置を差し替える
  if (headSkinned) {
    const bones = buildTemplateBones(model);
    const rootBone = bones[Math.max(0, model.jointParentIndices.findIndex((p) => p < 0))];
    root.add(rootBone);
    root.updateMatrixWorld(true);
    const skeleton = new THREE.Skeleton(bones); // ここでinverse bind行列が確定する
    headSkinned.bind(skeleton, headSkinned.matrixWorld);
    mouthSkinned?.bind(skeleton, mouthSkinned.matrixWorld);
  }

  const glb = (await new GLTFExporter().parseAsync(root, { binary: true })) as ArrayBuffer;
  const meta = buildTemplateMeta(inputs, compacted);

  const files: Zippable = {
    'template.glb': new Uint8Array(glb),
    'template_meta.json': strToU8(JSON.stringify(meta, null, 2)),
  };
  // 表情サンプラー重みも同梱する (Unity側の手動配置を1つ減らす)
  const samplerRes = await fetch(SAMPLER_BIN_URL);
  if (samplerRes.ok) {
    files['gnm_expression_decoder.bin'] = new Uint8Array(await samplerRes.arrayBuffer());
  }
  return zipBlob(files);
}

function buildTemplateMeta(
  inputs: UnityTemplateInputs,
  compacted: { vertexMap: Uint32Array; index: Uint32Array } | null,
): Record<string, unknown> {
  const { model, blinkVector, params, autoCycle } = inputs;
  const names = Array.from(
    { length: model.expressionCount },
    (_, i) => model.expressionNames[i] ?? `expr_${i}`,
  );
  return {
    formatVersion: TEMPLATE_FORMAT_VERSION,
    generator: 'gnm-head-avatar unityExport (template)',
    createdAt: new Date().toISOString(),
    nodes: {
      head: 'HeadTemplate',
      mouthInterior: compacted ? 'MouthInteriorTemplate' : null,
    },
    expression: {
      // blendshapeの並びと同一 (GLBのtargetNamesにも同じ名前が入る)
      names,
      // 目領域成分 (blinkはこのフラグの成分をクロスフェードで「置き換える」)
      eyeFlags: names.map((nm) => (/^(left|right)_eye/.test(nm) ? 1 : 0)),
      // まばたきベクトル (wink_left+wink_right合成の目領域のみ。係数=weightスケール)
      blinkVector,
      followRate: EXPR_FOLLOW_RATE,
      autoCycle,
    },
    tongue: { pose: OFFICIAL_TONGUE_POSE, coeffLimit: TONGUE_COEFF_LIMIT },
    blink: {
      periodMinSec: params.blinkPeriodMinSec,
      periodMaxSec: params.blinkPeriodMaxSec,
      durationMinMs: params.blinkDurationMinMs,
      durationMaxMs: params.blinkDurationMaxMs,
    },
    // 骨格の正本はtemplate.glbのskin (joint名・親子・テンプレbind位置)。
    // ゲストごとのbind位置はguest metaのjoints.bindPositionsで差し替える
    skinned: inputs.model.jointNames.length > 0,
    mouthInterior: compacted
      ? {
          // ローカル頂点index → 頭部頂点index。ゲストのneutral頂点から口腔内頂点を切り出す
          vertexMap: Array.from(compacted.vertexMap),
          // 頂点ごとのパーツID (0=肌(唇内縁) 1=口腔壁 2=歯 3=歯茎 4=舌)
          partId: Array.from(compacted.vertexMap, (g) => model.mouthPartId[g] ?? 0),
          // 頂点色 = sRGB空間で clamp01(肌色·scale + offset)。基準の肌色はゲストmetaで渡す
          colorModifiers: {
            default: SKIN_MODIFIER,
            byPartId: OFFICIAL_COLOR_MODIFIERS,
          },
        }
      : null,
    samplerFile: 'gnm_expression_decoder.bin',
  };
}

// ---------------------------------------------------------------------------
// ゲスト (毎回)
// ---------------------------------------------------------------------------

export interface UnityGuestInputs {
  model: GnmModel;
  build: GnmHeadBuild;
  ctx: GnmBuildContext;
  sourceCanvas: HTMLCanvasElement;
  params: Params;
}

/** ゲスト固有データを guest.glb + meta.json のzipにまとめる。 */
export async function exportUnityGuest(inputs: UnityGuestInputs): Promise<Blob> {
  const { model, build, ctx, sourceCanvas, params } = inputs;
  const n = model.vertexCount;
  const headSrc = build.headMesh.geometry;

  // --- Head: 未変換空間のneutral頂点 + UV + COLOR_0。相似変換はノードTransformに持たせる ---
  const headGeometry = new THREE.BufferGeometry();
  headGeometry.setAttribute('position', new THREE.BufferAttribute(build.neutralUntransformed.slice(), 3));
  headGeometry.setAttribute('uv', headSrc.getAttribute('uv').clone());
  headGeometry.setAttribute('normal', flatNormalAttribute(n));
  headGeometry.setAttribute('color', buildHeadColorAttribute(headSrc, n));
  headGeometry.setIndex(new THREE.BufferAttribute(model.triangles, 1));
  const photoTexture = canvasTexture(downscaleCanvas(sourceCanvas, EXPORT_TEXTURE_MAX_DIM));
  const headMesh = new THREE.Mesh(
    headGeometry,
    new THREE.MeshStandardMaterial({ map: photoTexture, roughness: 0.95, metalness: 0 }),
  );
  headMesh.name = 'Head';

  // フィットの相似変換 sim(v) = s·Rz(θ)·v + t をノードTRSで表現し、
  // 回転pivot分の平行移動 (0,0,-pivotZ) を合成する
  const pivotZ = build.fit.centerZ;
  const { s, cos, sin, tx, ty, tz } = build.fit.sim;
  const theta = Math.atan2(sin, cos);
  headMesh.position.set(tx, ty, tz - pivotZ);
  headMesh.quaternion.setFromAxisAngle(new THREE.Vector3(0, 0, 1), theta);
  headMesh.scale.setScalar(s);

  // --- HairShell: 最終モデル空間の静的メッシュ (相似は焼き込み済みなのでTransformなし) ---
  const hairMesh = buildHairExportMesh(build, sourceCanvas);
  if (hairMesh) hairMesh.position.z = -pivotZ;

  const root = new THREE.Group();
  root.name = 'HeadRoot'; // yaw/pitch回転はこのnodeに掛ける (実重心pivot)
  root.position.z = pivotZ;
  root.add(headMesh);
  if (hairMesh) root.add(hairMesh);

  const glb = (await new GLTFExporter().parseAsync(root, { binary: true })) as ArrayBuffer;
  const meta = buildGuestMeta(inputs, { hair: !!hairMesh });

  return zipBlob({
    'guest.glb': new Uint8Array(glb),
    'meta.json': strToU8(JSON.stringify(meta, null, 2)),
  });
}

function buildGuestMeta(inputs: UnityGuestInputs, nodes: { hair: boolean }): Record<string, unknown> {
  const { model, build, ctx, params } = inputs;
  const guestJoints = computeGuestJointPositions(model, build);
  return {
    formatVersion: GUEST_FORMAT_VERSION,
    generator: 'gnm-head-avatar unityExport (guest)',
    createdAt: new Date().toISOString(),
    image: { width: ctx.imageWidth, height: ctx.imageHeight },
    nodes: { head: 'Head', hair: nodes.hair ? 'HairShell' : null },
    expression: {
      // 残差ワープ由来の成分別振幅スケール (目領域の開口幅補正)。
      // weight = 係数 × exprScales[i] としてUnity側で乗算する
      exprScales: Array.from(build.exprScales.subarray(0, model.expressionCount)),
      intensity: params.gnmExprIntensity,
    },
    tongue: { poseAmount: params.gnmTonguePose },
    // ゲスト固有のjoint bind位置 (Headメッシュと同じ未変換空間)。
    // Unity側はボーンをこの位置へ動かし、Mesh.bindposesを計算し直す
    joints: guestJoints
      ? {
          bindPositions: Object.fromEntries(
            model.jointNames.map((name, j) => [
              name,
              [guestJoints[j * 3], guestJoints[j * 3 + 1], guestJoints[j * 3 + 2]],
            ]),
          ),
        }
      : null,
    // 口腔内の頂点色の基準色 (linear空間)。テンプレmetaのcolorModifiersと組で使う
    colors: {
      skinLinear: build.mouthSkinColorLinear,
      lipFallbackLinear: build.lipFallbackColorLinear,
    },
    blink: { enabled: params.blinkEnabled },
    view: {
      maxYawDeg: params.maxYawDeg,
      maxPitchDeg: params.maxPitchDeg,
      cameraFovDeg: params.cameraFovDeg,
      cameraDistance: params.cameraDistanceRatio,
      backgroundColor: params.backgroundColor,
    },
  };
}

// ---------------------------------------------------------------------------
// 共通ヘルパー
// ---------------------------------------------------------------------------

/** COLOR_0 = (fallback.rgb linear, photoW)。Unity側の写真mixシェーダの入力。 */
function buildHeadColorAttribute(headSrc: THREE.BufferGeometry, n: number): THREE.BufferAttribute {
  const fallback = headSrc.getAttribute('aFallback').array as Float32Array;
  const photoW = headSrc.getAttribute('aPhotoW').array as Float32Array;
  const color = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    color[i * 4] = fallback[i * 3];
    color[i * 4 + 1] = fallback[i * 3 + 1];
    color[i * 4 + 2] = fallback[i * 3 + 2];
    color[i * 4 + 3] = photoW[i];
  }
  return new THREE.BufferAttribute(color, 4);
}

/** 髪シェル (静的メッシュ)。写真RGB×髪マスクalphaを1枚のRGBAテクスチャへ事前合成する。 */
function buildHairExportMesh(build: GnmHeadBuild, sourceCanvas: HTMLCanvasElement): THREE.Mesh | null {
  const hair = build.hairMesh;
  if (!hair) return null;
  const src = hair.geometry;
  const alphaCanvas = (hair.material as THREE.MeshStandardMaterial).alphaMap?.image as
    | HTMLCanvasElement
    | undefined;
  if (!alphaCanvas) return null;

  const composite = downscaleCanvas(sourceCanvas, EXPORT_TEXTURE_MAX_DIM);
  const cctx = composite.getContext('2d')!;
  const img = cctx.getImageData(0, 0, composite.width, composite.height);
  // 髪マスク(正方形キャンバス)を写真サイズへ引き伸ばして読み、alphaチャンネルへ書く。
  // どちらも同じ画像UV空間なのでテクセル比が違っても対応は崩れない
  const maskResized = document.createElement('canvas');
  maskResized.width = composite.width;
  maskResized.height = composite.height;
  const mctx = maskResized.getContext('2d')!;
  mctx.drawImage(alphaCanvas, 0, 0, composite.width, composite.height);
  const mask = mctx.getImageData(0, 0, composite.width, composite.height).data;
  for (let i = 0; i < img.data.length / 4; i++) {
    img.data[i * 4 + 3] = mask[i * 4]; // グレースケールなのでRだけ読めばよい
  }
  cctx.putImageData(img, 0, 0);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', src.getAttribute('position').clone());
  geometry.setAttribute('uv', src.getAttribute('uv').clone());
  geometry.setAttribute('normal', src.getAttribute('normal').clone()); // 平坦(+Z)
  geometry.setIndex(src.getIndex()!.clone());

  const mesh = new THREE.Mesh(
    geometry,
    new THREE.MeshStandardMaterial({
      map: canvasTexture(composite),
      transparent: true, // GLBのalphaMode=BLENDになる (Web側はBLEND+alphaTest 0.3の併用)
      roughness: 0.95,
      metalness: 0,
      side: THREE.DoubleSide,
    }),
  );
  mesh.name = 'HairShell';
  return mesh;
}

/** JOINTS_0: 関節数J≤4なので全頂点 (0,1,2,3,…) の固定並びでよい (余りは0/重み0)。 */
function sequentialSkinIndexAttribute(count: number, jointCount: number): THREE.BufferAttribute {
  const idx = new Uint8Array(count * 4);
  for (let i = 0; i < count; i++) {
    for (let k = 0; k < Math.min(4, jointCount); k++) idx[i * 4 + k] = k;
  }
  return new THREE.BufferAttribute(idx, 4);
}

/** テンプレートのbind位置でボーン階層を作る (親からの相対位置)。 */
function buildTemplateBones(model: GnmModel): THREE.Bone[] {
  const bones = model.jointNames.map((name) => {
    const b = new THREE.Bone();
    b.name = name;
    return b;
  });
  const pos = model.templateJointPositions;
  for (let j = 0; j < bones.length; j++) {
    const p = model.jointParentIndices[j];
    if (p >= 0) {
      bones[p].add(bones[j]);
      bones[j].position.set(
        pos[j * 3] - pos[p * 3],
        pos[j * 3 + 1] - pos[p * 3 + 1],
        pos[j * 3 + 2] - pos[p * 3 + 2],
      );
    } else {
      bones[j].position.set(pos[j * 3], pos[j * 3 + 1], pos[j * 3 + 2]);
    }
  }
  return bones;
}

/**
 * ゲスト固有のjoint bind位置 (未変換空間)。
 * 公式 joint_positions_bind_pose (template + identity基底·係数) をそのまま計算し、
 * 眼球jointだけは「identity適用形状 → 実neutral (残差ワープ等の焼き込み後)」の
 * 眼球頂点平均変位を加える — 視線回転のpivotをワープ後の実際の眼球中心に合わせるため
 * (公式に無いこちら側の補正。ワープ量は数px・XYのみなので一次近似として整合する)。
 */
function computeGuestJointPositions(model: GnmModel, build: GnmHeadBuild): Float32Array | null {
  const j = model.jointNames.length;
  if (j === 0 || model.templateJointPositions.length !== j * 3) return null;
  const joints = new Float32Array(model.templateJointPositions);
  const coeffs = build.fit.coeffs;
  for (let k = 0; k < model.basisCount; k++) {
    const c = coeffs[k];
    if (c === 0) continue;
    const base = k * j * 3;
    for (let v = 0; v < j * 3; v++) joints[v] += model.jointIdentityBasis[base + v] * c;
  }

  const hasEye = model.jointNames.some((nm) => nm.includes('eye'));
  if (hasEye && model.eyeWeight.length === model.vertexCount) {
    const identityOnly = applyIdentity(model, coeffs); // 未変換空間 (fitと同じ係数)
    const neutral = build.neutralUntransformed;
    for (let ji = 0; ji < j; ji++) {
      if (!model.jointNames[ji].includes('eye')) continue;
      const sideSign = Math.sign(joints[ji * 3]) || 1;
      let dx = 0;
      let dy = 0;
      let dz = 0;
      let count = 0;
      for (let i = 0; i < model.vertexCount; i++) {
        if (model.eyeWeight[i] <= 128) continue;
        if (Math.sign(identityOnly[i * 3]) !== sideSign) continue;
        dx += neutral[i * 3] - identityOnly[i * 3];
        dy += neutral[i * 3 + 1] - identityOnly[i * 3 + 1];
        dz += neutral[i * 3 + 2] - identityOnly[i * 3 + 2];
        count++;
      }
      if (count > 0) {
        joints[ji * 3] += dx / count;
        joints[ji * 3 + 1] += dy / count;
        joints[ji * 3 + 2] += dz / count;
      }
    }
  }
  return joints;
}

function namedAttribute(array: Float32Array, itemSize: number, name: string): THREE.BufferAttribute {
  const attr = new THREE.BufferAttribute(array, itemSize);
  attr.name = name;
  return attr;
}

/** 全頂点+Zの平坦法線 (Webのライティング方針「写真の陰影のみ」と同じ)。 */
function flatNormalAttribute(count: number): THREE.BufferAttribute {
  const normals = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) normals[i * 3 + 2] = 1;
  return new THREE.BufferAttribute(normals, 3);
}

function zipBlob(files: Zippable): Blob {
  const zipped = zipSync(files, { level: 6 });
  return new Blob([zipped.buffer as ArrayBuffer], { type: 'application/zip' });
}

/** 長辺capまで縮小したコピーを返す (cap以下ならそのままコピー)。 */
function downscaleCanvas(src: HTMLCanvasElement, maxDim: number): HTMLCanvasElement {
  const scale = Math.min(1, maxDim / Math.max(src.width, src.height));
  const out = document.createElement('canvas');
  out.width = Math.max(2, Math.round(src.width * scale));
  out.height = Math.max(2, Math.round(src.height * scale));
  out.getContext('2d')!.drawImage(src, 0, 0, out.width, out.height);
  return out;
}

function canvasTexture(canvas: HTMLCanvasElement): THREE.CanvasTexture {
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/** 面積重み付き頂点法線 (three.jsのcomputeVertexNormalsと同じ流儀)。 */
function computeAreaWeightedNormals(positions: Float32Array, index: Uint32Array): Float32Array {
  const normals = new Float32Array(positions.length);
  for (let t = 0; t < index.length; t += 3) {
    const a = index[t] * 3;
    const b = index[t + 1] * 3;
    const c = index[t + 2] * 3;
    const abx = positions[b] - positions[a];
    const aby = positions[b + 1] - positions[a + 1];
    const abz = positions[b + 2] - positions[a + 2];
    const acx = positions[c] - positions[a];
    const acy = positions[c + 1] - positions[a + 1];
    const acz = positions[c + 2] - positions[a + 2];
    const nx = aby * acz - abz * acy;
    const ny = abz * acx - abx * acz;
    const nz = abx * acy - aby * acx;
    for (const v of [a, b, c]) {
      normals[v] += nx;
      normals[v + 1] += ny;
      normals[v + 2] += nz;
    }
  }
  for (let i = 0; i < normals.length; i += 3) {
    const len = Math.hypot(normals[i], normals[i + 1], normals[i + 2]);
    if (len < 1e-12) continue;
    normals[i] /= len;
    normals[i + 1] /= len;
    normals[i + 2] /= len;
  }
  return normals;
}
