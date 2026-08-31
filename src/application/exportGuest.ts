// ユースケース: 写真1枚から guest zip の成果物を作る。
//
// 6段を順に呼ぶ:
//
//     1. 推論      ランドマーク / 髪マスク / 深度・法線・前景
//     2. フィット   68 点を重ねて相似変換と identity 係数を得る
//     3. 眼球      写真の画素を眼球の極座標 UV へ焼く（左右 2 枚）
//     4. アトラス   写真を公式 UV アトラスへ焼き、残りを表面沿いの伝播で埋める
//     5. 髪シェル   深度 + 髪マスクからグリッドメッシュを作る
//     6. 組み立て   出力契約の値（`GuestArtifacts`）にまとめる
//
// **段「眼球」は段「フィット」の後。** 焼き込みが相似変換を入力に取る（左右の同定と、写真の面内
// 回転の打ち消し）。以前は段「眼球」が段「フィット」より前にあった（写真から 3 色を測るだけだったので
// ランドマークで足りた）。**焼き込みに変えた時点でこの順序は成立しない。**
//
// zip を書くのはこの段の外（`infrastructure/packaging`）。ここが返すのは値だけで、ファイルには
// 触らない。段の合成と「値になった時点で契約を満たしていること」までがユースケースの責務。
//
// 各段の出力は検査画像としてそのまま返す（`InspectionImages`）。
//
// 失敗の伝え方
// ------------
// 段ごとの失敗はそれぞれ固有の例外で上がる（`FaceNotDetectedError` /
// `LandmarkCorrespondenceError` / `DepthFitError` / `EmptyScalpError`）。ここで包み直さないのは、型が
// 失敗の種類（写真を選び直させるのか、対応表が壊れているのか）を持っていて、包むとその区別が
// message の中の文字列に落ちるため。段の名前と対処は例外の型から引ける（`describeFailure`）。
//
// **対処の文がここにあるのは presentation に判断を置かないため。** 「この失敗は写真を選び直せば
// 直るのか、環境の問題なのか」は失敗の意味づけであって表示の都合ではない。
//
// 推論は Port 経由で受け取る
// --------------------------
// このモジュールは `onnxruntime-web` も `@mediapipe/tasks-vision` も知らない。組み立てるのは
// composition root の責務。おかげでフィットとアトラスは偽の Port を渡してテストできる。

import {
  AtlasBake,
  BakeSettings,
  PROVENANCE_BLEND,
  PROVENANCE_PHOTO,
  provenanceInspectionImage,
} from '../domain/atlas/bake';
import {
  AlphaImage,
  GuestArtifacts,
  GuestManifest,
  HairShell,
  RgbImage,
  createGuestManifest,
  hairShellFromImageUv,
  makeGuestArtifacts,
} from '../domain/contract';
import {
  DepthNormalResult,
  HairMask,
  PersonSegmentation,
  ScalarField,
  isFullRect,
  sampleField,
} from '../domain/field';
import {
  ExporterError,
  FaceNotDetectedError,
  GpuUnavailableError,
  InputImageError,
  ModelFileNotFoundError,
  SkinColorUnavailableError,
} from '../domain/errors';
import { EYE_SIDES, EyeSide } from '../domain/eyes/layout';
import {
  EyeAlbedo,
  bakeEyeAlbedos,
  provenanceInspectionImage as eyeProvenanceImage,
} from '../domain/eyes/bake';
import { eyeUvGeometries } from '../domain/eyes/geometry';
import { headInferenceSquare, headOnlySquare } from '../domain/gnm/crop';
import {
  HeadFit,
  LandmarkCorrespondenceError,
  LandmarkModel,
  Similarity2d,
  buildDenseLandmarkModel,
  evaluateIbug68,
  evaluateModel,
  fitHead,
  regularizationScheduleFor,
  selectIbug68,
  selectModelPoints,
  xyOf,
} from '../domain/gnm/fit';
import { GnmHeadAsset, verticesOf } from '../domain/gnm/model';
import { RegionSilhouetteFit, refineEarNeckFit } from '../domain/gnm/silhouette';
import { DepthFitError, fitDepthToLandmarkZ } from '../domain/hair/depthFit';
import { hairShellMask } from '../domain/hair/mask';
import { GUIDED_MASK_MAX_DIMENSION } from '../domain/hair/maskRefine';
import { EmptyScalpError } from '../domain/hair/scalp';
import {
  DEFAULT_HAIR_SHELL_PARAMS,
  HairShellResult,
  buildHairShell,
} from '../domain/hair/shell';
import {
  InspectionImages,
  PhotoCanvas,
  downscaled,
  encodeNormalRgb,
  grayToRgb,
  normalizeToUint8,
  regionCanvas,
  triangleEdges,
} from '../domain/inspection';
import { PhotoRgb, maskedAverageSrgb, resampleLongestSide, validatePhoto } from '../domain/photo';
import {
  AtlasBaker,
  DepthNormalEstimator,
  FaceLandmarkDetector,
  HairImageProcessor,
  PersonSegmenter,
} from './ports';
import { DEFAULT_SETTINGS, ExportSettings, MILLIMETERS_PER_METER } from './settings';

/**
 * 段の名前。表示とログの正本。
 *
 * 名前で参照するのは、段を1つ挟んだときに添字が黙って別の段を指すのを防ぐため（実際に段「眼球」を
 * 挟んで添字がずれた）。
 */
export const STAGE_INFERENCE = '推論';
export const STAGE_FIT = 'フィット';
export const STAGE_EYES = '眼球';
export const STAGE_ATLAS = 'アトラス';
export const STAGE_HAIR = '髪シェル';
export const STAGE_ASSEMBLE = '組み立て';

/** 段の名前を走る順に並べたもの（1 始まりの番号がそのまま index + 1）。 */
export const STAGE_NAMES: readonly string[] = [
  STAGE_INFERENCE,
  STAGE_FIT,
  STAGE_EYES,
  STAGE_ATLAS,
  STAGE_HAIR,
  STAGE_ASSEMBLE,
];

/**
 * 段に固有の失敗と、その段の名前。失敗を包み直さずに段を引くための表。
 *
 * `EyeUvLayoutError` はここに入れない。あれは写真ではなく **GNM アセット**の側の問題（眼球 UV が
 * 同心円でない）で、写真を選び直しても直らない。
 */
const STAGE_OF_ERROR: readonly [new (...args: never[]) => Error, string][] = [
  [FaceNotDetectedError, STAGE_INFERENCE],
  [LandmarkCorrespondenceError, STAGE_FIT],
  [DepthFitError, STAGE_HAIR],
  [EmptyScalpError, STAGE_HAIR],
];

/**
 * 失敗の型 → 人向けの対処。上から順に照合するので、狭い型を先に置く。
 *
 * 閾値の数字はここに書かない。閾値を持っているのは検証側で、例外の message がその値を添えて上がって
 * くる。ここに書き写すと片方だけ変わったときに黙って嘘になる。
 */
const REMEDIES: readonly [new (...args: never[]) => Error, string][] = [
  [
    GpuUnavailableError,
    'WebGPU が使える環境（Chrome / Edge の最近の版）で開いてください。' +
      'WASM へ落ちても動きますが、DAViD の推論に数十秒かかります。',
  ],
  [
    ModelFileNotFoundError,
    'モデルかアセットを取得できませんでした。ネットワークを確認して読み込み直してください。',
  ],
  [
    FaceNotDetectedError,
    '顔を検出できませんでした。正面を向いた顔が大きく写っている写真を選んでください。',
  ],
  [InputImageError, '写真そのものが要求を満たしません。別の写真を選んでください。'],
  [
    LandmarkCorrespondenceError,
    '写真ではなく対応表かランドマーク定義の不具合です' +
      '（domain/gnm/fit.ts の MEDIAPIPE_IBUG68 を確認してください）。',
  ],
  [
    DepthFitError,
    '深度を GNM のスケールへ合わせられませんでした。顔の前面が大きく写っている写真を' +
      '選んでください。',
  ],
  [
    EmptyScalpError,
    '髪の範囲に頭部が掛かっていません。フィットが破綻している可能性があるので、' +
      '検査画像の landmarkFit を確認してください。',
  ],
];

/** 段「アトラス」の投影検査画像に打つ点数の上限。 */
export const MAX_PROJECTION_POINTS = 200_000;

const LANDMARK_COLOR: readonly [number, number, number] = [0, 220, 90];
const FITTED_LANDMARK_COLOR: readonly [number, number, number] = [240, 60, 60];
const PROJECTION_COLOR: readonly [number, number, number] = [0, 190, 60];
const BLENDED_PROJECTION_COLOR: readonly [number, number, number] = [40, 120, 240];
const REJECTED_PROJECTION_COLOR: readonly [number, number, number] = [255, 0, 255];
const HAIR_MASK_COLOR: readonly [number, number, number] = [250, 80, 220];
const WIRE_COLOR: readonly [number, number, number] = [255, 230, 60];

/** 失敗を「どの段で / 何が / どうすれば」に整えたもの。 */
export interface FailureReport {
  /** 段の名前。段に固有でない失敗は null。 */
  readonly stage: string | null;
  /** 例外の型名。同じ対処でも型が違えば原因が違う。 */
  readonly errorType: string;
  /** 例外の message（domain 側が日本語で書いている）。 */
  readonly cause: string;
  /** 人向けの対処。表に無い失敗は null（＝想定外なのでバグ）。 */
  readonly remedy: string | null;
}

/** 例外を FailureReport にする。入口はこれをそのまま表示すればよい。 */
export function describeFailure(error: unknown): FailureReport {
  const value = error instanceof Error ? error : new Error(String(error));
  return {
    stage: firstMatch(STAGE_OF_ERROR, value),
    errorType: value.name || value.constructor.name,
    cause: value.message,
    remedy: firstMatch(REMEDIES, value),
  };
}

function firstMatch(
  table: readonly [new (...args: never[]) => Error, string][],
  error: Error,
): string | null {
  for (const [type, value] of table) {
    if (error instanceof type) return value;
  }
  return null;
}

/** 想定内の失敗か（入口はこれを捕まえて `describeFailure` を出す）。 */
export function isPipelineError(error: unknown): boolean {
  return (
    error instanceof ExporterError ||
    error instanceof LandmarkCorrespondenceError ||
    error instanceof DepthFitError ||
    error instanceof EmptyScalpError
  );
}

/** 3Dビューの遅延入力。既存配列の参照を束ねるだけ。 */
export interface DebugSceneSource {
  readonly vertices: Float64Array;
  readonly asset: GnmHeadAsset;
  readonly skinAlbedo: Uint8Array;
  readonly atlasSize: number;
  readonly eyeAlbedos: Readonly<Record<EyeSide, Uint8Array>>;
  readonly eyeTextureSize: number;
  readonly hair: HairShell | null;
  readonly hairAlbedo: RgbImage | null;
  readonly hairAlpha: AlphaImage | null;
}

/** `exportGuest` が返すもの。 */
export interface ExportOutcome {
  /** 出力契約の値（zip に入るもの）。 */
  readonly artifacts: GuestArtifacts;
  /** 各段の検査画像。 */
  readonly inspection: InspectionImages;
  /**
   * 側 → 段「眼球」が焼いたテクスチャ。
   *
   * 画像そのものは `artifacts` にも入っている。こちらは写真上の虹彩半径など、検査用の測定値を
   * presentation へ渡すために返す。
   */
  readonly eyeAlbedos: Readonly<Record<EyeSide, EyeAlbedo>>;
  readonly debugSceneSource: DebugSceneSource;
  /** 段「フィット」の結果（残差の表示に使う）。 */
  readonly headFit: HeadFit;
  /** 段「髪シェル」の中間結果。髪が無ければ null。 */
  readonly hairShell: HairShellResult | null;
  /** 段「アトラス」の結果（内訳の表示に使う）。 */
  readonly atlas: AtlasBake;
}

/**
 * 写真1枚から guest zip の成果物と検査画像を作る。
 *
 * @throws 段に固有の失敗がそのまま上がる（`STAGE_OF_ERROR` 参照）
 */
export async function exportGuest(input: {
  photo: PhotoRgb;
  asset: GnmHeadAsset;
  landmarkDetector: FaceLandmarkDetector;
  segmenter: PersonSegmenter;
  depthNormal: DepthNormalEstimator;
  atlasBaker: AtlasBaker;
  hairImageProcessor: HairImageProcessor;
  settings?: ExportSettings;
  exporterVersion: string;
  onStage?: (stage: string) => void;
}): Promise<ExportOutcome> {
  const settings = input.settings ?? DEFAULT_SETTINGS;
  const { photo, asset } = input;
  validatePhoto(photo);
  const imageSize: [number, number] = [photo.width, photo.height];
  const canvas = PhotoCanvas.of(photo);
  const notify = input.onStage ?? ((): void => undefined);

  // 段1 推論 -------------------------------------------------------------
  // 検出器は 478 点（顔メッシュ 468 + 虹彩 10）をそのまま返す。フィットは前 468 点だけを、眼球は
  // 虹彩 10 点を見る（どの点を使うかは各段の関心）。
  //
  // 対応点モデルは写真に依らないので段の前で1回だけ作る。
  notify(STAGE_INFERENCE);
  const landmarkModel = buildDenseLandmarkModel(asset, asset.dense);
  // 平均形状の xy。**identity を当てた形ではない** — 切り出しも髪の種もフィットの前に要るので、粗い
  // 相似変換で写した平均形状で決める。1 つに束ねているのは、切り出しと髪の種が同じ形を見ていることを
  // 崩さないため。
  const meshXy = new Float64Array(asset.mesh.vertexCount * 2);
  for (let vertex = 0; vertex < asset.mesh.vertexCount; vertex++) {
    meshXy[vertex * 2] = asset.mesh.templateVertexPositions[vertex * 3];
    meshXy[vertex * 2 + 1] = asset.mesh.templateVertexPositions[vertex * 3 + 1];
  }

  const { landmarks478, segmentation, headInference, bodyInference } = await runInferencePorts(
    photo,
    input.landmarkDetector,
    input.segmenter,
    input.depthNormal,
    landmarkModel,
    meshXy,
    imageSize,
  );

  // 髪と装飾品を足すのはここ（アダプタではない）。装飾品は顎より下でも当たるので、位置の門に
  // ランドマークが要る。
  const hairMask = input.hairImageProcessor.refineMask(
    photo,
    hairShellMask({
      hair: segmentation.hair,
      accessory: segmentation.accessory,
      photoLandmarks: landmarks478,
      landmarkModel,
      meshXy,
      imageSize,
    }),
    Math.min(settings.hairTextureSize, GUIDED_MASK_MAX_DIMENSION),
  );

  // 基準の肌色はアトラスの塗りつぶしにだけ使う内部の値で、契約には出さない（口腔壁の色と、写真が
  // どこからも届かないテクセルの下地）。**顔の肌の場**から測る — アトラスからではない（アトラスの
  // 平均は首・服・耳まで含むうえ、焼く範囲の設定を変えると口腔壁の色が動く）。
  const skinBaseColor = maskedAverageSrgb(
    photo,
    resampleSegmentationToPhoto(segmentation.faceSkin, photo),
  );
  if (skinBaseColor === null) {
    throw new SkinColorUnavailableError(
      '顔の肌が1画素も取れず、基準の肌色を測れなかった（セグメンテーションが破綻している可能性）',
    );
  }
  let inspection: InspectionImages = inferenceInspection(
    canvas,
    landmarks478,
    hairMask,
    headInference,
    bodyInference,
  );

  // 段2 フィット ---------------------------------------------------------
  notify(STAGE_FIT);
  let headFit = fitHead(landmarks478, landmarkModel, {
    regularizationSchedule: regularizationScheduleFor(landmarkModel, settings.disagreementScale),
    identityClip: settings.identityClip,
  });
  const regionFit: RegionSilhouetteFit = refineEarNeckFit({
    initial: headFit,
    photoLandmarks: landmarks478,
    landmarkModel,
    asset,
    faceSkin: segmentation.faceSkin,
    bodySkin: segmentation.bodySkin,
    imageSize,
    identityClip: settings.identityClip,
  });
  headFit = regionFit.headFit;
  inspection = {
    ...inspection,
    landmarkFit: landmarkFitInspection(
      canvas,
      selectModelPoints(landmarks478, landmarkModel),
      landmarkModel,
      headFit,
    ),
    silhouetteFit: silhouetteFitInspection(canvas, regionFit),
  };
  const mesh = asset.mesh;

  // 段3 眼球 -------------------------------------------------------------
  // **相似変換が要るのでフィットの後。** 左右の同定と写真の面内回転の打ち消しの両方に使う。
  notify(STAGE_EYES);
  const eyeGeometries = eyeUvGeometries(mesh);
  const eyeAlbedos = bakeEyeAlbedos({
    photo,
    landmarks478,
    mesh,
    similarity: headFit.similarity,
    geometries: eyeGeometries,
    size: settings.eyeTextureSize,
  });
  inspection = { ...inspection, ...eyeInspection(eyeAlbedos) };

  // 段4 アトラス ---------------------------------------------------------
  notify(STAGE_ATLAS);
  const vertices = verticesOf(asset, headFit.identity);
  const bakeSettings: BakeSettings = {
    atlasSize: settings.skinAtlasSize,
    minFacing: 0.3,
    facingSoftness: 0.15,
    foregroundThreshold: settings.atlasForegroundThreshold,
    foregroundExponent: settings.atlasForegroundExponent,
    harmonicScreening: settings.atlasHarmonicScreening,
    depthCellPx: 4,
    occlusionTolerance: 0.004,
    chartDilationTexels: 8,
    interiorScale: 0.7,
  };
  const atlas = input.atlasBaker.bake({
    photo,
    vertices,
    triangles: mesh.triangles,
    vertexUvs: mesh.vertexUvs,
    componentId: mesh.componentId,
    similarity: headFit.similarity,
    personMask: bodyInference.foreground,
    skinBaseColor,
    settings: bakeSettings,
    // 耳の付け根を越えて補完させないための領域。
    fillRegionId: mesh.earRegion,
    photoOnlyRegion: mesh.atlasPhotoOnlyRegion,
    mouthRimRegion: mesh.mouthRimRegion,
  });
  inspection = {
    ...inspection,
    ...atlasInspection(canvas, atlas, headFit.similarity, bodyInference.foreground),
  };

  // 段5 髪シェル ---------------------------------------------------------
  notify(STAGE_HAIR);
  const legacyDepthFit = fitDepthToLandmarkZ(
    headInference.depth,
    selectIbug68(landmarks478),
    zOf(evaluateIbug68(landmarkModel, headFit.identity)),
    imageSize,
  );
  const shellBuild = buildHairShell({
    vertices,
    triangles: mesh.triangles,
    similarity: headFit.similarity,
    hairMask,
    depth: headInference.depth,
    normal: headInference.normal,
    imageSize,
    params: {
      ...DEFAULT_HAIR_SHELL_PARAMS,
      // 入口はミリ、domain はメートル。換算は settings が持つ係数だけを使う。
      liftMeters: settings.hairLiftMm / MILLIMETERS_PER_METER,
      rolloffMeters: settings.hairRolloffMm / MILLIMETERS_PER_METER,
    },
    depthFit: legacyDepthFit,
  });
  if (shellBuild !== null) {
    inspection = { ...inspection, ...hairInspection(photo, shellBuild, headFit.similarity) };
  }

  // 段6 組み立て ---------------------------------------------------------
  notify(STAGE_ASSEMBLE);
  // 髪テクスチャは写真を縮めたもの。UV は正規化座標なので縦横比を保った縮小なら対応は変わらない。
  // alpha は同じ解像度で引き直す（契約が形の一致を要求する）。
  let hairAlbedo: RgbImage | null =
    shellBuild === null ? null : resampleLongestSide(photo, settings.hairTextureSize);
  const hairAlpha: AlphaImage | null =
    hairAlbedo === null
      ? null
      : maskToUint8(hairMask.confidence, hairAlbedo.width, hairAlbedo.height);
  if (hairAlbedo !== null && hairAlpha !== null) {
    hairAlbedo = input.hairImageProcessor.decontaminateTexture(hairAlbedo, hairAlpha);
  }

  const manifest: GuestManifest = createGuestManifest({
    identity: headFit.identity,
    gnmVersion: asset.gnmVersion,
    gnmVariant: asset.gnmVariant,
    // 一辺は焼いた画像から引く（逆算した値を持ち回らない）。manifest と画像がズレる経路を作らない。
    atlasSize: atlas.surface.size,
    eyeTextureSize: eyeAlbedos[EYE_SIDES[0]].size,
    capturedAt: new Date(),
    exporterVersion: input.exporterVersion,
  });
  const artifacts = makeGuestArtifacts({
    manifest,
    skinAlbedo: atlas.albedo,
    eyeAlbedos: {
      left: eyeAlbedos.left.image,
      right: eyeAlbedos.right.image,
    },
    hair: shellBuild === null ? null : toContractShell(shellBuild),
    hairAlbedo,
    hairAlpha,
  });

  return {
    artifacts,
    inspection,
    eyeAlbedos,
    debugSceneSource: {
      vertices,
      asset,
      skinAlbedo: artifacts.skinAlbedo,
      atlasSize: manifest.atlas_size,
      eyeAlbedos: artifacts.eyeAlbedos,
      eyeTextureSize: manifest.eye_texture_size,
      hair: artifacts.hair,
      hairAlbedo: artifacts.hairAlbedo,
      hairAlpha: artifacts.hairAlpha,
    },
    headFit,
    hairShell: shellBuild,
    atlas,
  };
}

/**
 * ランドマーク → 切り出し → DAViD の順に走らせ、DAViD にセグメンタを重ねる。
 *
 * **ランドマークが先。** DAViD の切り出しはメッシュの投影域から決めるので（`domain/gnm/crop`）、
 * ランドマークが無いと切り出せない。3 つを同時に投げていた頃は切り出しを画像の中央正方形にするしか
 * なく、縦長の写真で顎から下が推論の外へ落ちていた。
 *
 * 切り出しを 2 つ走らせる理由: 覆うべき範囲が用途で違い、**1 つの正方形では両立しない**。
 *
 *     頭部（`headOnlySquare`）      深度・法線。隣人が入らないよう詰める
 *     全体（`headInferenceSquare`） 人物前景。胸まで覆う
 *
 * **2 つを別々の `DepthNormalResult` で返す**のは、あの型が「3 つが同じ切り出しに乗る」ことを不変
 * 条件にしているため — 詰めた深度と広い前景を 1 つに詰めると、その保証が嘘になる。
 */
async function runInferencePorts(
  photo: PhotoRgb,
  landmarkDetector: FaceLandmarkDetector,
  segmenter: PersonSegmenter,
  depthNormal: DepthNormalEstimator,
  landmarkModel: LandmarkModel,
  meshXy: Float64Array,
  imageSize: readonly [number, number],
): Promise<{
  landmarks478: Float64Array;
  segmentation: PersonSegmentation;
  headInference: DepthNormalResult;
  bodyInference: DepthNormalResult;
}> {
  const landmarks478 = await landmarkDetector.detect(photo);
  const headSquare = headOnlySquare(landmarks478, landmarkModel, meshXy, imageSize);
  const bodySquare = headInferenceSquare(landmarks478, landmarkModel, meshXy, imageSize);
  // ブラウザでは推論もセグメンテーションも同じ 1 スレッド（あるいは同じ GPU キュー）に乗るので、
  // 並行に投げても順に入る。デスクトップ側が worker を 1 本に限っているのと同じ理由で、GPU へ
  // 複数同時投入する経路は作らない。
  const headInference = await depthNormal.estimateSquare(photo, headSquare);
  const bodyInference = await depthNormal.estimateSquare(photo, bodySquare);
  const segmentation = await segmenter.segment(photo);
  return { landmarks478, segmentation, headInference, bodyInference };
}

/** (P, 3) から z だけ抜いた (P,)。 */
function zOf(points3: Float64Array): Float64Array {
  const out = new Float64Array(points3.length / 3);
  for (let point = 0; point < out.length; point++) out[point] = points3[point * 3 + 2];
  return out;
}

/** 顔の肌の場を写真の解像度へ引き直す（`maskedAverageSrgb` が写真と同じ形を要求する）。 */
function resampleSegmentationToPhoto(field: ScalarField, photo: PhotoRgb): Float32Array {
  if (isFullRect(field.rect) && field.width === photo.width && field.height === photo.height) {
    return field.values;
  }
  const out = new Float32Array(photo.width * photo.height);
  for (let row = 0; row < photo.height; row++) {
    const v = (row + 0.5) / photo.height;
    for (let column = 0; column < photo.width; column++) {
      out[row * photo.width + column] = sampleField(field, (column + 0.5) / photo.width, v);
    }
  }
  return out;
}

/** 生成した髪シェルを出力契約の型へ移す（UV の向きはここで読み替わる）。 */
function toContractShell(shell: HairShellResult): HairShell {
  return hairShellFromImageUv(shell.positions, shell.uvs, shell.triangles);
}

/**
 * マスクを `width x height` の uint8 にする。
 *
 * 場が既に画像全体をその解像度で覆っているなら引き直さない。引き直すと bilinear が一度かかって縁が
 * わずかに鈍る（同じ値を得るための無駄）。
 */
function maskToUint8(mask: ScalarField, width: number, height: number): AlphaImage {
  const data = new Uint8Array(width * height);
  if (isFullRect(mask.rect) && mask.width === width && mask.height === height) {
    for (let pixel = 0; pixel < data.length; pixel++) {
      data[pixel] = Math.round(Math.min(1, Math.max(0, mask.values[pixel])) * 255);
    }
    return { data, width, height };
  }
  for (let row = 0; row < height; row++) {
    const v = (row + 0.5) / height;
    for (let column = 0; column < width; column++) {
      const value = sampleField(mask, (column + 0.5) / width, v);
      data[row * width + column] = Math.round(Math.min(1, Math.max(0, value)) * 255);
    }
  }
  return { data, width, height };
}

// ---------------------------------------------------------------------------
// 検査画像
// ---------------------------------------------------------------------------
/**
 * 段「推論」の 5 枚。
 *
 * `hairMask` に描くのは**雑音床を引いた後の確信度** — つまり zip に入る `hair_alpha` そのもの。生の
 * マスクを描くと、検査画像では膜が見えているのに出力では切られている（あるいは逆）という食い違いが
 * 起きる。
 *
 * 深度・法線と前景を**別の推論から取る**のは、後段がそう使うから。**両者は覆う範囲が違うので、
 * `depth` と `foreground` の黒い縁が一致しないのが正常。**
 */
function inferenceInspection(
  canvas: PhotoCanvas,
  landmarks478: Float64Array,
  hairMask: HairMask,
  headInference: DepthNormalResult,
  bodyInference: DepthNormalResult,
): InspectionImages {
  return {
    photoLandmarks: canvas.withPoints(landmarks478, LANDMARK_COLOR, null, 1),
    hairMask: canvas.tinted(canvas.rasterizeField(hairMask.confidence), HAIR_MASK_COLOR),
    depth: canvas.fieldImage(headInference.depth),
    normal: downscaled(
      encodeNormalRgb(
        headInference.normal,
        headInference.depth.width,
        headInference.depth.height,
      ),
    ),
    foreground: canvas.fieldImage(bodyInference.foreground),
  };
}

/**
 * 段「フィット」の 1 枚: 写真の対応点（緑）とフィット後の GNM の同じ点（赤）。
 *
 * **フィットが実際に使った点をそのまま描く**（68 点ではなく密対応の 468 点）。片方だけ 68 点にすると、
 * 緑と赤が別の集合になって「どこが合っていないか」が読めない。
 */
function landmarkFitInspection(
  canvas: PhotoCanvas,
  photoPoints: Float64Array,
  landmarkModel: LandmarkModel,
  headFit: HeadFit,
): RgbImage {
  const fitted = headFit.similarity.apply(xyOf(evaluateModel(landmarkModel, headFit.identity)));
  const withPhoto = canvas.withPoints(photoPoints, LANDMARK_COLOR, null, 2);
  return canvas.withPoints(fitted, FITTED_LANDMARK_COLOR, withPhoto, 1);
}

/** 体肌の観測輪郭（緑）と、フィットしたGNMの耳・首外周（赤）。 */
function silhouetteFitInspection(canvas: PhotoCanvas, fit: RegionSilhouetteFit): RgbImage {
  const observed = canvas.withPoints(fit.observedPixels, LANDMARK_COLOR, null, 3);
  return canvas.withPoints(fit.fittedPixels, FITTED_LANDMARK_COLOR, observed, 2);
}

/**
 * 段「眼球」の左右の焼いた絵と、その由来。
 *
 * 由来は左右を横に並べて 1 枚にする。左右で伸ばしの量が違えば側の同定かフィットのどちらかがおかしい
 * ので、並べた方が読める。
 */
function eyeInspection(eyeAlbedos: Readonly<Record<EyeSide, EyeAlbedo>>): InspectionImages {
  const size = eyeAlbedos.left.size;
  const provenance = [EYE_SIDES[0], EYE_SIDES[1]].map((side) =>
    eyeProvenanceImage(eyeAlbedos[side].provenance, size),
  );
  const merged = new Uint8Array(size * size * 2 * 3);
  for (let row = 0; row < size; row++) {
    for (let side = 0; side < 2; side++) {
      for (let column = 0; column < size; column++) {
        const source = (row * size + column) * 3;
        const target = (row * size * 2 + side * size + column) * 3;
        merged[target] = provenance[side].data[source];
        merged[target + 1] = provenance[side].data[source + 1];
        merged[target + 2] = provenance[side].data[source + 2];
      }
    }
  }
  return {
    leftEyeAlbedo: { data: eyeAlbedos.left.image, width: size, height: size },
    rightEyeAlbedo: { data: eyeAlbedos.right.image, width: size, height: size },
    eyeAlbedoProvenance: { data: merged, width: size * 2, height: size },
  };
}

/** 段「アトラス」の検査画像。投影位置とDAViDの採否を別々に残す。 */
function atlasInspection(
  canvas: PhotoCanvas,
  bake: AtlasBake,
  similarity: Similarity2d,
  personMask: ScalarField,
): InspectionImages {
  const photoTexels: number[] = [];
  const blendTexels: number[] = [];
  for (let texel = 0; texel < bake.provenance.length; texel++) {
    if (bake.provenance[texel] === PROVENANCE_PHOTO) photoTexels.push(texel);
    else if (bake.provenance[texel] === PROVENANCE_BLEND) blendTexels.push(texel);
  }
  const project = (texels: readonly number[]): Float64Array => {
    const stride = Math.max(1, Math.ceil(texels.length / MAX_PROJECTION_POINTS));
    const points: number[] = [];
    for (let slot = 0; slot < texels.length; slot += stride) {
      const texel = texels[slot];
      const [x, y] = similarity.applyPoint(
        bake.surface.position[texel * 3],
        bake.surface.position[texel * 3 + 1],
      );
      points.push(x, y);
    }
    return Float64Array.from(points);
  };

  const size = bake.surface.size;
  const candidate = new Uint8Array(size * size);
  const candidatePixels: number[] = [];
  const candidateTexels: number[] = [];
  for (let texel = 0; texel < candidate.length; texel++) {
    if (bake.surface.triangleIndex[texel] < 0) continue;
    if (bake.surface.chartIndex[texel] !== 0) continue;
    if (bake.surface.normal[texel * 3 + 2] < bake.settings.minFacing) continue;
    candidate[texel] = 1;
    const [x, y] = similarity.applyPoint(
      bake.surface.position[texel * 3],
      bake.surface.position[texel * 3 + 1],
    );
    candidatePixels.push(x, y);
    candidateTexels.push(texel);
  }
  const acceptedPoints: number[] = [];
  const rejectedPoints: number[] = [];
  const accepted = new Uint8Array(candidate.length);
  candidateTexels.forEach((texel, slot) => {
    const x = candidatePixels[slot * 2];
    const y = candidatePixels[slot * 2 + 1];
    const foreground = sampleField(personMask, x / canvas.photoWidth, y / canvas.photoHeight);
    if (foreground >= bake.settings.foregroundThreshold) {
      accepted[texel] = 1;
      acceptedPoints.push(x, y);
    } else {
      rejectedPoints.push(x, y);
    }
  });

  const rejectedOverlay = canvas.withPoints(
    Float64Array.from(rejectedPoints),
    REJECTED_PROJECTION_COLOR,
    null,
    0,
  );
  const gateOverlay = canvas.withPoints(
    Float64Array.from(acceptedPoints),
    PROJECTION_COLOR,
    rejectedOverlay,
    0,
  );

  const albedoGate = new Uint8Array(bake.albedo.length);
  for (let texel = 0; texel < candidate.length; texel++) {
    if (accepted[texel] !== 0) {
      albedoGate[texel * 3] = bake.albedo[texel * 3];
      albedoGate[texel * 3 + 1] = bake.albedo[texel * 3 + 1];
      albedoGate[texel * 3 + 2] = bake.albedo[texel * 3 + 2];
    } else if (candidate[texel] !== 0) {
      albedoGate[texel * 3] = REJECTED_PROJECTION_COLOR[0];
      albedoGate[texel * 3 + 1] = REJECTED_PROJECTION_COLOR[1];
      albedoGate[texel * 3 + 2] = REJECTED_PROJECTION_COLOR[2];
    } else {
      for (let channel = 0; channel < 3; channel++) {
        albedoGate[texel * 3 + channel] = Math.round(bake.albedo[texel * 3 + channel] * 0.2);
      }
    }
  }

  const blendOverlay = canvas.withPoints(
    project(blendTexels),
    BLENDED_PROJECTION_COLOR,
    null,
    0,
  );
  return {
    atlasProjection: canvas.withPoints(
      project(photoTexels),
      PROJECTION_COLOR,
      blendOverlay,
      0,
    ),
    atlasProjectionGate: gateOverlay,
    atlasAlbedoGate: downscaled({ data: albedoGate, width: size, height: size }),
    atlasAlbedo: downscaled({ data: bake.albedo, width: size, height: size }),
    atlasProvenance: downscaled(provenanceInspectionImage(bake.provenance, size)),
  };
}

/**
 * 段「髪シェル」の 2 枚: 三角形のワイヤと、格子上の厚み。
 *
 * ワイヤは写真全体ではなくシェルの範囲を切り出して描く。全身写真では頭が小さく、全体を間引いた土台
 * では格子が塗り潰しに見えてメッシュを検査できない。
 */
function hairInspection(
  photo: PhotoRgb,
  shell: HairShellResult,
  similarity: Similarity2d,
): InspectionImages {
  const pixels = new Float64Array(shell.vertexCount * 2);
  for (let vertex = 0; vertex < shell.vertexCount; vertex++) {
    const [x, y] = similarity.applyPoint(
      shell.positions[vertex * 3],
      shell.positions[vertex * 3 + 1],
    );
    pixels[vertex * 2] = x;
    pixels[vertex * 2 + 1] = y;
  }
  const { canvas: region, localPoints } = regionCanvas(photo, pixels);
  const { starts, ends } = triangleEdges(shell.triangles);
  const startXy = new Float64Array(starts.length * 2);
  const endXy = new Float64Array(ends.length * 2);
  for (let edge = 0; edge < starts.length; edge++) {
    startXy[edge * 2] = localPoints[starts[edge] * 2];
    startXy[edge * 2 + 1] = localPoints[starts[edge] * 2 + 1];
    endXy[edge * 2] = localPoints[ends[edge] * 2];
    endXy[edge * 2 + 1] = localPoints[ends[edge] * 2 + 1];
  }
  return {
    hairShellWire: region.withSegments(startXy, endXy, WIRE_COLOR),
    hairThickness: grayToRgb(
      normalizeToUint8(shell.thickness),
      shell.grid.columns,
      shell.grid.rows,
    ),
  };
}
