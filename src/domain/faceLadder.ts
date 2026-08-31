// 顔検出の二段検出（解像度の階段 → 主役の周りを切って再検出）。
//
// **正本はデスクトップ側（1-10/2608_Obayashi_GNMHeadExporter）の
// `infrastructure/face_landmarks.py` の `detect_two_pass`。** あちらは縮小に PIL が要るので
// infrastructure に置いているが、こちらは PIL と同じ縮小を `domain/resample` に持っているので純粋な
// 層に置ける（**依存が内向きのままで、ブラウザ無しで検証できる**）。
//
// 縮小は **LANCZOS**（あちらと同じ）。canvas の `drawImage` で代用しない — 直している不具合が縮小の
// エイリアシングそのものなので、フィルタの質を落とす近似は原因側に戻る。
//
// ## 写真をそのまま検出器へ渡さない理由
//
// `face_landmarker.task` の中身は検出器と 478 点の推定器の 2 本で、MediaPipe 自身が「顔を見つける →
// その周りを切る → 点を取る」をやっている。**こちらが足すのは、内側の検出へ渡す画像の解像度を選ぶ
// 層だけ。** 必要な理由は 2 つあり、向きが逆:
//
//     大きすぎる  内部の縮小がエイリアシングを起こし、正面の明瞭な顔でも検出が落ちる
//     小さすぎる  顔が数十画素になり、位置が取れない
//
// **そして「良い解像度」は単調でない** — ある写真が長辺 512 で通って 1024 で落ち、別の写真が 512 で
// 落ちて 1024 で通る。だから 1 つの値を選ぶ設計は成立せず、階段を探索する。
//
//     段1（探索）  写真全体を階段の**全段**で縮小して検出し、出てきた顔を全部集める。
//                  集めた顔から主役を 1 人選ぶのが `domain/faceSubject`
//     段2（精密）  主役の一辺を `REFINE_CROP_SPAN_FACTOR` 倍した正方形を**元解像度から**切り出し、
//                  同じ階段を大きい側から試して再検出し、全体座標へ戻す
//
// **段1 は通った段で打ち切らない。** どの解像度で顔が見えるかは写真ごとに違ううえ、段によって見える
// 顔の数も順序も変わる。1 段で止めると顔を取り逃がし、選ばれる顔が解像度で揺れる。
//
// **段1 の結果を最終出力にしない。** 段1 は縮小画像なので、大きな写真では顔幅が数十画素になり
// ランドマークの精度が出ない（2160x3840 の写真で口の位置がずれるのはこれ）。段1 は位置だけを担い、
// 点の精度は段2 が持つ。
//
// **段2 が全滅したら段1 の粗い結果で妥協せずに落とす。** 妥協して返すと、後段は「検出できた点」と
// して精度の落ちた点を受け取り、誰も気付けない。

import { FaceNotDetectedError } from './errors';
import {
  FaceSquare,
  faceSquareOfLandmarks,
  imageCenter,
  sameFaceIndex,
  subjectFace,
} from './faceSubject';
import { PhotoRgb, cropPhotoRect } from './photo';
import { LANCZOS3, resamplePilToLongSide } from './resample';

/** 探索階段の下端（長辺、画素）。ここから `LADDER_STEP` 倍で写真の長辺まで登る。 */
export const SCOUT_LADDER_FLOOR = 256;

/** 階段の刻み（長辺の倍率）。2 は「解像度を 1 オクターブずつ動かす」。 */
export const LADDER_STEP = 2;

/**
 * 段2 のクロップの一辺 / 段1 が測った顔の一辺。
 *
 * 段1 は縮小画像なので一辺の推定が粗い。2 倍取れば推定が半分〜2 倍外れても顔がクロップから出ない。
 * 大きすぎる方の害は小さい（段2 は階段で解像度を選び直すので、失敗ではなく解像度の選択に吸収される）。
 */
export const REFINE_CROP_SPAN_FACTOR = 2.0;

/**
 * 1 回の推論で写っている顔を全部返す関数。
 *
 * 各要素は (点数, 2) を平坦にした**その画像の画素座標**。1 つも無ければ `FaceNotDetectedError`。
 */
export type DetectFaces = (photo: PhotoRgb) => Float64Array[];

/**
 * `floor` から `step` 倍で登り、最後に `longSide` そのものを置く長辺の列（昇順）。
 *
 * 末尾を元の長辺にするのは、**縮小しないと通らない写真と、縮小すると通らない写真の両方を階段で
 * 覆う**ため。写真より大きい段は縮小が効かず同じ検出になるので入れない。
 */
export function scaleLadder(
  longSide: number,
  floor = SCOUT_LADDER_FLOOR,
  step = LADDER_STEP,
): number[] {
  if (longSide < 1) throw new Error(`長辺が ${longSide}`);
  if (floor < 1 || step < 2) throw new Error(`階段が作れない: floor=${floor} step=${step}`);
  const rungs: number[] = [];
  let side = floor;
  while (side < longSide) {
    rungs.push(side);
    side *= step;
  }
  rungs.push(longSide);
  return [...new Set(rungs)];
}

/**
 * 顔の外接正方形を `spanFactor` 倍したクロップ矩形 (left, top, right, bottom)。
 *
 * 受け取るのがランドマークではなく `FaceSquare` なのは、段1 が選ぶ主役が**複数段の検出を束ねた
 * 代表**で、対応するランドマークがそもそも 1 組に定まらないため。
 *
 * 画像の外へは出さない（端の顔では正方形が崩れるが、はみ出した領域を足しても情報は増えない）。
 */
export function cropBoxAroundFace(
  face: FaceSquare,
  width: number,
  height: number,
  spanFactor: number,
): [number, number, number, number] {
  const half = (face.span * spanFactor) / 2;
  return [
    Math.trunc(Math.max(0, face.centerX - half)),
    Math.trunc(Math.max(0, face.centerY - half)),
    Math.trunc(Math.min(width, face.centerX + half)),
    Math.trunc(Math.min(height, face.centerY + half)),
  ];
}

interface Rung {
  readonly longSide: number;
  readonly photo: PhotoRgb;
  readonly scaleX: number;
  readonly scaleY: number;
}

/**
 * 各段の (要求した長辺, 縮小画像, 軸ごとの倍率) を順に出す。同じ画素数の段は 1 度だけ。
 *
 * 倍率は要求値ではなく**実際の画素数の比**。丸めで 1 画素ずれた分を無視すると、座標を戻すときに
 * 系統的な誤差が残る。
 */
function* rungsWithoutRepeats(photo: PhotoRgb, rungs: readonly number[]): Generator<Rung> {
  const seen = new Set<string>();
  for (const longSide of rungs) {
    const scaled = resamplePilToLongSide(photo, longSide, LANCZOS3);
    const size = `${scaled.width}x${scaled.height}`;
    if (seen.has(size)) continue;
    seen.add(size);
    yield {
      longSide,
      photo: scaled,
      scaleX: scaled.width / photo.width,
      scaleY: scaled.height / photo.height,
    };
  }
}

function rungNote(rung: Rung): string {
  return `長辺 ${rung.longSide}（${rung.photo.width}x${rung.photo.height}）`;
}

/** 縮小画像の座標を元画像へ戻す（原点をずらす場合は `originX/Y` を足す）。 */
function toSourceCoordinates(
  points: Float64Array,
  scaleX: number,
  scaleY: number,
  originX = 0,
  originY = 0,
): Float64Array {
  const out = new Float64Array(points.length);
  for (let point = 0; point < points.length / 2; point++) {
    out[point * 2] = points[point * 2] / scaleX + originX;
    out[point * 2 + 1] = points[point * 2 + 1] / scaleY + originY;
  }
  return out;
}

/** 二段検出の結果と、どの段で何が起きたか（失敗の切り分けに使う）。 */
export interface TwoPassResult {
  /** 主役 1 人ぶんの点（全体画像の画素座標）。 */
  readonly landmarks: Float64Array;
  readonly scoutNotes: readonly string[];
  readonly refineNotes: readonly string[];
}

/**
 * 1 回だけ検出する関数を二段検出に組み上げる。
 *
 * `detectFaces` を引数に取るのは、座標の往復（縮小・クロップ・戻し）と主役の同定を検出器なしで
 * 検証できるようにするため。この関数は検出器の中身を知らない。
 *
 * @throws FaceNotDetectedError 段1 が全滅した、または段2 が全滅した。**どちらで落ちたかと試した
 *   条件をメッセージに書く**（運用でここを読んで原因を切り分ける）
 */
export function detectTwoPass(input: {
  detectFaces: DetectFaces;
  photo: PhotoRgb;
  /** 外接正方形を測るのに読む点数（虹彩を含めない）。 */
  faceMeshCount: number;
  ladderFloor?: number;
  ladderStep?: number;
  cropSpanFactor?: number;
  /** 主役を測る対象点（画像画素座標）。既定は画像中心。 */
  subjectTarget?: readonly [number, number] | null;
}): TwoPassResult {
  const {
    detectFaces,
    photo,
    faceMeshCount,
    ladderFloor = SCOUT_LADDER_FLOOR,
    ladderStep = LADDER_STEP,
    cropSpanFactor = REFINE_CROP_SPAN_FACTOR,
    subjectTarget = null,
  } = input;
  const { width, height } = photo;
  const rungs = scaleLadder(Math.max(width, height), ladderFloor, ladderStep);

  // --- 段1（探索）: 全段回して顔を集める -----------------------------------
  const scoutNotes: string[] = [];
  const scouted: Float64Array[] = [];
  for (const rung of rungsWithoutRepeats(photo, rungs)) {
    const note = rungNote(rung);
    let faces: Float64Array[];
    try {
      faces = detectFaces(rung.photo);
    } catch (error) {
      if (!(error instanceof FaceNotDetectedError)) throw error;
      scoutNotes.push(`${note} 検出できず`);
      continue;
    }
    for (const face of faces) {
      scouted.push(toSourceCoordinates(face, rung.scaleX, rung.scaleY));
    }
    scoutNotes.push(`${note} ${faces.length} 件`);
  }
  if (scouted.length === 0) {
    throw new FaceNotDetectedError(
      '写真から顔を検出できませんでした。正面を向いた顔が大きく写っている写真を選んでください。' +
        `（元 ${width}x${height}、段1: ${scoutNotes.join(' / ')}）`,
    );
  }

  const subject = subjectFace(
    scouted.map((face) => faceSquareOfLandmarks(face, faceMeshCount)),
    [width, height],
    subjectTarget,
  );

  // --- 段2（精密）: 主役の周りを元解像度から切って再検出 --------------------
  const [left, top, right, bottom] = cropBoxAroundFace(subject, width, height, cropSpanFactor);
  const crop = cropPhotoRect(photo, left, top, right, bottom);
  // 大きい側から降りる。クロップは既に顔だけなので細部が多いほど良く、写真全体より小さいので
  // エイリアシングが問題にならない。
  const cropRungs = scaleLadder(
    Math.max(crop.width, crop.height),
    ladderFloor,
    ladderStep,
  ).reverse();

  const refineNotes: string[] = [];
  for (const rung of rungsWithoutRepeats(crop, cropRungs)) {
    const note = rungNote(rung);
    let faces: Float64Array[];
    try {
      faces = detectFaces(rung.photo);
    } catch (error) {
      if (!(error instanceof FaceNotDetectedError)) throw error;
      refineNotes.push(`${note} 検出できず`);
      continue;
    }
    const points = faces.map((face) =>
      toSourceCoordinates(face, rung.scaleX, rung.scaleY, left, top),
    );
    const squares = points.map((face) => faceSquareOfLandmarks(face, faceMeshCount));
    // **段1 で選んだ主役がどれかを毎回同定する。** クロップの隅に入った別人を掴んでも座標としては
    // 妥当な値で返るため、ここで見なければ後段は気付けない。
    const index = sameFaceIndex(squares, subject);
    if (index === null) {
      refineNotes.push(`${note} 主役と同定できる顔が無い（${squares.length} 件）`);
      continue;
    }
    refineNotes.push(`${note} 成功（${points.length} 件中 ${index + 1} 件目）`);
    return { landmarks: points[index], scoutNotes, refineNotes };
  }
  throw new FaceNotDetectedError(
    '顔の位置は掴めましたが、その周りを切り出しての再検出が通りませんでした。' +
      `（元 ${width}x${height}、クロップ ${right - left}x${bottom - top}、` +
      `段2: ${refineNotes.join(' / ')}）`,
  );
}
