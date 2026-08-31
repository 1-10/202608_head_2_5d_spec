// 眼球 UV ↔ 正面（GNM の XY 平面）の対応を、メッシュから測る。
//
// 写真を眼球の UV へ焼くには 2 つの写像が要る:
//
//     角度  UV の角度 ↔ 頭部の解剖学的な向き（上はどちらか、解剖学的左はどちらか）
//     半径  UV の半径 ↔ 正面から見た眼球上の半径（limbus を 1 とした比）
//
// **どちらも仮定しない。メッシュから測る。** `GnmHeadMesh` は眼球コンポーネントの頂点について
// 3D 位置（GNM 空間: X = 解剖学的左 / Y = 上 / Z = 前）と UV の両方を持っているので、対応は
// 実際に回帰できる。写しを定数で置くと、GNM のアセットが変わったときに黙って嘘になる（絵が
// 回るだけで例外にはならない）。
//
// 測った結果（v3_0 / head）
// -------------------------
// **角度**: 左右どちらも `UV の角度 = atan2(ΔY, ΔX) + offset`、**反転なし**（offset は
// ±0.04°、残差 RMS 0.32〜0.33°）。反転を仮定した側は残差 RMS が 103° でまったく当たらない。
// つまり **UV の角度 0 が解剖学的左、90° が上**で、左右で同じ。**鼻側 / 耳側は左右で入れ
// 替わる**ので、1 枚を共用できない理由がここにも出る。
//
// **半径**: 線形ではない。**線形に写すと瞳孔が 1.6 倍に膨らむ**（実物の瞳孔は limbus の
// 0.35〜0.5 倍だが、線形写像だと写真の 0.35 が指す 3D の投影半径は limbus の 0.55 倍になる）。
// 実測の profile を使う。
//
// **identity では動かない。** identity 基底の眼球への寄与を ±3σ で振っても profile と limbus
// 半径は 1 つも変わらない（実測: 最大差 0.00000）。眼球は剛体として扱われているので、平均形状
// から測って写真ごとに測り直さない。
//
// 眼球コンポーネントは 2 枚の殻を持つ
// ------------------------------------
// `left_eye` / `right_eye` の 795 頂点は**連結していない 2 枚の面**でできている:
//
//     眼球内殻   385 頂点  UV 半径 [0.0000, 0.4608]  pupils / irises / scleras
//     外殻       410 頂点  UV 半径 [0.0012, 0.6566]  どの色グループにも属さない（角膜殻）
//
// **測るのは内殻。** 内殻の選び方は **UV 中心（半径最小の頂点）を含む連結成分**で、選んだ成分の
// UV 半径の上限が `SCLERA_OUTER_RADIUS_UV` と一致することで照合する（別の性質で選んで別の性質で
// 確かめるので、選び方が壊れたら落ちる）。**UV の切れ目で頂点が複製されるため、外殻は split
// 空間では 2 つの成分に割れる**ので「成分がちょうど 2 つ」では検査できない。

import { Similarity2d } from '../gnm/fit';
import { GnmHeadMesh } from '../gnm/model';
import {
  EYE_COMPONENT_NAMES,
  EYE_SIDES,
  EYE_SIDE_LEFT,
  EYE_SIDE_RIGHT,
  EYE_UV_CENTER,
  EyeSide,
  IRIS_OUTER_RADIUS_UV,
  SCLERA_OUTER_RADIUS_UV,
} from './layout';

/**
 * 同じ環の頂点をまとめるときの UV 半径の隙間の閾値。
 *
 * 実測（内殻 385 頂点）: 同じ環の中の半径のばらつきは 5e-8 以下、隣の環との最小の隙間は
 * 0.00431。1e-4 は前者の 2000 倍・後者の 1/43 で、両者の間に広い余裕を持って入る。**閾値を
 * 観測された段差の縁ではなく間に置く**（縁に置くと、環の間隔が少し変わったときに 2 つの環が
 * 1 つに融ける）。
 */
export const RING_GAP_UV = 1e-4;

/** 角度の回帰から外す中心付近の UV 半径（最内の環 0.0246 より内側で、中心の 1 頂点だけを外す）。 */
export const ANGLE_FIT_MINIMUM_RADIUS_UV = 0.02;

/**
 * 角度の対応が「回転 + 反転の有無」で表せていると認める残差 RMS の上限。
 *
 * 実測は 0.32〜0.33°。上限を 5° に置くのは、**limbus の環が 24 点 = 15° 刻み**で並んでいるため。
 * 残差がその半分（7.5°）に近づくと、どの頂点がどの方位にあるのかが隣の頂点と入れ替わりうる。
 * 超えたら例外にする。**絵が静かに回るのではなく落ちるべき**。
 */
export const MAXIMUM_ANGLE_RESIDUAL_DEGREES = 5.0;

/** limbus の環に必要な頂点数（実測 24）。「4 方位を 2 重に覆える最小」が 8。 */
export const MINIMUM_LIMBUS_RING_POINTS = 8;

/**
 * 眼球の UV レイアウトが前提と違う（角度が回転 + 反転で表せない等）。
 *
 * 写真ではなくアセットの側の問題なので、写真を選び直しても直らない。
 */
export class EyeUvLayoutError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'EyeUvLayoutError';
  }
}

/** 片側の眼球について、UV ↔ 正面 の対応を測った結果。 */
export interface EyeUvGeometry {
  readonly side: EyeSide;
  /** 内殻の環の UV 半径（昇順）。 */
  readonly ringRadiiUv: Float64Array;
  /** 各環の正面投影半径 / limbus の投影半径。 */
  readonly ringLimbusFractions: Float64Array;
  /** UV の角度が解剖学的な角度に対して反転しているか。 */
  readonly angleFlipped: boolean;
  /** `UV角度 = ±解剖学的角度 + offset` の offset。 */
  readonly angleOffsetRadians: number;
  /** 上の式の残差 RMS（当てはまりの実測値）。 */
  readonly angleResidualRmsRadians: number;
  /** limbus の環の重心（GNM 空間の XY、メートル）。 */
  readonly centerXy: readonly [number, number];
  /** limbus の環の半径（GNM 空間、メートル）。 */
  readonly limbusRadiusMeters: number;
  /** 内殻の頂点数。 */
  readonly vertexCount: number;
  /** 投影半径比が最大になる環の index（= シルエット）。 */
  readonly silhouetteIndex: number;
  /** これより外の UV は写真のどの画素も指さない（面が裏側へ回る）。 */
  readonly silhouetteRadiusUv: number;
}

/** 角度の当てはまりを人が読める単位で（報告と検査に使う）。 */
export function angleResidualRmsDegrees(geometry: EyeUvGeometry): number {
  return (geometry.angleResidualRmsRadians * 180) / Math.PI;
}

/**
 * UV 半径 → 正面投影半径 / limbus 半径。シルエットより外は NaN。
 *
 * 環と環の間は線形補間する。環の間隔（実測 0.004〜0.049 UV）は焼くテクスチャのテクセル間隔より
 * 粗いので、補間なしでは階段が出る。曲線を当てはめずに線形にするのは、環の間の形を知らない
 * ため（知らないものを滑らかに見せると、測った値と作った値の区別が絵から消える）。
 */
export function limbusFraction(geometry: EyeUvGeometry, radiusUv: number): number {
  if (radiusUv > geometry.silhouetteRadiusUv) return Number.NaN;
  const end = geometry.silhouetteIndex + 1;
  const radii = geometry.ringRadiiUv;
  const fractions = geometry.ringLimbusFractions;
  if (radiusUv <= radii[0]) return fractions[0];
  for (let ring = 1; ring < end; ring++) {
    if (radiusUv <= radii[ring]) {
      const t = (radiusUv - radii[ring - 1]) / (radii[ring] - radii[ring - 1]);
      return fractions[ring - 1] + (fractions[ring] - fractions[ring - 1]) * t;
    }
  }
  return fractions[end - 1];
}

/**
 * UV の角度 → 解剖学的な角度（GNM 正面の `atan2(ΔY, ΔX)`）。
 *
 * 焼くときはテクセルから写真を引く（gather）ので、要るのはこの向き。
 */
export function anatomicalAngleOf(geometry: EyeUvGeometry, uvAngle: number): number {
  const angle = uvAngle - geometry.angleOffsetRadians;
  return geometry.angleFlipped ? -angle : angle;
}

/** 解剖学的な角度 → UV の角度（`anatomicalAngleOf` の逆）。 */
export function uvAngleOf(geometry: EyeUvGeometry, anatomicalAngle: number): number {
  return (geometry.angleFlipped ? -anatomicalAngle : anatomicalAngle) + geometry.angleOffsetRadians;
}

/** 片側の眼球の UV ↔ 正面 の対応をメッシュから測る。 */
export function eyeUvGeometry(mesh: GnmHeadMesh, side: EyeSide): EyeUvGeometry {
  const component = componentMask(mesh, side);
  const interior = interiorShell(mesh, component);

  const indices: number[] = [];
  for (let vertex = 0; vertex < mesh.vertexCount; vertex++) if (interior[vertex] !== 0) indices.push(vertex);

  const radiusUv = new Float64Array(indices.length);
  const uvAngle = new Float64Array(indices.length);
  const positions = new Float64Array(indices.length * 3);
  indices.forEach((vertex, slot) => {
    const du = mesh.vertexUvs[vertex * 2] - EYE_UV_CENTER;
    const dv = mesh.vertexUvs[vertex * 2 + 1] - EYE_UV_CENTER;
    radiusUv[slot] = Math.hypot(du, dv);
    uvAngle[slot] = Math.atan2(dv, du);
    positions[slot * 3] = mesh.templateVertexPositions[vertex * 3];
    positions[slot * 3 + 1] = mesh.templateVertexPositions[vertex * 3 + 1];
    positions[slot * 3 + 2] = mesh.templateVertexPositions[vertex * 3 + 2];
  });

  const rings = ringGroups(radiusUv);
  const { center, radius: limbusRadius } = limbusCircle(positions, radiusUv, rings);

  const anatomicalAngle = new Float64Array(indices.length);
  const usable = new Uint8Array(indices.length);
  for (let slot = 0; slot < indices.length; slot++) {
    anatomicalAngle[slot] = Math.atan2(
      positions[slot * 3 + 1] - center[1],
      positions[slot * 3] - center[0],
    );
    usable[slot] = radiusUv[slot] >= ANGLE_FIT_MINIMUM_RADIUS_UV ? 1 : 0;
  }
  const { offset, flipped, residualRms } = fitAngle(uvAngle, anatomicalAngle, usable, side);

  const ringRadiiUv = new Float64Array(rings.length);
  const ringLimbusFractions = new Float64Array(rings.length);
  rings.forEach((ring, index) => {
    let radiusTotal = 0;
    let projectedTotal = 0;
    for (const slot of ring) {
      radiusTotal += radiusUv[slot];
      projectedTotal += Math.hypot(
        positions[slot * 3] - center[0],
        positions[slot * 3 + 1] - center[1],
      );
    }
    ringRadiiUv[index] = radiusTotal / ring.length;
    ringLimbusFractions[index] = projectedTotal / ring.length / limbusRadius;
  });

  let silhouetteIndex = 0;
  for (let ring = 1; ring < ringLimbusFractions.length; ring++) {
    if (ringLimbusFractions[ring] > ringLimbusFractions[silhouetteIndex]) silhouetteIndex = ring;
  }

  if (!(limbusRadius > 0)) throw new Error(`limbus 半径が正でない: ${limbusRadius}`);
  return {
    side,
    ringRadiiUv,
    ringLimbusFractions,
    angleFlipped: flipped,
    angleOffsetRadians: offset,
    angleResidualRmsRadians: residualRms,
    centerXy: center,
    limbusRadiusMeters: limbusRadius,
    vertexCount: indices.length,
    silhouetteIndex,
    silhouetteRadiusUv: ringRadiiUv[silhouetteIndex],
  };
}

/** 両側の対応をまとめて測る。 */
export function eyeUvGeometries(mesh: GnmHeadMesh): Record<EyeSide, EyeUvGeometry> {
  return {
    left: eyeUvGeometry(mesh, EYE_SIDE_LEFT),
    right: eyeUvGeometry(mesh, EYE_SIDE_RIGHT),
  };
}

/**
 * 写真テクスチャを貼る眼球内殻の頂点maskを返す。
 *
 * 外側の角膜殻を不透明な写真テクスチャで描くと内殻を完全に隠すため、3Dデバッグビューの
 * サブメッシュ選別にも、UV幾何の測定と同じ正本を使う。
 */
export function eyeInteriorVertexMask(mesh: GnmHeadMesh, side: EyeSide): Uint8Array {
  return interiorShell(mesh, componentMask(mesh, side));
}

/**
 * 写真の目を解剖学的な側へ割り当てる。
 *
 * **画像上の左右では決めない。** 写真が鏡像かどうかで見た目の左右は入れ替わるので、見た目で
 * 決めた側は鏡像の写真でそのまま反転する。代わりに `left_eye` / `right_eye` コンポーネントの
 * 重心を `similarity` で写真へ射影し、どちらの虹彩中心に近いかで決める。`similarity` は 68 点
 * から独立にフィットされているので、眼球の割り当てを眼球の情報から導く循環にはならない。
 *
 * **2 つの群を独立に「近い方」へ入れない。** 両方が同じ側に入りうる（顔が横を向いていると
 * 射影がずれる）。2 通りの割り当てのうち距離の和が小さい方を採るので、必ず別々の側に入る。
 */
export function assignEyeSides(
  mesh: GnmHeadMesh,
  similarity: Similarity2d,
  photoIrisCenters: Record<string, readonly [number, number]>,
): Record<string, EyeSide> {
  const { names, straight, swapped } = sideAssignmentCosts(mesh, similarity, photoIrisCenters);
  const chosen: [EyeSide, EyeSide] =
    straight <= swapped ? [EYE_SIDE_LEFT, EYE_SIDE_RIGHT] : [EYE_SIDE_RIGHT, EYE_SIDE_LEFT];
  return { [names[0]]: chosen[0], [names[1]]: chosen[1] };
}

/**
 * `assignEyeSides` が選ばなかった割り当てとの距離の和の差（写真ピクセル）。
 *
 * 0 に近いほど「どちらでもよかった」ことになる。検査で余裕を数値で見るために分けてある。
 */
export function sideAssignmentMargin(
  mesh: GnmHeadMesh,
  similarity: Similarity2d,
  photoIrisCenters: Record<string, readonly [number, number]>,
): number {
  const { straight, swapped } = sideAssignmentCosts(mesh, similarity, photoIrisCenters);
  return Math.abs(straight - swapped);
}

function sideAssignmentCosts(
  mesh: GnmHeadMesh,
  similarity: Similarity2d,
  photoIrisCenters: Record<string, readonly [number, number]>,
): { names: string[]; straight: number; swapped: number } {
  const names = Object.keys(photoIrisCenters);
  if (names.length !== 2) throw new Error(`目の群が 2 つでない: ${names.sort()}`);

  const projected: Record<EyeSide, [number, number]> = { left: [0, 0], right: [0, 0] };
  for (const side of EYE_SIDES) {
    const mask = componentMask(mesh, side);
    let count = 0;
    let totalX = 0;
    let totalY = 0;
    for (let vertex = 0; vertex < mesh.vertexCount; vertex++) {
      if (mask[vertex] === 0) continue;
      count++;
      totalX += mesh.templateVertexPositions[vertex * 3];
      totalY += mesh.templateVertexPositions[vertex * 3 + 1];
    }
    projected[side] = similarity.applyPoint(totalX / count, totalY / count);
  }

  const totalDistance = (sides: readonly [EyeSide, EyeSide]): number =>
    names.reduce((total, name, slot) => {
      const center = photoIrisCenters[name];
      const target = projected[sides[slot]];
      return total + Math.hypot(center[0] - target[0], center[1] - target[1]);
    }, 0);

  return {
    names,
    straight: totalDistance([EYE_SIDE_LEFT, EYE_SIDE_RIGHT]),
    swapped: totalDistance([EYE_SIDE_RIGHT, EYE_SIDE_LEFT]),
  };
}

// ---------------------------------------------------------------------------
// 測定の中身
// ---------------------------------------------------------------------------
/**
 * 側 → その眼球コンポーネントの頂点マスク。
 *
 * GNM の名前が解剖学的な左右と食い違っていないことを重心の X の符号で確かめる（GNM 空間は
 * X の正が解剖学的左）。名前だけを信じると、アセットの側で左右を入れ替えられたときに
 * キャッチライトが左右逆に焼かれて、絵からは気付けない。
 */
function componentMask(mesh: GnmHeadMesh, side: EyeSide): Uint8Array {
  const name = EYE_COMPONENT_NAMES[side];
  const componentIndex = mesh.componentNames.indexOf(name);
  if (componentIndex < 0) {
    throw new EyeUvLayoutError(
      `メッシュに ${name} が無い（あるのは ${mesh.componentNames.join(', ')}）`,
    );
  }
  const mask = new Uint8Array(mesh.vertexCount);
  let count = 0;
  let totalX = 0;
  for (let vertex = 0; vertex < mesh.vertexCount; vertex++) {
    if (mesh.componentId[vertex] !== componentIndex) continue;
    mask[vertex] = 1;
    count++;
    totalX += mesh.templateVertexPositions[vertex * 3];
  }
  if (count === 0) throw new EyeUvLayoutError(`${name} の頂点が 1 つも無い`);
  const centroidX = totalX / count;
  if (centroidX > 0 !== (side === EYE_SIDE_LEFT)) {
    throw new EyeUvLayoutError(
      `${name} の重心 X が ${centroidX.toFixed(4)} で、解剖学的な ${side} と合わない` +
        '（GNM 空間は X の正が解剖学的左）',
    );
  }
  return mask;
}

/** 眼球コンポーネントのうち**眼球内殻**の頂点マスクを返す。 */
function interiorShell(mesh: GnmHeadMesh, component: Uint8Array): Uint8Array {
  let centerVertex = -1;
  let smallestRadius = Infinity;
  for (let vertex = 0; vertex < mesh.vertexCount; vertex++) {
    if (component[vertex] === 0) continue;
    const radius = Math.hypot(
      mesh.vertexUvs[vertex * 2] - EYE_UV_CENTER,
      mesh.vertexUvs[vertex * 2 + 1] - EYE_UV_CENTER,
    );
    if (radius < smallestRadius) {
      smallestRadius = radius;
      centerVertex = vertex;
    }
  }
  if (centerVertex < 0) throw new EyeUvLayoutError('眼球コンポーネントが空');

  const labels = connectedComponents(mesh.triangles, component, mesh.vertexCount);
  const shell = new Uint8Array(mesh.vertexCount);
  let outer = 0;
  let count = 0;
  for (let vertex = 0; vertex < mesh.vertexCount; vertex++) {
    if (component[vertex] === 0 || labels[vertex] !== labels[centerVertex]) continue;
    shell[vertex] = 1;
    count++;
    const radius = Math.hypot(
      mesh.vertexUvs[vertex * 2] - EYE_UV_CENTER,
      mesh.vertexUvs[vertex * 2 + 1] - EYE_UV_CENTER,
    );
    if (radius > outer) outer = radius;
  }
  if (Math.abs(outer - SCLERA_OUTER_RADIUS_UV) > 1e-3 * SCLERA_OUTER_RADIUS_UV) {
    throw new EyeUvLayoutError(
      `眼球内殻の UV 半径の上限が ${outer.toFixed(6)} で、強膜の外径` +
        ` ${SCLERA_OUTER_RADIUS_UV} と一致しない（${count} 頂点）。` +
        '内殻の選び方かレイアウトの実測値が合っていない',
    );
  }
  return shell;
}

/**
 * `keep` の頂点だけで三角形をたどり、連結成分のラベルを返す。
 *
 * ラベルは成分の中の最小の頂点 index。眼球は 1500 三角形しか無いので素朴な union-find で十分。
 */
function connectedComponents(
  triangles: Uint32Array,
  keep: Uint8Array,
  vertexCount: number,
): Int32Array {
  const parent = new Int32Array(vertexCount);
  for (let vertex = 0; vertex < vertexCount; vertex++) parent[vertex] = vertex;
  const find = (vertex: number): number => {
    let root = vertex;
    while (parent[root] !== root) root = parent[root];
    let walk = vertex;
    while (parent[walk] !== root) {
      const next = parent[walk];
      parent[walk] = root;
      walk = next;
    }
    return root;
  };
  const union = (first: number, second: number): void => {
    const rootFirst = find(first);
    const rootSecond = find(second);
    if (rootFirst === rootSecond) return;
    if (rootFirst < rootSecond) parent[rootSecond] = rootFirst;
    else parent[rootFirst] = rootSecond;
  };
  for (let triangle = 0; triangle < triangles.length / 3; triangle++) {
    const a = triangles[triangle * 3];
    const b = triangles[triangle * 3 + 1];
    const c = triangles[triangle * 3 + 2];
    if (keep[a] === 0 || keep[b] === 0 || keep[c] === 0) continue;
    union(a, b);
    union(b, c);
  }
  const labels = new Int32Array(vertexCount);
  for (let vertex = 0; vertex < vertexCount; vertex++) labels[vertex] = find(vertex);
  return labels;
}

/** UV 半径が同じ頂点をまとめた環の slot を、半径の昇順で返す。 */
function ringGroups(radiusUv: Float64Array): number[][] {
  const order = [...radiusUv.keys()].sort((first, second) => radiusUv[first] - radiusUv[second]);
  const rings: number[][] = [];
  let current: number[] = [];
  for (let index = 0; index < order.length; index++) {
    if (index > 0 && radiusUv[order[index]] - radiusUv[order[index - 1]] > RING_GAP_UV) {
      rings.push(current);
      current = [];
    }
    current.push(order[index]);
  }
  if (current.length > 0) rings.push(current);
  return rings;
}

/**
 * limbus の環（UV 半径が `IRIS_OUTER_RADIUS_UV` の環）の重心と半径。
 *
 * 重心が**写真の虹彩中心に対応する点**で、半径が**写真の虹彩半径に対応する長さ**。眼球中心を
 * 球フィットで出さずに limbus の環そのものから取るのは、写真側で測れるのが limbus の円だから
 * （MediaPipe の虹彩 5 点）。合わせる相手と同じものを測る。
 */
function limbusCircle(
  positions: Float64Array,
  radiusUv: Float64Array,
  rings: number[][],
): { center: [number, number]; radius: number } {
  let bestRing = 0;
  let bestDistance = Infinity;
  rings.forEach((ring, index) => {
    const mean = ring.reduce((total, slot) => total + radiusUv[slot], 0) / ring.length;
    const distance = Math.abs(mean - IRIS_OUTER_RADIUS_UV);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestRing = index;
    }
  });
  const limbus = rings[bestRing];
  if (limbus.length < MINIMUM_LIMBUS_RING_POINTS) {
    throw new EyeUvLayoutError(
      `limbus の環が ${limbus.length} 点しかない` +
        `（${MINIMUM_LIMBUS_RING_POINTS} 点必要）。環の検出が壊れている`,
    );
  }
  let totalX = 0;
  let totalY = 0;
  for (const slot of limbus) {
    totalX += positions[slot * 3];
    totalY += positions[slot * 3 + 1];
  }
  const center: [number, number] = [totalX / limbus.length, totalY / limbus.length];
  let totalRadius = 0;
  for (const slot of limbus) {
    totalRadius += Math.hypot(positions[slot * 3] - center[0], positions[slot * 3 + 1] - center[1]);
  }
  return { center, radius: totalRadius / limbus.length };
}

/**
 * `UV角度 = ±解剖学的角度 + offset` を当てはめて (offset, 反転, 残差RMS) を返す。
 *
 * 符号は 2 通りしかないので両方試して残差の小さい方を採る。角度の平均と残差は複素数の単位
 * ベクトルの平均で取る（角度をそのまま平均すると ±π の境目で壊れる）。
 */
function fitAngle(
  uvAngle: Float64Array,
  anatomicalAngle: Float64Array,
  usable: Uint8Array,
  side: EyeSide,
): { offset: number; flipped: boolean; residualRms: number } {
  const results: { residualRms: number; offset: number; flipped: boolean }[] = [];
  for (const flipped of [false, true]) {
    const sign = flipped ? -1 : 1;
    let sumReal = 0;
    let sumImaginary = 0;
    let count = 0;
    for (let slot = 0; slot < uvAngle.length; slot++) {
      if (usable[slot] === 0) continue;
      const difference = uvAngle[slot] - sign * anatomicalAngle[slot];
      sumReal += Math.cos(difference);
      sumImaginary += Math.sin(difference);
      count++;
    }
    const offset = Math.atan2(sumImaginary / count, sumReal / count);
    let squared = 0;
    for (let slot = 0; slot < uvAngle.length; slot++) {
      if (usable[slot] === 0) continue;
      const difference = uvAngle[slot] - sign * anatomicalAngle[slot] - offset;
      const residual = Math.atan2(Math.sin(difference), Math.cos(difference));
      squared += residual * residual;
    }
    results.push({ residualRms: Math.sqrt(squared / count), offset, flipped });
  }
  results.sort((first, second) => first.residualRms - second.residualRms);
  const best = results[0];
  if ((best.residualRms * 180) / Math.PI > MAXIMUM_ANGLE_RESIDUAL_DEGREES) {
    const detail = results
      .map(
        (result) =>
          `${result.flipped ? '反転' : '正転'} 残差 ${((result.residualRms * 180) / Math.PI).toFixed(2)}°`,
      )
      .join(', ');
    throw new EyeUvLayoutError(
      `${side} の眼球 UV の角度が「回転 + 反転の有無」で表せない（${detail}、` +
        `上限 ${MAXIMUM_ANGLE_RESIDUAL_DEGREES}°）。同心円レイアウトという前提そのものが崩れている`,
    );
  }
  return best;
}
