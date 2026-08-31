// 3D ビューの法線。**正本は Unity 側の `Scripts/GnmNormals` と `GnmHeadInstance.FlattenNormals`。**
//
// ## 重みは内角（面積ではない）
//
// 面積重み（非正規化 cross をそのまま足す形）は合算長がモデル寸法と三角形密度に比例する。メートル
// 単位の 0.25m の頭部では、まぶた・鼻孔・歯・眼球のような密な領域で合算長が 1e-5 前後まで落ちて、
// 正規化がゼロベクトルへ潰れる。**ゼロ法線は面を真っ黒にする**（照明モデルのせいだと誤診しやすい）。
// 内角重み（Thürmer & Wüthrich）は無次元なので合算長が常に 2π 前後で、寸法にも三角形密度にも依存
// しない。密度差の大きい GNM のトポロジ向き。
//
// ## 複製前のトポロジで集約する
//
// UV の切れ目で複製した頂点をそのまま別物として扱うと、シーム上に陰影の継ぎ目が出る。
// `uvSplitSource` の指す元頂点へ集約してから複製先へ配り直す。
//
// ## +Z 固定にするのは写真を貼る領域だけ
//
// exporter は顔の立体感を**写真に写った陰影**として焼く。実法線でライティングすると写真の影と
// 陰影が二重に掛かるので、肌・眼球・髪の法線は +Z 固定に落とす。
//
// **口腔内（`flat_color` の領域）は実法線を残す。** 写真を持たず単色なので二重に掛かる影が無く、
// 逆に +Z 固定だと開口時に歯・歯茎・舌が真っ平らな切り絵に見える。ただし口腔壁は `skin` の部分集合で
// 肌と頂点を共有するため、**共有頂点は肌側を優先して +Z へ落とす**（唇の内縁だけ実法線が残ると、
// 開口時にそこへ筋状の陰影が出る）。
//
// ## 法線は「スキニングの前」に作ってから回す
//
// +Z 固定もメッシュ空間の値として持ち、位置と同じ LBS の回転を掛ける。シェーダ側で定数に置き換える
// 形では代替できない — 首を回しても陰影が動かなくなる（Unity 側も Mesh に焼いて skinning へ通している）。

// ## 実法線を作るのは口腔内に触れる三角形だけ
//
// +Z へ落とす頂点の実法線は**捨てる**ので、計算する理由が無い。全 35,324 三角形を毎フレーム回すと
// 実測 5.7ms かかり、60fps の予算 16.7ms のうち 3 分の 1 を捨てることになる。実法線を残す頂点に
// 接する三角形だけへ絞る（`planNormals` が 1 回だけ数える）。**結果は全部回した場合と同一** —
// 残す頂点の法線はその頂点に接する三角形だけで決まり、落とす頂点は上書きされるため。

import { GnmPreviewAsset } from './asset';

/** 写真を貼る領域の法線。メッシュ空間で +Z。 */
export const FLAT_NORMAL: readonly [number, number, number] = [0, 0, 1];

/**
 * どの三角形と頂点を実法線の計算に回すか。シーンごとに 1 回作る。
 *
 * `keepReal` が 0 の頂点は `flattenNormals` が +Z で上書きするので、そこへ寄与するだけの三角形を
 * 回しても結果に出ない。`triangles` は**実法線を残す頂点に 1 つでも触れる三角形**（領域は問わない
 * — 口腔壁の奥の頂点は境界で肌の三角形にも接する）、`vertices` はその三角形が使う split 頂点。
 */
export interface NormalPlan {
  /** (split 頂点数) 1 = 実法線 / 0 = +Z 固定。 */
  readonly keepReal: Uint8Array;
  readonly triangles: Uint32Array;
  readonly vertices: Uint32Array;
  /** 複製前の頂点数（ここへ集約してから複製先へ配る）。 */
  readonly weldedCount: number;
}

/** `keepReal` から計算対象を数える。 */
export function planNormals(
  triangles: Uint32Array,
  keepReal: Uint8Array,
  weldedCount: number,
): NormalPlan {
  const picked: number[] = [];
  const vertices = new Set<number>();
  const triangleCount = triangles.length / 3;
  for (let triangle = 0; triangle < triangleCount; triangle++) {
    const i0 = triangles[triangle * 3];
    const i1 = triangles[triangle * 3 + 1];
    const i2 = triangles[triangle * 3 + 2];
    if (keepReal[i0] === 0 && keepReal[i1] === 0 && keepReal[i2] === 0) continue;
    picked.push(triangle);
    vertices.add(i0);
    vertices.add(i1);
    vertices.add(i2);
  }
  return {
    keepReal,
    triangles: Uint32Array.from(picked),
    vertices: Uint32Array.from([...vertices].sort((first, second) => first - second)),
    weldedCount,
  };
}

/**
 * 内角重みの頂点法線を `out` へ書く（`plan` が挙げた頂点だけ）。
 *
 * **`out` の残りには触れない。** 呼び側は `flattenNormals` で +Z を書き込む前提。
 *
 * @param positions (split 頂点数, 3)
 * @param scratch (weldedCount, 3) の作業領域。渡すと確保しない
 * @returns 向きを決められなかった分割頂点の数（0 でなければ接する三角形が全部退化している）
 */
export function recalculateNormals(
  positions: Float64Array,
  triangles: Uint32Array,
  uvSplitSource: Uint32Array,
  plan: NormalPlan,
  out: Float32Array,
  scratch?: Float64Array,
): number {
  const vertexCount = uvSplitSource.length;
  const weldedCount = plan.weldedCount;
  if (positions.length !== vertexCount * 3 || out.length !== vertexCount * 3) {
    throw new Error(
      `positions(${positions.length / 3}) / out(${out.length / 3}) が` +
        ` split 頂点数 ${vertexCount} と合わない`,
    );
  }
  const accumulated =
    scratch !== undefined && scratch.length >= weldedCount * 3
      ? scratch
      : new Float64Array(weldedCount * 3);
  for (const vertex of plan.vertices) {
    const slot = uvSplitSource[vertex] * 3;
    accumulated[slot] = 0;
    accumulated[slot + 1] = 0;
    accumulated[slot + 2] = 0;
  }

  for (const triangle of plan.triangles) {
    const i0 = triangles[triangle * 3];
    const i1 = triangles[triangle * 3 + 1];
    const i2 = triangles[triangle * 3 + 2];
    const x0 = positions[i0 * 3];
    const y0 = positions[i0 * 3 + 1];
    const z0 = positions[i0 * 3 + 2];
    const x1 = positions[i1 * 3];
    const y1 = positions[i1 * 3 + 1];
    const z1 = positions[i1 * 3 + 2];
    const x2 = positions[i2 * 3];
    const y2 = positions[i2 * 3 + 1];
    const z2 = positions[i2 * 3 + 2];

    const e01x = x1 - x0;
    const e01y = y1 - y0;
    const e01z = z1 - z0;
    const e02x = x2 - x0;
    const e02y = y2 - y0;
    const e02z = z2 - z0;

    // GNM の巻き順ではこの順序が外向き。長さは 2 × 面積。
    const cx = e01y * e02z - e01z * e02y;
    const cy = e01z * e02x - e01x * e02z;
    const cz = e01x * e02y - e01y * e02x;
    const twiceArea = Math.hypot(cx, cy, cz);
    if (twiceArea <= 0) continue; // 退化三角形は向きを持たないので寄与させない

    const ux = cx / twiceArea;
    const uy = cy / twiceArea;
    const uz = cz / twiceArea;

    // |u × w| はどのコーナーでも 2 × 面積で等しいので、内角は dot だけで出せる。
    // 内角の和は π なので atan2 は 1 三角形あたり 2 回で済む。
    const angle0 = Math.atan2(twiceArea, e01x * e02x + e01y * e02y + e01z * e02z);
    const angle1 = Math.atan2(
      twiceArea,
      (x2 - x1) * (x0 - x1) + (y2 - y1) * (y0 - y1) + (z2 - z1) * (z0 - z1),
    );
    const angle2 = Math.PI - angle0 - angle1;

    for (const [vertex, angle] of [
      [i0, angle0],
      [i1, angle1],
      [i2, angle2],
    ] as const) {
      const slot = uvSplitSource[vertex] * 3;
      accumulated[slot] += ux * angle;
      accumulated[slot + 1] += uy * angle;
      accumulated[slot + 2] += uz * angle;
    }
  }

  // 合算長は 2π 前後になるので、寸法に依存する閾値は要らない。
  for (const vertex of plan.vertices) {
    const slot = uvSplitSource[vertex] * 3;
    const x = accumulated[slot];
    const y = accumulated[slot + 1];
    const z = accumulated[slot + 2];
    const length = Math.hypot(x, y, z);
    if (length <= 0) continue;
    accumulated[slot] = x / length;
    accumulated[slot + 1] = y / length;
    accumulated[slot + 2] = z / length;
  }

  // 判定は分割頂点側で行う。除外領域（角膜）の元頂点は面を 1 枚も持たず 0 のままだが、どの分割頂点
  // からも参照されないので、元頂点側で数えると誤検知になる。
  let undetermined = 0;
  for (const vertex of plan.vertices) {
    const slot = uvSplitSource[vertex] * 3;
    const x = accumulated[slot];
    const y = accumulated[slot + 1];
    const z = accumulated[slot + 2];
    if (x === 0 && y === 0 && z === 0) {
      out[vertex * 3] = FLAT_NORMAL[0];
      out[vertex * 3 + 1] = FLAT_NORMAL[1];
      out[vertex * 3 + 2] = FLAT_NORMAL[2];
      undetermined++;
      continue;
    }
    out[vertex * 3] = x;
    out[vertex * 3 + 1] = y;
    out[vertex * 3 + 2] = z;
  }
  return undetermined;
}

/** `keepReal` が 0 の頂点を +Z へ落とす。 */
export function flattenNormals(normals: Float32Array, keepReal: Uint8Array): void {
  if (normals.length !== keepReal.length * 3) {
    throw new Error(`normals(${normals.length / 3}) と keepReal(${keepReal.length}) の長さが合わない`);
  }
  for (let vertex = 0; vertex < keepReal.length; vertex++) {
    if (keepReal[vertex] !== 0) continue;
    normals[vertex * 3] = FLAT_NORMAL[0];
    normals[vertex * 3 + 1] = FLAT_NORMAL[1];
    normals[vertex * 3 + 2] = FLAT_NORMAL[2];
  }
}

/**
 * 法線に LBS の回転だけを掛けて `out` へ書く（長さは正規化し直す）。
 *
 * 平行移動は掛けない。重みで混ぜた回転は直交行列にならないので、掛けた後に正規化が要る。
 */
export function skinNormals(
  preview: GnmPreviewAsset,
  normals: Float32Array,
  skinMatrices: Float64Array,
  out: Float32Array,
): void {
  const { vertexCount } = preview;
  if (normals.length !== vertexCount * 3 || out.length !== vertexCount * 3) {
    throw new Error('法線の配列の長さが頂点数と合わない');
  }
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    const x = normals[vertex * 3];
    const y = normals[vertex * 3 + 1];
    const z = normals[vertex * 3 + 2];
    let outX = 0;
    let outY = 0;
    let outZ = 0;
    for (let slot = 0; slot < 2; slot++) {
      const weight = preview.skinJointWeights[vertex * 2 + slot];
      if (weight === 0) continue;
      const base = preview.skinJointIndices[vertex * 2 + slot] * 12;
      outX += weight * (skinMatrices[base] * x + skinMatrices[base + 1] * y + skinMatrices[base + 2] * z);
      outY +=
        weight * (skinMatrices[base + 3] * x + skinMatrices[base + 4] * y + skinMatrices[base + 5] * z);
      outZ +=
        weight * (skinMatrices[base + 6] * x + skinMatrices[base + 7] * y + skinMatrices[base + 8] * z);
    }
    const length = Math.hypot(outX, outY, outZ);
    if (length <= 0) {
      out[vertex * 3] = FLAT_NORMAL[0];
      out[vertex * 3 + 1] = FLAT_NORMAL[1];
      out[vertex * 3 + 2] = FLAT_NORMAL[2];
      continue;
    }
    out[vertex * 3] = outX / length;
    out[vertex * 3 + 1] = outY / length;
    out[vertex * 3 + 2] = outZ / length;
  }
}

/** 12 要素の剛体変換の回転だけを法線へ当てる（髪シェル用。長さは変わらない）。 */
export function rotateNormals(normals: Float32Array, transform: Float64Array): Float32Array {
  const out = new Float32Array(normals.length);
  for (let vertex = 0; vertex < normals.length / 3; vertex++) {
    const x = normals[vertex * 3];
    const y = normals[vertex * 3 + 1];
    const z = normals[vertex * 3 + 2];
    out[vertex * 3] = transform[0] * x + transform[1] * y + transform[2] * z;
    out[vertex * 3 + 1] = transform[3] * x + transform[4] * y + transform[5] * z;
    out[vertex * 3 + 2] = transform[6] * x + transform[7] * y + transform[8] * z;
  }
  return out;
}
