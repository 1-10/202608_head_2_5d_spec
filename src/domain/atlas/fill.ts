// アトラスの埋まっていない領域の埋め方。
//
// 段1（写真）で埋まらなかったテクセルを作る。色の往復と標本化そのものは `domain/photo` が
// 持つ。写真色を3D曲面上の最寄り有効色へclampし、screened harmonicで馴染ませる。
//
// **後頭部を埋める平均色はアトラスの半分以上を占める**ので、リニア光で平均を取ることがそのまま
// 見た目に出る（sRGB のまま平均すると暗い側へ寄る）。

/** 調和項に対するowner事前値の強さ。旧harmonic実装と同じ値。 */
export const DEFAULT_SURFACE_HARMONIC_SCREENING = 0.1;
export const MINIMUM_SURFACE_HARMONIC_SCREENING = 0.0;
export const MAXIMUM_SURFACE_HARMONIC_SCREENING = 2.0;

/** screened harmonic共役勾配法の停止条件。 */
export const SURFACE_HARMONIC_TOLERANCE = 1e-6;
export const SURFACE_HARMONIC_MAXIMUM_ITERATIONS = 256;

/** 取得元1点の偶発色を抑えるため、その周囲で参照する観測済み頂点数。 */
export const SURFACE_SOURCE_COLOR_NEIGHBORS = 8;

/**
 * 表面距離で写真平面(x,y)の移動を奥行き(z)より重くする倍率。
 *
 * 正面写真から見えない裏側へ色を運ぶときは、同じ写真位置から奥へ回り込む経路を優先する。
 * 平方距離の倍率なので9は、x/y方向の移動1mをz方向の3m相当として測る。
 */
export const SURFACE_PHOTO_PLANE_DISTANCE_SCALE = 9.0;

/**
 * 隣接面の折れを補完経路へ加える倍率。
 *
 * 辺を挟む面の法線dotを `d` として `1 + 16 * (1 - d) ** 2` を距離へ掛ける。滑らかな頭皮では
 * ほぼ1のまま、60度の折れで5倍、90度で17倍になる。
 */
export const SURFACE_DIHEDRAL_PENALTY_SCALE = 16.0;

/** GNM解剖領域をまたぐ補完経路の距離倍率。到達不能にはせず、同領域を優先する。 */
export const SURFACE_REGION_CROSSING_SCALE = 10_000.0;

/** mask のテクセルの平均色 (3,)。1 つも無ければ中間グレー。 */
export function meanColor(color: Float32Array, mask: Uint8Array): Float32Array {
  let count = 0;
  const total = [0, 0, 0];
  for (let texel = 0; texel < mask.length; texel++) {
    if (mask[texel] === 0) continue;
    count++;
    total[0] += color[texel * 3];
    total[1] += color[texel * 3 + 1];
    total[2] += color[texel * 3 + 2];
  }
  if (count === 0) return new Float32Array([0.5, 0.5, 0.5]);
  return new Float32Array([total[0] / count, total[1] / count, total[2] / count]);
}

/**
 * 埋まっているテクセルの色を、region の中へ 1 テクセルずつ広げる（in-place）。
 *
 * 各反復で、埋まっている 8 近傍の平均色を取る。純粋な最近傍コピーより滑らかで、斜め方向の
 * 階段が出ない。
 *
 * @returns この呼び出しで新たに埋めたテクセル
 */
export function propagateOutward(
  color: Float32Array,
  filled: Uint8Array,
  region: Uint8Array,
  width: number,
  height: number,
  iterations: number,
): Uint8Array {
  const newly = new Uint8Array(filled.length);
  const target: number[] = [];
  for (let pass = 0; pass < iterations; pass++) {
    target.length = 0;
    const sums: number[] = [];
    for (let row = 0; row < height; row++) {
      for (let column = 0; column < width; column++) {
        const texel = row * width + column;
        if (region[texel] === 0 || filled[texel] !== 0) continue;
        let weight = 0;
        let red = 0;
        let green = 0;
        let blue = 0;
        for (let dy = -1; dy <= 1; dy++) {
          const otherRow = row + dy;
          if (otherRow < 0 || otherRow >= height) continue;
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            const otherColumn = column + dx;
            if (otherColumn < 0 || otherColumn >= width) continue;
            const other = otherRow * width + otherColumn;
            if (filled[other] === 0) continue;
            weight += 1;
            red += color[other * 3];
            green += color[other * 3 + 1];
            blue += color[other * 3 + 2];
          }
        }
        if (weight === 0) continue;
        target.push(texel);
        sums.push(red / weight, green / weight, blue / weight);
      }
    }
    if (target.length === 0) break;
    for (let slot = 0; slot < target.length; slot++) {
      const texel = target[slot];
      color[texel * 3] = sums[slot * 3];
      color[texel * 3 + 1] = sums[slot * 3 + 1];
      color[texel * 3 + 2] = sums[slot * 3 + 2];
      filled[texel] = 1;
      newly[texel] = 1;
    }
  }
  return newly;
}

/**
 * 写真色を3D曲面上の最寄り有効色へクランプし、局所的に平滑化した場を返す。
 *
 * UV画像のXY距離は使わない。写真色をメッシュ頂点へ集約し、未割当頂点を3D辺長で測った最寄り色
 * へクランプする。その後、screened harmonic で有限回だけ平滑化する。
 *
 * `sourceWeight` は float の (H, W)。**二値ではない** — 写真の門が傾斜なので、半分しか信用して
 * いないテクセルの色を全力で外へ運ばないため、barycentric 重みにそのまま掛ける。
 */
export function fillSurfaceClampedSmooth(input: {
  color: Float32Array;
  sourceWeight: Float64Array;
  region: Uint8Array;
  triangleIndex: Int32Array;
  barycentric: Float32Array;
  triangles: Uint32Array;
  vertexPositions: Float64Array;
  fallback: Float32Array | null;
  vertexRegionId: Uint8Array | null;
  harmonicScreening?: number;
}): Float32Array {
  const {
    color,
    sourceWeight,
    region,
    triangleIndex,
    barycentric,
    triangles,
    vertexPositions,
    vertexRegionId,
  } = input;
  const harmonicScreening = input.harmonicScreening ?? DEFAULT_SURFACE_HARMONIC_SCREENING;
  const texelCount = sourceWeight.length;
  if (color.length !== texelCount * 3 || region.length !== texelCount) {
    throw new Error('color / sourceWeight / region の形が揃っていない');
  }
  if (triangleIndex.length !== texelCount || barycentric.length !== texelCount * 3) {
    throw new Error('表面対応の形がアトラスと揃っていない');
  }
  const vertexCount = vertexPositions.length / 3;

  const fallbackColor = input.fallback ?? new Float32Array(3);
  const regionIds =
    vertexRegionId ?? new Uint8Array(vertexCount);
  if (regionIds.length !== vertexCount) {
    throw new Error(`vertexRegionId の長さが ${regionIds.length}（期待 ${vertexCount}）`);
  }

  // 写真テクセルの重み付き色を、頂点ごとのヒストグラムへ集約する。
  const vertexColorSum = new Float64Array(vertexCount * 3);
  const vertexWeight = new Float64Array(vertexCount);
  for (let texel = 0; texel < texelCount; texel++) {
    if (region[texel] === 0) continue;
    const weight = sourceWeight[texel];
    if (!(weight > 0)) continue;
    const triangle = triangleIndex[texel];
    if (triangle < 0) continue;
    for (let corner = 0; corner < 3; corner++) {
      const vertex = triangles[triangle * 3 + corner];
      const contribution = weight * barycentric[texel * 3 + corner];
      vertexWeight[vertex] += contribution;
      vertexColorSum[vertex * 3] += color[texel * 3] * contribution;
      vertexColorSum[vertex * 3 + 1] += color[texel * 3 + 1] * contribution;
      vertexColorSum[vertex * 3 + 2] += color[texel * 3 + 2] * contribution;
    }
  }

  // region のテクセルが乗る三角形の頂点だけを補完の対象にする。
  const regionVertices = new Uint8Array(vertexCount);
  for (let texel = 0; texel < texelCount; texel++) {
    if (region[texel] === 0) continue;
    const triangle = triangleIndex[texel];
    if (triangle < 0) continue;
    for (let corner = 0; corner < 3; corner++) regionVertices[triangles[triangle * 3 + corner]] = 1;
  }

  // UV seam の複製頂点は3D位置で同じ node へ統合する（シームだけは同じ頂点色を共有できる）。
  const { nodePositions, vertexToNode, nodeCount } = mergeNodesByPosition(vertexPositions);
  const nodeColor = new Float64Array(nodeCount * 3);
  const nodeWeight = new Float64Array(nodeCount);
  const nodeRegion = new Uint8Array(nodeCount);
  const nodeFillRegion = new Uint8Array(nodeCount);
  const nodeColorSum = new Float64Array(nodeCount * 3);
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    const node = vertexToNode[vertex];
    nodeWeight[node] += vertexWeight[vertex];
    nodeColorSum[node * 3] += vertexColorSum[vertex * 3];
    nodeColorSum[node * 3 + 1] += vertexColorSum[vertex * 3 + 1];
    nodeColorSum[node * 3 + 2] += vertexColorSum[vertex * 3 + 2];
    if (regionVertices[vertex] !== 0) nodeRegion[node] = 1;
    if (regionIds[vertex] > nodeFillRegion[node]) nodeFillRegion[node] = regionIds[vertex];
  }
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    if (nodeFillRegion[vertexToNode[vertex]] !== regionIds[vertex]) {
      throw new Error('同じ3D位置のseam頂点でvertexRegionIdが一致しない');
    }
  }

  const nodeKnown = new Uint8Array(nodeCount);
  for (let node = 0; node < nodeCount; node++) {
    if (nodeWeight[node] > 0) {
      nodeKnown[node] = 1;
      nodeColor[node * 3] = nodeColorSum[node * 3] / nodeWeight[node];
      nodeColor[node * 3 + 1] = nodeColorSum[node * 3 + 1] / nodeWeight[node];
      nodeColor[node * 3 + 2] = nodeColorSum[node * 3 + 2] / nodeWeight[node];
    } else {
      nodeColor[node * 3] = fallbackColor[0];
      nodeColor[node * 3 + 1] = fallbackColor[1];
      nodeColor[node * 3 + 2] = fallbackColor[2];
    }
  }

  // node 空間の三角形のうち、3 頂点すべてが region に入るものだけで辺を張る。
  const usableTriangles: number[] = [];
  for (let triangle = 0; triangle < triangles.length / 3; triangle++) {
    const a = vertexToNode[triangles[triangle * 3]];
    const b = vertexToNode[triangles[triangle * 3 + 1]];
    const c = vertexToNode[triangles[triangle * 3 + 2]];
    if (nodeRegion[a] !== 0 && nodeRegion[b] !== 0 && nodeRegion[c] !== 0) {
      usableTriangles.push(a, b, c);
    }
  }
  const regionNodeTriangles = Uint32Array.from(usableTriangles);
  const { edgeA, edgeB } = surfaceEdges(regionNodeTriangles);
  const transportScale = dihedralTransportScale(regionNodeTriangles, nodePositions, edgeA, edgeB);
  for (let edge = 0; edge < edgeA.length; edge++) {
    if (nodeFillRegion[edgeA[edge]] !== nodeFillRegion[edgeB[edge]]) {
      transportScale[edge] *= SURFACE_REGION_CROSSING_SCALE;
    }
  }

  clampAndSmoothSurfaceColor(
    nodeColor,
    nodeKnown,
    nodeRegion,
    edgeA,
    edgeB,
    nodePositions,
    transportScale,
    harmonicScreening,
  );

  // node 色を頂点へ戻し、テクセルへ barycentric で配る。
  const field = new Float32Array(texelCount * 3);
  for (let texel = 0; texel < texelCount; texel++) {
    field[texel * 3] = fallbackColor[0];
    field[texel * 3 + 1] = fallbackColor[1];
    field[texel * 3 + 2] = fallbackColor[2];
  }
  for (let texel = 0; texel < texelCount; texel++) {
    if (region[texel] === 0) continue;
    const triangle = triangleIndex[texel];
    if (triangle < 0) continue;
    let red = 0;
    let green = 0;
    let blue = 0;
    for (let corner = 0; corner < 3; corner++) {
      const node = vertexToNode[triangles[triangle * 3 + corner]];
      const weight = barycentric[texel * 3 + corner];
      red += nodeColor[node * 3] * weight;
      green += nodeColor[node * 3 + 1] * weight;
      blue += nodeColor[node * 3 + 2] * weight;
    }
    field[texel * 3] = red;
    field[texel * 3 + 1] = green;
    field[texel * 3 + 2] = blue;
  }
  return field;
}

/**
 * 同じ 3D 位置の頂点を 1 つの node に統合する（位置の辞書順で並べる）。
 *
 * UV の切れ目で複製された頂点は位置が厳密に一致するので、キーは位置そのままでよい。
 */
function mergeNodesByPosition(vertexPositions: Float64Array): {
  nodePositions: Float64Array;
  vertexToNode: Int32Array;
  nodeCount: number;
} {
  const vertexCount = vertexPositions.length / 3;
  const groups = new Map<string, number[]>();
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    const key = `${vertexPositions[vertex * 3]},${vertexPositions[vertex * 3 + 1]},${
      vertexPositions[vertex * 3 + 2]
    }`;
    const existing = groups.get(key);
    if (existing) existing.push(vertex);
    else groups.set(key, [vertex]);
  }
  const keys = [...groups.keys()].sort((first, second) => {
    const a = first.split(',').map(Number);
    const b = second.split(',').map(Number);
    for (let axis = 0; axis < 3; axis++) {
      if (a[axis] !== b[axis]) return a[axis] - b[axis];
    }
    return 0;
  });
  const nodeCount = keys.length;
  const nodePositions = new Float64Array(nodeCount * 3);
  const vertexToNode = new Int32Array(vertexCount);
  keys.forEach((key, node) => {
    const members = groups.get(key) as number[];
    for (const vertex of members) vertexToNode[vertex] = node;
    const first = members[0];
    nodePositions[node * 3] = vertexPositions[first * 3];
    nodePositions[node * 3 + 1] = vertexPositions[first * 3 + 1];
    nodePositions[node * 3 + 2] = vertexPositions[first * 3 + 2];
  });
  return { nodePositions, vertexToNode, nodeCount };
}

/** 三角形から重複のない無向辺を作る（`(a, b)` の昇順、a < b）。 */
export function surfaceEdges(triangles: Uint32Array): { edgeA: Int32Array; edgeB: Int32Array } {
  const seen = new Set<number>();
  const pairs: [number, number][] = [];
  const triangleCount = triangles.length / 3;
  for (let triangle = 0; triangle < triangleCount; triangle++) {
    for (const [first, second] of [
      [0, 1],
      [1, 2],
      [2, 0],
    ] as const) {
      const a = triangles[triangle * 3 + first];
      const b = triangles[triangle * 3 + second];
      if (a === b) continue;
      const low = Math.min(a, b);
      const high = Math.max(a, b);
      const key = low * 4294967296 + high;
      if (seen.has(key)) continue;
      seen.add(key);
      pairs.push([low, high]);
    }
  }
  pairs.sort((first, second) => (first[0] !== second[0] ? first[0] - second[0] : first[1] - second[1]));
  const edgeA = new Int32Array(pairs.length);
  const edgeB = new Int32Array(pairs.length);
  pairs.forEach(([low, high], index) => {
    edgeA[index] = low;
    edgeB[index] = high;
  });
  return { edgeA, edgeB };
}

/**
 * 辺を挟む面の法線差を、補完経路の距離倍率へ変換する。
 *
 * メッシュの接続と頂点位置だけでは、耳の付け根の折れも滑らかな頭皮と同じ経路になる。面法線の
 * 不連続を距離へ加えることで、色は滑らかな面を優先して進み、折れを越える場合だけ弱くなる。
 * 境界辺は比較相手の面がないため倍率1とする。
 */
export function dihedralTransportScale(
  triangles: Uint32Array,
  positions: Float64Array,
  edgeA: Int32Array,
  edgeB: Int32Array,
  penaltyScale = SURFACE_DIHEDRAL_PENALTY_SCALE,
): Float64Array {
  const out = new Float64Array(edgeA.length).fill(1);
  if (edgeA.length === 0) return out;
  if (penaltyScale < 0) throw new Error(`penaltyScale は0以上: ${penaltyScale}`);

  const edgeSlot = new Map<number, number>();
  for (let edge = 0; edge < edgeA.length; edge++) {
    edgeSlot.set(edgeA[edge] * 4294967296 + edgeB[edge], edge);
  }
  const count = new Float64Array(edgeA.length);
  const normalSum = new Float64Array(edgeA.length * 3);

  const triangleCount = triangles.length / 3;
  for (let triangle = 0; triangle < triangleCount; triangle++) {
    const a = triangles[triangle * 3];
    const b = triangles[triangle * 3 + 1];
    const c = triangles[triangle * 3 + 2];
    const abx = positions[b * 3] - positions[a * 3];
    const aby = positions[b * 3 + 1] - positions[a * 3 + 1];
    const abz = positions[b * 3 + 2] - positions[a * 3 + 2];
    const acx = positions[c * 3] - positions[a * 3];
    const acy = positions[c * 3 + 1] - positions[a * 3 + 1];
    const acz = positions[c * 3 + 2] - positions[a * 3 + 2];
    let nx = aby * acz - abz * acy;
    let ny = abz * acx - abx * acz;
    let nz = abx * acy - aby * acx;
    const length = Math.hypot(nx, ny, nz);
    if (length > 1e-12) {
      nx /= length;
      ny /= length;
      nz /= length;
    } else {
      nx = 0;
      ny = 0;
      nz = 0;
    }
    for (const [first, second] of [
      [0, 1],
      [1, 2],
      [2, 0],
    ] as const) {
      const u = triangles[triangle * 3 + first];
      const v = triangles[triangle * 3 + second];
      if (u === v) continue;
      const slot = edgeSlot.get(Math.min(u, v) * 4294967296 + Math.max(u, v));
      if (slot === undefined) continue;
      count[slot] += 1;
      normalSum[slot * 3] += nx;
      normalSum[slot * 3 + 1] += ny;
      normalSum[slot * 3 + 2] += nz;
    }
  }

  for (let edge = 0; edge < edgeA.length; edge++) {
    let pairDot = 1;
    if (count[edge] > 1) {
      const sumNorm2 =
        normalSum[edge * 3] ** 2 + normalSum[edge * 3 + 1] ** 2 + normalSum[edge * 3 + 2] ** 2;
      pairDot = (sumNorm2 - count[edge]) / (count[edge] * (count[edge] - 1));
    }
    const bend = 1 - Math.min(1, Math.max(-1, pairDot));
    out[edge] = 1 + penaltyScale * bend * bend;
  }
  return out;
}

/**
 * 最寄りの観測頂点を事前値にし、メッシュ表面で調和的に延長する（`color` を in-place で更新）。
 *
 * 取得元を複数混ぜてから運ぶと、首の側面のような未観測領域が、離れた服や背景寄りの色まで
 * 平均した灰色になる。そこで Dijkstra で決めた最寄り取得元を事前値として保持する。ただし
 * 取得元1点がたまたま黒い場合に備え、近傍の観測済み頂点のmedianから孤立した色だけをmedianへ
 * 置き換える。そのowner場をscreen項で拘束したグラフ・ラプラシアンで解き、取得元の選択を
 * 保ちながら境界をメッシュ上の滑らかな勾配へ変える。
 */
function clampAndSmoothSurfaceColor(
  color: Float64Array,
  known: Uint8Array,
  region: Uint8Array,
  rawEdgeA: Int32Array,
  rawEdgeB: Int32Array,
  positions: Float64Array,
  rawTransportScale: Float64Array,
  harmonicScreening: number,
): void {
  const nodeCount = region.length;
  const unknown = new Uint8Array(nodeCount);
  let unknownCount = 0;
  let knownInRegion = 0;
  for (let node = 0; node < nodeCount; node++) {
    if (region[node] !== 0 && known[node] === 0) {
      unknown[node] = 1;
      unknownCount++;
    }
    if (region[node] !== 0 && known[node] !== 0) knownInRegion++;
  }
  if (unknownCount === 0 || knownInRegion === 0) return;

  const usableEdges: number[] = [];
  for (let edge = 0; edge < rawEdgeA.length; edge++) {
    if (region[rawEdgeA[edge]] !== 0 && region[rawEdgeB[edge]] !== 0) usableEdges.push(edge);
  }
  if (usableEdges.length === 0) return;
  const edgeA = Int32Array.from(usableEdges, (edge) => rawEdgeA[edge]);
  const edgeB = Int32Array.from(usableEdges, (edge) => rawEdgeB[edge]);
  const transportScale = Float64Array.from(usableEdges, (edge) => rawTransportScale[edge]);

  const edgeLength = new Float64Array(edgeA.length);
  for (let edge = 0; edge < edgeA.length; edge++) {
    const dx = positions[edgeA[edge] * 3] - positions[edgeB[edge] * 3];
    const dy = positions[edgeA[edge] * 3 + 1] - positions[edgeB[edge] * 3 + 1];
    const dz = positions[edgeA[edge] * 3 + 2] - positions[edgeB[edge] * 3 + 2];
    edgeLength[edge] =
      Math.sqrt(SURFACE_PHOTO_PLANE_DISTANCE_SCALE * (dx * dx + dy * dy) + dz * dz) *
      transportScale[edge];
  }

  const sourceCount = Math.min(SURFACE_SOURCE_COLOR_NEIGHBORS, knownInRegion);
  const { nearestSources, settledCount } = nearestSurfaceSources(
    known,
    region,
    edgeA,
    edgeB,
    edgeLength,
    sourceCount,
  );

  // 取得元1点の偶発色を近傍 median へ寄せる（孤立した外れ値だけ）。
  const stableSourceColor = Float64Array.from(color);
  for (let node = 0; node < nodeCount; node++) {
    if (!(region[node] !== 0 && known[node] !== 0)) continue;
    const samples: number[][] = [];
    for (let slot = 0; slot < sourceCount; slot++) {
      const neighbour = nearestSources[node * sourceCount + slot];
      if (neighbour < 0) continue;
      samples.push([color[neighbour * 3], color[neighbour * 3 + 1], color[neighbour * 3 + 2]]);
    }
    if (samples.length === 0) continue;
    const center = [0, 1, 2].map((channel) => median(samples.map((sample) => sample[channel])));
    const localScale = median(
      samples.map((sample) => Math.hypot(...[0, 1, 2].map((channel) => sample[channel] - center[channel]))),
    );
    const sourceResidual = Math.hypot(
      ...[0, 1, 2].map((channel) => color[node * 3 + channel] - center[channel]),
    );
    if (sourceResidual > Math.max(3 * localScale, 0.05)) {
      stableSourceColor[node * 3] = center[0];
      stableSourceColor[node * 3 + 1] = center[1];
      stableSourceColor[node * 3 + 2] = center[2];
    }
  }

  const reachable = new Uint8Array(nodeCount);
  for (let node = 0; node < nodeCount; node++) {
    if (unknown[node] !== 0 && settledCount[node] > 0) reachable[node] = 1;
  }
  for (let node = 0; node < nodeCount; node++) {
    if (region[node] !== 0 && known[node] !== 0) {
      color[node * 3] = stableSourceColor[node * 3];
      color[node * 3 + 1] = stableSourceColor[node * 3 + 1];
      color[node * 3 + 2] = stableSourceColor[node * 3 + 2];
    }
  }
  for (let node = 0; node < nodeCount; node++) {
    if (reachable[node] === 0) continue;
    const owner = nearestSources[node * sourceCount];
    color[node * 3] = stableSourceColor[owner * 3];
    color[node * 3 + 1] = stableSourceColor[owner * 3 + 1];
    color[node * 3 + 2] = stableSourceColor[owner * 3 + 2];
  }

  // owner選択と同じ曲面計量にする。ここだけ通常の3D辺長に戻すと、ownerでは避けた写真平面上の
  // 近道をharmonic項が再び強く引き込み、首側面が灰色へ戻る。
  const positive = [...edgeLength].filter((length) => length > 0);
  const lengthFloor = positive.length > 0 ? Math.max(median(positive) * 1e-3, 1e-12) : 1;
  const harmonicWeight = new Float64Array(edgeA.length);
  for (let edge = 0; edge < edgeA.length; edge++) {
    harmonicWeight[edge] = 1 / Math.max(edgeLength[edge], lengthFloor);
  }
  const anchors = new Uint8Array(nodeCount);
  for (let node = 0; node < nodeCount; node++) {
    anchors[node] = region[node] !== 0 && known[node] !== 0 ? 1 : 0;
  }
  solveScreenedHarmonicColors(
    color,
    anchors,
    reachable,
    edgeA,
    edgeB,
    harmonicWeight,
    harmonicScreening,
  );

  for (let node = 0; node < nodeCount; node++) {
    if (reachable[node] === 0) continue;
    for (let channel = 0; channel < 3; channel++) {
      color[node * 3 + channel] = Math.min(1, Math.max(0, color[node * 3 + channel]));
    }
  }
}

function median(values: readonly number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/**
 * Dijkstraで各頂点に近い観測元を返す。
 *
 * 各 node について、近い順に最大 `sourceCount` 個の異なる取得元を確定させる。
 */
export function nearestSurfaceSources(
  known: Uint8Array,
  region: Uint8Array,
  edgeA: Int32Array,
  edgeB: Int32Array,
  edgeLength: Float64Array,
  sourceCount: number,
): { nearestSources: Int32Array; settledCount: Int16Array } {
  const nodeCount = region.length;
  // 有向辺を node ごとに並べた CSR。
  const degree = new Int32Array(nodeCount + 1);
  for (let edge = 0; edge < edgeA.length; edge++) {
    degree[edgeA[edge] + 1]++;
    degree[edgeB[edge] + 1]++;
  }
  for (let node = 0; node < nodeCount; node++) degree[node + 1] += degree[node];
  const offsets = Int32Array.from(degree);
  const cursor = Int32Array.from(degree);
  const directedTo = new Int32Array(edgeA.length * 2);
  const directedLength = new Float64Array(edgeA.length * 2);
  for (let edge = 0; edge < edgeA.length; edge++) {
    directedTo[cursor[edgeA[edge]]] = edgeB[edge];
    directedLength[cursor[edgeA[edge]]++] = edgeLength[edge];
    directedTo[cursor[edgeB[edge]]] = edgeA[edge];
    directedLength[cursor[edgeB[edge]]++] = edgeLength[edge];
  }

  const nearestSources = new Int32Array(nodeCount * sourceCount).fill(-1);
  const settledCount = new Int16Array(nodeCount);
  const heap = new MinHeap();
  for (let node = 0; node < nodeCount; node++) {
    if (known[node] !== 0 && region[node] !== 0) heap.push(0, node, node);
  }
  while (heap.size > 0) {
    const { distance, node, source } = heap.pop();
    const count = settledCount[node];
    if (count >= sourceCount) continue;
    let duplicate = false;
    for (let slot = 0; slot < count; slot++) {
      if (nearestSources[node * sourceCount + slot] === source) {
        duplicate = true;
        break;
      }
    }
    if (duplicate) continue;
    nearestSources[node * sourceCount + count] = source;
    settledCount[node] = count + 1;
    for (let slot = offsets[node]; slot < offsets[node + 1]; slot++) {
      heap.push(distance + Math.max(directedLength[slot], 1e-12), directedTo[slot], source);
    }
  }
  return { nearestSources, settledCount };
}

/** (距離, node, source) の最小ヒープ。距離が同じなら node → source の昇順（決定的）。 */
class MinHeap {
  private distances: number[] = [];
  private nodes: number[] = [];
  private sources: number[] = [];

  get size(): number {
    return this.distances.length;
  }

  private less(first: number, second: number): boolean {
    if (this.distances[first] !== this.distances[second]) {
      return this.distances[first] < this.distances[second];
    }
    if (this.nodes[first] !== this.nodes[second]) return this.nodes[first] < this.nodes[second];
    return this.sources[first] < this.sources[second];
  }

  private swap(first: number, second: number): void {
    [this.distances[first], this.distances[second]] = [this.distances[second], this.distances[first]];
    [this.nodes[first], this.nodes[second]] = [this.nodes[second], this.nodes[first]];
    [this.sources[first], this.sources[second]] = [this.sources[second], this.sources[first]];
  }

  push(distance: number, node: number, source: number): void {
    this.distances.push(distance);
    this.nodes.push(node);
    this.sources.push(source);
    let index = this.distances.length - 1;
    while (index > 0) {
      const parent = (index - 1) >> 1;
      if (!this.less(index, parent)) break;
      this.swap(index, parent);
      index = parent;
    }
  }

  pop(): { distance: number; node: number; source: number } {
    const result = { distance: this.distances[0], node: this.nodes[0], source: this.sources[0] };
    const last = this.distances.length - 1;
    this.swap(0, last);
    this.distances.pop();
    this.nodes.pop();
    this.sources.pop();
    let index = 0;
    for (;;) {
      const left = index * 2 + 1;
      const right = left + 1;
      let smallest = index;
      if (left < this.distances.length && this.less(left, smallest)) smallest = left;
      if (right < this.distances.length && this.less(right, smallest)) smallest = right;
      if (smallest === index) break;
      this.swap(index, smallest);
      index = smallest;
    }
    return result;
  }
}

/**
 * owner色を事前値とし、観測色を固定したscreened harmonic場を解く。
 *
 * `color[unknown]` はDijkstraで選んだowner色。これをscreen項として残すため、首側面が遠い服色
 * との全体平均へ引かれるのを抑えつつ、owner境界だけを連続化できる。
 */
function solveScreenedHarmonicColors(
  color: Float64Array,
  anchors: Uint8Array,
  unknown: Uint8Array,
  edgeA: Int32Array,
  edgeB: Int32Array,
  edgeWeight: Float64Array,
  screening: number,
  tolerance = SURFACE_HARMONIC_TOLERANCE,
  maximumIterations = SURFACE_HARMONIC_MAXIMUM_ITERATIONS,
): void {
  const nodeCount = unknown.length;
  const unknownIds: number[] = [];
  const pointToUnknown = new Int32Array(nodeCount).fill(-1);
  for (let node = 0; node < nodeCount; node++) {
    if (unknown[node] !== 0) {
      pointToUnknown[node] = unknownIds.length;
      unknownIds.push(node);
    }
  }
  if (unknownIds.length === 0) return;
  if (screening < 0) throw new Error(`screening は0以上: ${screening}`);
  const size = unknownIds.length;

  const degree = new Float64Array(nodeCount);
  for (let edge = 0; edge < edgeA.length; edge++) {
    degree[edgeA[edge]] += edgeWeight[edge];
    degree[edgeB[edge]] += edgeWeight[edge];
  }
  const diagonal = new Float64Array(size);
  for (let slot = 0; slot < size; slot++) {
    diagonal[slot] = degree[unknownIds[slot]];
    if (!(diagonal[slot] > 0)) throw new Error('調和補完の未知頂点に辺が無い');
  }

  const initial = new Float64Array(size * 3);
  for (let slot = 0; slot < size; slot++) {
    for (let channel = 0; channel < 3; channel++) {
      initial[slot * 3 + channel] = color[unknownIds[slot] * 3 + channel];
    }
  }
  const screenDiagonal = new Float64Array(size);
  const rightHand = new Float64Array(size * 3);
  for (let slot = 0; slot < size; slot++) {
    screenDiagonal[slot] = screening * diagonal[slot];
    for (let channel = 0; channel < 3; channel++) {
      rightHand[slot * 3 + channel] = screenDiagonal[slot] * initial[slot * 3 + channel];
    }
  }
  const interiorEdges: number[] = [];
  for (let edge = 0; edge < edgeA.length; edge++) {
    const slotA = pointToUnknown[edgeA[edge]];
    const slotB = pointToUnknown[edgeB[edge]];
    if (slotA >= 0 && anchors[edgeB[edge]] !== 0) {
      for (let channel = 0; channel < 3; channel++) {
        rightHand[slotA * 3 + channel] += color[edgeB[edge] * 3 + channel] * edgeWeight[edge];
      }
    }
    if (slotB >= 0 && anchors[edgeA[edge]] !== 0) {
      for (let channel = 0; channel < 3; channel++) {
        rightHand[slotB * 3 + channel] += color[edgeA[edge] * 3 + channel] * edgeWeight[edge];
      }
    }
    if (slotA >= 0 && slotB >= 0) interiorEdges.push(edge);
  }

  const multiply = (values: Float64Array, out: Float64Array): void => {
    for (let slot = 0; slot < size; slot++) {
      const scale = diagonal[slot] + screenDiagonal[slot];
      for (let channel = 0; channel < 3; channel++) {
        out[slot * 3 + channel] = scale * values[slot * 3 + channel];
      }
    }
    for (const edge of interiorEdges) {
      const slotA = pointToUnknown[edgeA[edge]];
      const slotB = pointToUnknown[edgeB[edge]];
      const weight = edgeWeight[edge];
      for (let channel = 0; channel < 3; channel++) {
        out[slotA * 3 + channel] -= weight * values[slotB * 3 + channel];
        out[slotB * 3 + channel] -= weight * values[slotA * 3 + channel];
      }
    }
  };

  const solution = Float64Array.from(initial);
  const multiplied = new Float64Array(size * 3);
  multiply(solution, multiplied);
  const residual = new Float64Array(size * 3);
  for (let index = 0; index < residual.length; index++) {
    residual[index] = rightHand[index] - multiplied[index];
  }
  const channelNorm = (values: Float64Array): number[] => {
    const totals = [0, 0, 0];
    for (let slot = 0; slot < size; slot++) {
      for (let channel = 0; channel < 3; channel++) {
        totals[channel] += values[slot * 3 + channel] ** 2;
      }
    }
    return totals.map(Math.sqrt);
  };
  const initialNorm = channelNorm(residual).map((value) => Math.max(value, 1e-20));
  const preconditioned = new Float64Array(size * 3);
  const applyPreconditioner = (): void => {
    for (let slot = 0; slot < size; slot++) {
      for (let channel = 0; channel < 3; channel++) {
        preconditioned[slot * 3 + channel] = residual[slot * 3 + channel] / diagonal[slot];
      }
    }
  };
  applyPreconditioner();
  const direction = Float64Array.from(preconditioned);
  const dot = (first: Float64Array, second: Float64Array): number[] => {
    const totals = [0, 0, 0];
    for (let slot = 0; slot < size; slot++) {
      for (let channel = 0; channel < 3; channel++) {
        totals[channel] += first[slot * 3 + channel] * second[slot * 3 + channel];
      }
    }
    return totals;
  };
  let residualDot = dot(residual, preconditioned);

  for (let iteration = 0; iteration < maximumIterations; iteration++) {
    const norms = channelNorm(residual);
    if (norms.every((value, channel) => value <= tolerance * initialNorm[channel])) break;
    multiply(direction, multiplied);
    const denominator = dot(direction, multiplied);
    const step = [0, 0, 0];
    const usable = [false, false, false];
    for (let channel = 0; channel < 3; channel++) {
      usable[channel] = Math.abs(denominator[channel]) > 1e-30 && Math.abs(residualDot[channel]) > 1e-30;
      step[channel] = usable[channel] ? residualDot[channel] / denominator[channel] : 0;
    }
    for (let slot = 0; slot < size; slot++) {
      for (let channel = 0; channel < 3; channel++) {
        solution[slot * 3 + channel] += direction[slot * 3 + channel] * step[channel];
        residual[slot * 3 + channel] -= multiplied[slot * 3 + channel] * step[channel];
      }
    }
    applyPreconditioner();
    const nextDot = dot(residual, preconditioned);
    const ratio = [0, 0, 0];
    for (let channel = 0; channel < 3; channel++) {
      ratio[channel] = usable[channel] ? nextDot[channel] / residualDot[channel] : 0;
    }
    for (let slot = 0; slot < size; slot++) {
      for (let channel = 0; channel < 3; channel++) {
        direction[slot * 3 + channel] =
          preconditioned[slot * 3 + channel] + direction[slot * 3 + channel] * ratio[channel];
      }
    }
    residualDot = nextDot;
  }

  for (let slot = 0; slot < size; slot++) {
    for (let channel = 0; channel < 3; channel++) {
      color[unknownIds[slot] * 3 + channel] = Math.min(
        1,
        Math.max(0, solution[slot * 3 + channel]),
      );
    }
  }
}
