// アトラスのテクセル → GNM 表面の対応（UV ラスタライズ）。
//
// ベイクは「アトラスのテクセルが GNM 表面のどこか」を全テクセルぶん引けることが前提。UV から
// 3D へ逆引きする代わりに、三角形を UV 空間へラスタライズして barycentric で位置と法線を
// 補間する。三角形ごとに走査範囲が数十テクセルに限られるので、テクセル側から三角形を探すより
// 速く、境界の扱いも自分で決められる。
//
// conservative rasterization
// --------------------------
// テクセル中心が三角形の内側かどうかで決めると、chart の境界に 1px の隙間が空く（境界の
// テクセルは中心が外側にあることが多い）。空いた隙間はビューアーで継ぎ目に見える。そこで
// **三角形がテクセルの正方形に触れていれば書く**。触れているかの判定は三角形と軸並行正方形の
// SAT（分離軸定理）で、分離軸は「x, y の 2 軸」と「三角形の 3 辺の法線」で足りる（凸形状同士の
// 2D なのでこれで完全）。
//
// テクセル中心が外側のテクセル（fringe）は barycentric が負になるので、[0,1] にクランプして
// 和 1 に正規化し直す。三角形上の点へ寄せることになり、外挿で法線が暴れるのを防げる。中心が
// 内側のテクセル（exact）はクランプしない。
//
// アトラス配列の座標規約（ここが正本）
// ------------------------------------
// `atlas[row * size + col]` で `col = u * size` / `row = (1 - v) * size`。v は上向き（GNM の
// UV は頭頂が v≈0.94、首が v≈0.04）なので、行 0 を v=1 にすると配列をそのまま PNG / JPEG に
// 書いたときに頭が上に来て、Unity が GNM の UV でそのまま引ける。テクセル `(row, col)` の
// 中心は `(col + 0.5, row + 0.5)`。

/** identity に依存しない、公式 UV とテクセルの対応。 */
export interface AtlasLayout {
  /** (size, size) 元の `triangles` の行 index。-1 はどの三角形も触れていないテクセル。 */
  readonly triangleIndex: Int32Array;
  /** (size, size, 3) `triangleIndex` の三角形の重み。行和 1。 */
  readonly barycentric: Float32Array;
  /**
   * (size, size) UV 上で繋がっている塊の id。三角形数の降順で 0 から振る
   * （0 = 最大 chart = 外から見える肌）。-1 は chart の外。
   */
  readonly chartIndex: Int32Array;
  /**
   * (size, size) テクセル中心が三角形の内側にあるか。0 かつ被覆ありのテクセルが
   * conservative rasterization で拾った縁。
   */
  readonly centerInside: Uint8Array;
  /**
   * (N,) 焼く構成要素の三角形 index。頂点法線の計算にも使うため、テクセルを覆わない微小
   * 三角形も含む。
   */
  readonly selectedTriangleIndex: Int32Array;
  readonly size: number;
}

/**
 * アトラスの各テクセルが指す、identity 適用後の GNM 表面上の点。
 *
 * `AtlasLayout` の静的な対応に、guest ごとの位置と法線を束縛した値。
 */
export interface AtlasSurface extends AtlasLayout {
  /** (size, size, 3) GNM 空間の位置（メートル）。 */
  readonly position: Float32Array;
  /** (size, size, 3) GNM 空間の単位法線（頂点法線の補間）。 */
  readonly normal: Float32Array;
}

/** 三角形が触れているテクセル（ベイクで埋める対象）。 */
export function coveredMask(layout: AtlasLayout): Uint8Array {
  const out = new Uint8Array(layout.triangleIndex.length);
  for (let texel = 0; texel < out.length; texel++) {
    out[texel] = layout.triangleIndex[texel] >= 0 ? 1 : 0;
  }
  return out;
}

export function chartCount(layout: AtlasLayout): number {
  let maximum = -1;
  for (const chart of layout.chartIndex) if (chart > maximum) maximum = chart;
  return maximum + 1;
}

/** 公式 UV をラスタライズして、identity 非依存のテクセル対応を作る。 */
export function buildAtlasLayout(
  triangles: Uint32Array,
  vertexUvs: Float32Array,
  componentId: Uint8Array,
  atlasSize: number,
  skinComponentId = 0,
): AtlasLayout {
  const vertexCount = vertexUvs.length / 2;
  const triangleCount = triangles.length / 3;
  if (componentId.length !== vertexCount) {
    throw new Error(`componentId の長さが ${componentId.length}（期待 ${vertexCount}）`);
  }
  if (triangleCount === 0) throw new Error('triangles が空');
  if (atlasSize <= 0) throw new Error(`atlasSize が ${atlasSize}`);

  // 構成要素の境界を跨ぐ三角形は無い（頂点グループが排他）ので、3 頂点すべてが対象なら採用、
  // という単純な判定で足りる。跨いでいたら被覆が欠けるので検証する。
  const selected: number[] = [];
  let straddling = 0;
  for (let triangle = 0; triangle < triangleCount; triangle++) {
    let matches = 0;
    for (let corner = 0; corner < 3; corner++) {
      const vertex = triangles[triangle * 3 + corner];
      if (vertex >= vertexCount) throw new Error('triangles の頂点 index が範囲外');
      if (componentId[vertex] === skinComponentId) matches++;
    }
    if (matches === 3) selected.push(triangle);
    else if (matches > 0) straddling++;
  }
  if (straddling !== 0) {
    throw new Error(
      `構成要素 ${skinComponentId} の境界を跨ぐ三角形が ${straddling} 個ある。` +
        ' 3 頂点すべてで判定すると被覆が欠ける。',
    );
  }
  if (selected.length === 0) throw new Error(`構成要素 ${skinComponentId} の三角形が無い`);

  const selectedTriangleIndex = Int32Array.from(selected);
  // 三角形の角をテクセル座標へ（`col = u * size` / `row = (1 - v) * size`）。
  const cornerTexels = new Float64Array(selected.length * 6);
  for (let local = 0; local < selected.length; local++) {
    const triangle = selected[local];
    for (let corner = 0; corner < 3; corner++) {
      const vertex = triangles[triangle * 3 + corner];
      cornerTexels[local * 6 + corner * 2] = vertexUvs[vertex * 2] * atlasSize;
      cornerTexels[local * 6 + corner * 2 + 1] = (1 - vertexUvs[vertex * 2 + 1]) * atlasSize;
    }
  }

  const { localIndex, centerInside } = rasterize(cornerTexels, selected.length, atlasSize);
  const triangleIndex = new Int32Array(atlasSize * atlasSize);
  for (let texel = 0; texel < triangleIndex.length; texel++) {
    triangleIndex[texel] = localIndex[texel] >= 0 ? selected[localIndex[texel]] : -1;
  }

  return {
    triangleIndex,
    barycentric: barycentricOfTexels(cornerTexels, localIndex, centerInside, atlasSize),
    chartIndex: chartIndexOfTexels(triangles, selectedTriangleIndex, localIndex, vertexCount),
    centerInside,
    selectedTriangleIndex,
    size: atlasSize,
  };
}

/** 静的な `layout` に identity 適用後の位置と法線を束縛する。 */
export function bindAtlasSurface(
  layout: AtlasLayout,
  vertices: Float64Array,
  triangles: Uint32Array,
): AtlasSurface {
  const vertexCount = vertices.length / 3;
  const triangleCount = triangles.length / 3;
  if (layout.selectedTriangleIndex.length === 0) throw new Error('layout に対象三角形が無い');
  for (const triangle of layout.selectedTriangleIndex) {
    if (triangle >= triangleCount) throw new Error('layout の三角形 index が triangles の範囲外');
  }

  const normals = vertexNormals(vertices, triangles, layout.selectedTriangleIndex);
  const texelCount = layout.size * layout.size;
  const position = new Float32Array(texelCount * 3);
  const normal = new Float32Array(texelCount * 3);
  for (let texel = 0; texel < texelCount; texel++) {
    const triangle = layout.triangleIndex[texel];
    if (triangle < 0) continue;
    let px = 0;
    let py = 0;
    let pz = 0;
    let nx = 0;
    let ny = 0;
    let nz = 0;
    for (let corner = 0; corner < 3; corner++) {
      const vertex = triangles[triangle * 3 + corner];
      if (vertex >= vertexCount) throw new Error('layout のテクセル対応が範囲外を指している');
      const weight = layout.barycentric[texel * 3 + corner];
      px += vertices[vertex * 3] * weight;
      py += vertices[vertex * 3 + 1] * weight;
      pz += vertices[vertex * 3 + 2] * weight;
      nx += normals[vertex * 3] * weight;
      ny += normals[vertex * 3 + 1] * weight;
      nz += normals[vertex * 3 + 2] * weight;
    }
    position[texel * 3] = px;
    position[texel * 3 + 1] = py;
    position[texel * 3 + 2] = pz;
    const length = Math.hypot(nx, ny, nz);
    if (length > 0) {
      normal[texel * 3] = nx / length;
      normal[texel * 3 + 1] = ny / length;
      normal[texel * 3 + 2] = nz / length;
    }
  }
  return { ...layout, position, normal };
}

/** 公式 UV をラスタライズして、テクセルごとの GNM 表面上の点を作る。 */
export function buildAtlasSurface(
  vertices: Float64Array,
  triangles: Uint32Array,
  vertexUvs: Float32Array,
  componentId: Uint8Array,
  atlasSize: number,
  skinComponentId = 0,
): AtlasSurface {
  const layout = buildAtlasLayout(triangles, vertexUvs, componentId, atlasSize, skinComponentId);
  if (vertices.length / 3 !== vertexUvs.length / 2) {
    throw new Error('vertices と vertexUvs の頂点数が合わない');
  }
  return bindAtlasSurface(layout, vertices, triangles);
}

// ---------------------------------------------------------------------------
// ラスタライズ
// ---------------------------------------------------------------------------
function cross2(ax: number, ay: number, bx: number, by: number): number {
  return ax * by - ay * bx;
}

/**
 * 三角形を UV 空間へ conservative にラスタライズする。
 *
 * 重なりの解決: 中心が内側のテクセル（exact）が縁（fringe）に必ず勝つ。同じ優先度で競合したら
 * 三角形 index の小さい方が勝つ。**三角形を降順に走って上書きする**ことで「最後に書いた値が
 * 残る」規則がそのまま優先順位になる（決定的）。
 */
function rasterize(
  corners: Float64Array,
  triangleCount: number,
  size: number,
): { localIndex: Int32Array; centerInside: Uint8Array } {
  const exact = new Int32Array(size * size).fill(-1);
  const fringe = new Int32Array(size * size).fill(-1);

  for (let local = triangleCount - 1; local >= 0; local--) {
    const base = local * 6;
    const ax = corners[base];
    const ay = corners[base + 1];
    const bx = corners[base + 2];
    const by = corners[base + 3];
    const cx = corners[base + 4];
    const cy = corners[base + 5];

    const area2 = cross2(bx - ax, by - ay, cx - ax, cy - ay);
    // UV が潰れた三角形はテクセルを覆わない（面積 0）。0 除算を避けて落とす。
    if (!(Math.abs(area2) > 0)) continue;
    const sign = area2 >= 0 ? 1 : -1;

    // 正方形 [c, c+1] が三角形の bbox に触れる c の範囲。1 テクセルぶん広めに取って
    // 取りこぼしを無くし、実際に触れているかは SAT に任せる。
    const columnLow = Math.min(Math.max(Math.floor(Math.min(ax, bx, cx)) - 1, 0), size - 1);
    const columnHigh = Math.min(Math.max(Math.floor(Math.max(ax, bx, cx)) + 1, 0), size - 1);
    const rowLow = Math.min(Math.max(Math.floor(Math.min(ay, by, cy)) - 1, 0), size - 1);
    const rowHigh = Math.min(Math.max(Math.floor(Math.max(ay, by, cy)) + 1, 0), size - 1);

    // 辺ベクトルの成分和の半分が、単位正方形を辺法線へ射影したときの支持半径。辺関数と同じ
    // スケール（正規化していない法線）で測っているのでそのまま比較できる。
    const supportA = 0.5 * (Math.abs(cx - bx) + Math.abs(cy - by));
    const supportB = 0.5 * (Math.abs(ax - cx) + Math.abs(ay - cy));
    const supportC = 0.5 * (Math.abs(bx - ax) + Math.abs(by - ay));

    for (let row = rowLow; row <= rowHigh; row++) {
      const py = row + 0.5;
      for (let column = columnLow; column <= columnHigh; column++) {
        const px = column + 0.5;
        const edgeA = sign * cross2(cx - bx, cy - by, px - bx, py - by);
        const edgeB = sign * cross2(ax - cx, ay - cy, px - cx, py - cy);
        const edgeC = sign * cross2(bx - ax, by - ay, px - ax, py - ay);
        const inside = edgeA >= 0 && edgeB >= 0 && edgeC >= 0;
        const touching = edgeA >= -supportA && edgeB >= -supportB && edgeC >= -supportC;
        if (!touching) continue;
        if (inside) exact[row * size + column] = local;
        else fringe[row * size + column] = local;
      }
    }
  }

  const centerInside = new Uint8Array(size * size);
  const localIndex = new Int32Array(size * size);
  for (let texel = 0; texel < localIndex.length; texel++) {
    const isExact = exact[texel] >= 0;
    centerInside[texel] = isExact ? 1 : 0;
    localIndex[texel] = isExact ? exact[texel] : fringe[texel];
  }
  return { localIndex, centerInside };
}

/** 被覆テクセルの barycentric を求める。縁のテクセルは三角形上へ寄せる。 */
function barycentricOfTexels(
  corners: Float64Array,
  localIndex: Int32Array,
  centerInside: Uint8Array,
  size: number,
): Float32Array {
  const out = new Float32Array(localIndex.length * 3);
  for (let texel = 0; texel < localIndex.length; texel++) {
    const local = localIndex[texel];
    if (local < 0) continue;
    const row = Math.floor(texel / size);
    const column = texel - row * size;
    const px = column + 0.5;
    const py = row + 0.5;
    const base = local * 6;
    const ax = corners[base];
    const ay = corners[base + 1];
    const bx = corners[base + 2];
    const by = corners[base + 3];
    const cx = corners[base + 4];
    const cy = corners[base + 5];
    const area2 = cross2(bx - ax, by - ay, cx - ax, cy - ay);
    const sign = area2 >= 0 ? 1 : -1;
    const absArea2 = Math.abs(area2);
    let wa = (sign * cross2(cx - bx, cy - by, px - bx, py - by)) / absArea2;
    let wb = (sign * cross2(ax - cx, ay - cy, px - cx, py - cy)) / absArea2;
    let wc = (sign * cross2(bx - ax, by - ay, px - ax, py - ay)) / absArea2;
    if (centerInside[texel] === 0) {
      // 縁のテクセルは中心が三角形の外にあるので重みが負になる。クランプして和 1 に直すと
      // 三角形の内側の点になり、法線と位置が外挿で暴れない（ズレは半テクセル）。
      wa = Math.min(1, Math.max(0, wa));
      wb = Math.min(1, Math.max(0, wb));
      wc = Math.min(1, Math.max(0, wc));
      const total = wa + wb + wc;
      wa /= total;
      wb /= total;
      wc /= total;
    }
    out[texel * 3] = wa;
    out[texel * 3 + 1] = wb;
    out[texel * 3 + 2] = wc;
  }
  return out;
}

/**
 * 面積重みの頂点法線 (V, 3)。
 *
 * 面法線ではなく頂点法線を使う理由は正面判定を滑らかにするため（面法線だと閾値の境界が
 * 三角形の形に沿ってギザギザになる）。外積の長さが面積の 2 倍なので、正規化せずに足すと
 * 面積重みになる。
 */
export function vertexNormals(
  vertices: Float64Array,
  triangles: Uint32Array,
  selectedTriangles: Int32Array,
): Float64Array {
  const out = new Float64Array(vertices.length);
  for (const triangle of selectedTriangles) {
    const a = triangles[triangle * 3];
    const b = triangles[triangle * 3 + 1];
    const c = triangles[triangle * 3 + 2];
    const abx = vertices[b * 3] - vertices[a * 3];
    const aby = vertices[b * 3 + 1] - vertices[a * 3 + 1];
    const abz = vertices[b * 3 + 2] - vertices[a * 3 + 2];
    const acx = vertices[c * 3] - vertices[a * 3];
    const acy = vertices[c * 3 + 1] - vertices[a * 3 + 1];
    const acz = vertices[c * 3 + 2] - vertices[a * 3 + 2];
    const nx = aby * acz - abz * acy;
    const ny = abz * acx - abx * acz;
    const nz = abx * acy - aby * acx;
    for (const vertex of [a, b, c]) {
      out[vertex * 3] += nx;
      out[vertex * 3 + 1] += ny;
      out[vertex * 3 + 2] += nz;
    }
  }
  for (let vertex = 0; vertex < out.length / 3; vertex++) {
    const length = Math.hypot(out[vertex * 3], out[vertex * 3 + 1], out[vertex * 3 + 2]);
    if (length > 0) {
      out[vertex * 3] /= length;
      out[vertex * 3 + 1] /= length;
      out[vertex * 3 + 2] /= length;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// chart（UV 上で繋がっている塊）
// ---------------------------------------------------------------------------
/**
 * 三角形ごとの chart id。三角形数の降順で 0 から振る。
 *
 * per-vertex UV 空間では「同じ頂点 index を共有する = UV も連続」なので、頂点の連結成分が
 * そのまま chart になる。id を三角形数の降順にするのは、最大 chart が「外から見える肌」だと
 * 決めたいから。残りは口腔壁のような内側の面で、写真には絶対に写らない。
 */
function chartLabels(
  triangles: Uint32Array,
  selectedTriangles: Int32Array,
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
    // 小さい index を根にすると、Python の「最小 label の伝播」と同じ根が選ばれる。
    if (rootFirst < rootSecond) parent[rootSecond] = rootFirst;
    else parent[rootFirst] = rootSecond;
  };
  for (const triangle of selectedTriangles) {
    const a = triangles[triangle * 3];
    const b = triangles[triangle * 3 + 1];
    const c = triangles[triangle * 3 + 2];
    union(a, b);
    union(b, c);
  }

  const counts = new Map<number, number>();
  for (const triangle of selectedTriangles) {
    const root = find(triangles[triangle * 3]);
    counts.set(root, (counts.get(root) ?? 0) + 1);
  }
  // 三角形数の降順、同数なら root の昇順（決定的）。
  const roots = [...counts.keys()].sort((first, second) => {
    const difference = (counts.get(second) as number) - (counts.get(first) as number);
    return difference !== 0 ? difference : first - second;
  });
  const rank = new Map<number, number>();
  roots.forEach((root, index) => rank.set(root, index));

  const out = new Int32Array(selectedTriangles.length);
  for (let local = 0; local < selectedTriangles.length; local++) {
    out[local] = rank.get(find(triangles[selectedTriangles[local] * 3])) as number;
  }
  return out;
}

/** テクセルごとの chart id。被覆なしは -1。 */
function chartIndexOfTexels(
  triangles: Uint32Array,
  selectedTriangles: Int32Array,
  localIndex: Int32Array,
  vertexCount: number,
): Int32Array {
  const labels = chartLabels(triangles, selectedTriangles, vertexCount);
  const out = new Int32Array(localIndex.length);
  for (let texel = 0; texel < localIndex.length; texel++) {
    out[texel] = localIndex[texel] >= 0 ? labels[localIndex[texel]] : -1;
  }
  return out;
}
