// 写真に複数写った顔から「主役」を 1 人選ぶ。
//
// 検出器は写っている顔を全部返す。そのうち誰を書き出すかは**判断**なので、推論のアダプタではなく
// ここが持つ。MediaPipe の `numFaces=1`（最も確からしい 1 つ）へ任せていたのをやめた理由は、選ばれる
// 顔が「確からしさ」という不可視の量で決まり、写真の解像度やモデルの版で黙って変わるため。ここに
// 規則を書けば、なぜその人が選ばれたのかを読んで説明できる。
//
// 同じ顔をまとめてから選ぶ
// ------------------------
// 検出の当たり方は解像度に強く依存し、**どの解像度で当たるかは写真ごとに違う**。当たり方が段ごとに
// 違う以上、複数人が写った写真では**段によって見える顔の数も順序も変わる**。MediaPipe は返す順序を
// 契約していないので、順序に意味を持たせることもできない。だから探索は解像度の階段を全段回し、
// 出てきた顔を**同じ顔ごとに束ねてから**選ぶ。
//
// 主役の規則
// ----------
//     得点 = 顔の一辺 − 対象点からの距離        （最大が勝ち）
//
// **係数を持たない。** 一辺と距離を同じ画素の物差しで引くこの形は「各顔を一辺を半径とする円と見て、
// 対象点が最も深く入っている顔を選ぶ」と読める。大きい顔ほど円が大きいので中央から離れても許され、
// 中央にぴったりでも小さい顔は勝てない。**係数を足すとこの意味が消え、代わりに窓の中で動かせる調整
// つまみが残る。**

/**
 * 顔の外接正方形（画像画素座標の 中心 x, 中心 y, 一辺）。
 *
 * 顔をこの 3 つの数へ落とすのは、主役の判断に要るのが**大きさと位置だけ**だから。ランドマークを
 * 持ち回すと、判断の入力に「点の並び」という関係の無い知識が混ざる。
 */
export interface FaceSquare {
  readonly centerX: number;
  readonly centerY: number;
  readonly span: number;
}

/**
 * 2 つの検出を同じ顔と見なす中心距離の上限（**小さい方**の一辺に対する比）。
 *
 * **顔は重ならない**ので、別の 2 つの顔の中心距離は最低でも `(一辺a + 一辺b) / 2` ある。境を
 * **小さい方の半分**に置けば、その下限を必ず下回るので別の顔を巻き込まない。
 *
 * **大きい方で測ってはいけない。** 大人と乳児のように一辺が大きく違う 2 人が頬を寄せると、
 * `0.5 × 大きい方` が 2 人の中心距離を上回って束ねてしまう。束ねた代表はどちらの顔でもない中央値に
 * なり、**主役の規則が一度も採点していない顔が書き出される。**
 */
export const SAME_FACE_CENTER_SHIFT = 0.5;

/** 2 つの顔の中心距離（画素）。 */
export function centerDistance(a: FaceSquare, b: FaceSquare): number {
  return Math.hypot(a.centerX - b.centerX, a.centerY - b.centerY);
}

/** 中心距離を**小さい方**の一辺で割った比 — `SAME_FACE_CENTER_SHIFT` と同じ尺度。 */
export function centerShiftRatio(a: FaceSquare, b: FaceSquare): number {
  const smallestSpan = Math.min(a.span, b.span);
  if (!(smallestSpan > 0)) throw new Error('顔の一辺が 0 以下');
  return centerDistance(a, b) / smallestSpan;
}

/** 2 つの検出が同じ顔を指しているか。 */
export function isSameFace(a: FaceSquare, b: FaceSquare): boolean {
  return centerShiftRatio(a, b) < SAME_FACE_CENTER_SHIFT;
}

/** 画像中心の画素座標。`subjectFace` の対象点の既定。 */
export function imageCenter(imageSize: readonly [number, number]): [number, number] {
  const [width, height] = imageSize;
  if (!(width > 0 && height > 0)) throw new Error(`画像の大きさが ${width}x${height}`);
  return [width / 2, height / 2];
}

/**
 * 並びの正本 — `一辺降順 → x 昇順 → y 昇順`。
 *
 * 得点が同点でも答えを一意にするために要る（左右対称の配置は必ず同点になる）。
 */
function orderKey(face: FaceSquare): [number, number, number] {
  return [-face.span, face.centerX, face.centerY];
}

function compareOrderKey(a: FaceSquare, b: FaceSquare): number {
  const keyA = orderKey(a);
  const keyB = orderKey(b);
  for (let slot = 0; slot < 3; slot++) {
    if (keyA[slot] !== keyB[slot]) return keyA[slot] - keyB[slot];
  }
  return 0;
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/**
 * 解像度違いで何度も出てきた同じ顔を 1 つに束ね、`orderKey` の順に並べて返す。
 *
 * 束ねる関係（`isSameFace`）は推移的とは限らないので、繋がっているものを全部たぐって 1 つの塊に
 * する（連結成分）。半端に切ると、同じ顔が 2 件残って得点の比較に二重に乗る。
 *
 * 代表の中心と一辺は**中央値**。平均にしないのは、ある段だけ顔の一部しか掴めなかったときに、
 * その 1 件へ代表が引きずられるため。
 */
export function mergeFaceSquares(squares: readonly FaceSquare[]): FaceSquare[] {
  const faces = squares.map((square) => ({
    centerX: square.centerX,
    centerY: square.centerY,
    span: square.span,
  }));
  const unvisited = new Set(faces.keys());
  const merged: FaceSquare[] = [];

  while (unvisited.size > 0) {
    const seed = Math.min(...unvisited);
    unvisited.delete(seed);
    const cluster = [seed];
    const queue = [seed];
    let head = 0;
    while (head < queue.length) {
      const current = queue[head++];
      for (const other of [...unvisited].sort((a, b) => a - b)) {
        if (!isSameFace(faces[current], faces[other])) continue;
        unvisited.delete(other);
        cluster.push(other);
        queue.push(other);
      }
    }
    merged.push({
      centerX: median(cluster.map((index) => faces[index].centerX)),
      centerY: median(cluster.map((index) => faces[index].centerY)),
      span: median(cluster.map((index) => faces[index].span)),
    });
  }
  return merged.sort(compareOrderKey);
}

/** 主役の得点 `一辺 − 対象点からの距離`（大きいほど主役らしい）。 */
export function subjectScore(face: FaceSquare, target: readonly [number, number]): number {
  return face.span - Math.hypot(face.centerX - target[0], face.centerY - target[1]);
}

/**
 * 得点が最大の顔の index。同点は `orderKey` の先頭で決める。
 *
 * 同点の解きほぐしを呼び出し順ではなく `orderKey` に委ねるのは、**入力の並びに答えを依存させない
 * ため**。階段が顔を返す順は解像度で変わるので、呼び出し順で決めると同じ写真から違う答えが出る。
 */
export function subjectIndex(
  faces: readonly FaceSquare[],
  target: readonly [number, number],
): number {
  if (faces.length === 0) throw new Error('顔が 1 つも渡されていない');
  let best = 0;
  for (let index = 1; index < faces.length; index++) {
    const scoreDifference = subjectScore(faces[index], target) - subjectScore(faces[best], target);
    if (scoreDifference > 0) {
      best = index;
      continue;
    }
    if (scoreDifference === 0 && compareOrderKey(faces[index], faces[best]) < 0) best = index;
  }
  return best;
}

/**
 * 検出した顔を束ね、主役を 1 つ返す（この層の入口）。
 *
 * @param target 主役を測る対象点（画像画素座標）。既定は画像中心。**引数にしてあるのは対象点が
 *   外から与えられうるため**
 */
export function subjectFace(
  squares: readonly FaceSquare[],
  imageSize: readonly [number, number],
  target: readonly [number, number] | null = null,
): FaceSquare {
  const faces = mergeFaceSquares(squares);
  if (faces.length === 0) throw new Error('顔が 1 つも渡されていない');
  return faces[subjectIndex(faces, target ?? imageCenter(imageSize))];
}

/**
 * `reference` と同じ顔と見なせるものの中で、中心が最も近い顔の index。
 *
 * 同じ顔が 1 つも無ければ null。**「先頭が違えば全部捨てる」ではなく「どれが同じ顔かを決める」**の
 * が役目 — 検出器が複数の顔を返す以上、目当ての顔が先頭に来る保証は無く、先頭だけを見ると正解が
 * 2 件目にある段まで捨ててしまう。
 */
export function sameFaceIndex(
  faces: readonly FaceSquare[],
  reference: FaceSquare,
): number | null {
  const candidates = faces
    .map((face, index) => ({ face, index }))
    .filter(({ face }) => isSameFace(face, reference));
  if (candidates.length === 0) return null;
  candidates.sort((first, second) => {
    const difference =
      centerDistance(first.face, reference) - centerDistance(second.face, reference);
    return difference !== 0 ? difference : compareOrderKey(first.face, second.face);
  });
  return candidates[0].index;
}

/** 478 点のランドマークから顔の外接正方形を作る（顔メッシュの 468 点の bbox）。 */
export function faceSquareOfLandmarks(landmarks: Float64Array, faceMeshCount: number): FaceSquare {
  let lowX = Infinity;
  let lowY = Infinity;
  let highX = -Infinity;
  let highY = -Infinity;
  for (let point = 0; point < faceMeshCount; point++) {
    const x = landmarks[point * 2];
    const y = landmarks[point * 2 + 1];
    lowX = Math.min(lowX, x);
    highX = Math.max(highX, x);
    lowY = Math.min(lowY, y);
    highY = Math.max(highY, y);
  }
  return {
    centerX: (lowX + highX) / 2,
    centerY: (lowY + highY) / 2,
    span: Math.max(highX - lowX, highY - lowY),
  };
}
