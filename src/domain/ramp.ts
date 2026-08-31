// 0..1 へ滑らかに写す傾斜。
//
// 段の門を「通る / 通らない」の二値にすると、境界のテクセルや頂点で色が跳ぶ。写真から
// 焼けた領域と作った領域の境界が**線として焼き付く**のがその症状。門を傾斜にすると、
// 色の差は同じでも帯に散る。
//
// domain のどの段からも使うのでここに 1 つだけ置く（`atlas` から `hair` を import
// しないため。依存の向きが混ざると、髪の都合でアトラスが動く）。

/** `low`..`high` を 0..1 へ三次で滑らかに写す（外側はクランプ）。 */
export function smoothstep(low: number, high: number, value: number): number {
  if (!(high > low)) throw new Error(`smoothstep の区間が正でない: low=${low} high=${high}`);
  const t = Math.min(1, Math.max(0, (value - low) / (high - low)));
  return t * t * (3 - 2 * t);
}

/** `smoothstep` を配列へその場で適用する（大きな場を作り直さないため）。 */
export function smoothstepInPlace(low: number, high: number, values: Float64Array): Float64Array {
  if (!(high > low)) throw new Error(`smoothstep の区間が正でない: low=${low} high=${high}`);
  const span = high - low;
  for (let index = 0; index < values.length; index++) {
    const t = Math.min(1, Math.max(0, (values[index] - low) / span));
    values[index] = t * t * (3 - 2 * t);
  }
  return values;
}

/**
 * `level` を境にした門を `softness` の半幅でぼかす（0 で二値の門）。
 *
 * `level` は傾斜の**中点**。二値だったときの閾値をそのまま中点に置けるので、閾値に
 * 付いている根拠（実測で決めた値）が `softness` を足しても生き残る。
 */
export function gate(value: number, level: number, softness: number): number {
  if (softness <= 0) return value >= level ? 1 : 0;
  return smoothstep(level - softness, level + softness, value);
}
