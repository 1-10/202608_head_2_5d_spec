// 出力契約の型。
//
// guest zip に入る成果物を、ファイルにする前の値として表す。Unity 側が読む形式そのものが
// 契約なので、ここの型を変えることは出力仕様を変えること。
//
// 契約の対象:
//     guest.json           identity 係数（個数は GNM アセットの成分数）
//     skin_albedo.jpg      GNM 公式 UV アトラス（一辺は入口が選ぶ・sRGB）
//     left_eye_albedo.png  解剖学的左の眼球テクスチャ（写真を焼いたもの）
//     right_eye_albedo.png 解剖学的右の眼球テクスチャ（同上）
//     hair_shell.bin       髪シェルのジオメトリ（GNM 空間の position / uv / index）
//     hair_albedo.jpg      髪シェル用テクスチャ（UV は画像 UV 空間）
//     hair_alpha.png       髪マスク
//
// Exporter は頭部ジオメトリを持たない。頭部の頂点は Unity 側が identity 係数から再構成する
// ため、この契約に頭部頂点は現れない。
//
// **この型が自分で契約を守る。** Unity 側は食い違いを見つけたら明示エラーで止める契約なので、
// 書き出す側も同じ検査を自分で通す。値を作った時点で落とせば、zip を書いてから Unity で
// 気付く往復が消える。検査を生成関数に置いているのは、途中で書き換えた値が検査を通らずに
// zip へ流れる経路を作らないため。
//
// skin_albedo の行と UV の対応（この向きが契約）
// --------------------------------------------
// 行 0 が v = 1 側。テクセル `(row, col)` の中心が指す UV は:
//
//     u = (col + 0.5) / W
//     v = 1 - (row + 0.5) / H
//
// 配列を画像としてそのまま保存した状態そのもの（上下を入れ替えない）。GNM の公式 UV は
// 頭頂が v ≈ 0.94 なので、この向きで保存すると画像の上に頭頂が来る。
//
// ベイク側の正本は `domain/atlas/surface` の「アトラス配列の座標規約」。この 2 式はそこと
// 同じ規約を消費側の言葉で書いたもので、`atlasRowColToUv` が両者を突き合わせるための実行
// 可能な形になっている。
//
// hair_shell の UV
// ----------------
// `uv_origin` が示すのは **hair_shell.bin の UV と skin_albedo の UV の両方**。
// `bottom-left` すなわち v = 0 が画像の下端。`domain/field` の「画像 UV 空間」（v 下向き）
// とは v の向きが逆なので、髪シェル生成の出力をここへ入れるときは `hairShellFromImageUv` で
// 読み替える。読み替えを型の入口に置いているのは、向きの取り違えが「テクスチャが上下逆に
// 貼られる」という気付きにくい形で出るため。
//
// 口腔内の色は契約に出さない
// --------------------------
// 歯・歯茎・舌はテクスチャを持たない面で、**その色は消費側が決める**。公式は
// `色 = 肌色 × scale + offset` でゲストの肌色から作る規則を持つが、消費側はこれを採らず
// 専用のマテリアルで色と質感を決めている。**この契約は口腔内の色を運ばない。**
//
// **口腔壁（`mouth_sock`）だけは Exporter が塗って送る。** 406 頂点が `skin` 構成要素の
// 一部で肌の UV を持つため、`skin_albedo` の中の 1 領域として必ず入る（送らないという選択肢
// が無い）。写真には写らないので、`domain/atlas/bake` が肌色から作った色を焼き込んでいる。
//
// **消費側が実際にどちらであるかをここに書かない。** Exporter からは検証できないので、
// 向こうが変えたときに黙って嘘になる。
//
// **眼球も公式の規則を使わない。** 公式は肌色から作るが、こちらは写真の実画素を焼く。
// 意図した逸脱。眼球は左右とも常にその側の写真が由来であり、選択肢が無いので由来フィールドは
// 持たない。
//
// 眼球テクスチャは左右 2 枚
// -------------------------
// **1 枚を両目で共用しない。** 共用は「回転対称な同心円しか描かない」という制約と引き換え
// だった。写真の画素を焼くとキャッチライトも片目だけの充血も入るので、その制約は満たせない。
//
// 名前は**解剖学的な側**。どちらの写真の目がどちら側かは `domain/eyes/geometry.assignEyeSides`
// が相似変換とコンポーネントの重心から決める。
//
// 絵は UV(0.5, 0.5) を中心とする極座標に写真を焼いたもの。境界半径は `domain/eyes/layout` が
// 正本（**半径の数値をここへ写さない**）。
//
// skin_albedo と違い眼球は JPEG にしない。虹彩の細かい模様と瞳孔の縁で JPEG の量子化が
// リンギングを出す。**向きは skin_albedo と同じ規約に従い、上下は結果に効く**。

import { EYE_SIDES, EyeSide } from './eyes/layout';

/**
 * guest zip の形式バージョン。zip の中身の構成が変わったら上げる。
 *
 * 1 → 2 で `skin_base_color` を落とした。**消費側が動き出した後の初めての変更**なので
 * 上げている。消費側は 2 を読めるようになるまで、この Exporter が書いた zip を読めない。
 * **古い版を黙って読ませない**のが版を持つ理由。
 */
export const FORMAT_VERSION = 2;

/** パッケージのバージョンが取れない場合の `exporterVersion`。 */
export const UNKNOWN_EXPORTER_VERSION = '0+unknown';

/**
 * zip に入るテクスチャの色空間。**リニアではない**。
 *
 * `uvOrigin` と同じ性格の値で、消費側が選ぶ設定ではなく「このバイト列が何であるか」の申告
 * である。値は 1 つしか許さない。**申告するのはテクスチャに ICC プロファイルを埋めていない
 * ため。** リニアとして読まれると全体が白く浮く。
 */
export const COLOR_SPACE = 'srgb';

/**
 * UV の原点。v = 0 が下。skin_albedo と hair_shell の両方に効く。
 *
 * `infrastructure/gnmb` が GNMB コンテナのヘッダへ書く値と同じでなければならない。同じで
 * あることは `infrastructure/packaging` が書き出しの度に突き合わせる。
 */
export const UV_ORIGIN = 'bottom-left';

export const MANIFEST_NAME = 'guest.json';
export const SKIN_ALBEDO_NAME = 'skin_albedo.jpg';
export const HAIR_SHELL_NAME = 'hair_shell.bin';
export const HAIR_ALBEDO_NAME = 'hair_albedo.jpg';
export const HAIR_ALPHA_NAME = 'hair_alpha.png';

/**
 * 側 → 眼球テクスチャのエントリ名。
 *
 * 側の語から組み立てるのは、名前と側の対応を 2 箇所に書かないため（`domain/eyes/layout` の
 * `EYE_SIDES` が正本）。
 */
export const EYE_ALBEDO_NAMES: Readonly<Record<EyeSide, string>> = {
  left: 'left_eye_albedo.png',
  right: 'right_eye_albedo.png',
};

/** どの写真でも zip に入る 4 つ。眼球テクスチャは左右とも常に入る。 */
export const ALWAYS_ENTRY_NAMES: readonly string[] = [
  MANIFEST_NAME,
  SKIN_ALBEDO_NAME,
  ...EYE_SIDES.map((side) => EYE_ALBEDO_NAMES[side]),
];

/** 髪が無い写真では zip に入らない 3 つ。3 つ揃うか 3 つとも無いかの二択。 */
export const HAIR_ENTRY_NAMES: readonly string[] = [
  HAIR_SHELL_NAME,
  HAIR_ALBEDO_NAME,
  HAIR_ALPHA_NAME,
];

/**
 * zip のファイル名。時刻は `capturedAt` から作る（別に時計を読まない）。
 */
export function zipNameOf(manifest: GuestManifest): string {
  const captured = capturedAtDate(manifest);
  const pad = (value: number, width = 2): string => String(value).padStart(width, '0');
  return (
    `guest_${captured.getFullYear()}${pad(captured.getMonth() + 1)}${pad(captured.getDate())}` +
    `${pad(captured.getHours())}${pad(captured.getMinutes())}${pad(captured.getSeconds())}.zip`
  );
}

/**
 * アトラス配列の `(row, col)` が指す UV を返す（モジュール冒頭の 2 式）。
 *
 * 式を関数にしておくのは、規約をテストと検査画像から実際に呼べる形にするため。コメントに
 * 書いた式は写した先とズレても黙っているが、これはズレたら落ちる。
 */
export function atlasRowColToUv(row: number, column: number, size: number): [number, number] {
  return [(column + 0.5) / size, 1 - (row + 0.5) / size];
}

/** guest.json の内容。 */
export interface GuestManifest {
  /** guest zip の形式バージョン（`FORMAT_VERSION`）。 */
  readonly format_version: number;
  /** 書き出したパッケージのバージョン。 */
  readonly exporter_version: string;
  /** GNM アセットのバージョン。Unity が保持する公式GNMと照合する。 */
  readonly gnm_version: string;
  /** GNM アセットの variant。同上。 */
  readonly gnm_variant: string;
  /**
   * `identity` の個数。冗長だが消費側の読み込みが「読む個数」を配列を触る前に知るために
   * 持つ。**個数は GNM アセットの成分数で決まる**ので、この実装は特定の値を要求しない。
   */
  readonly identity_count: number;
  /** identity 係数。この zip の唯一の数値成果物。 */
  readonly identity: readonly number[];
  /** skin_albedo の一辺（テクセル）。 */
  readonly atlas_size: number;
  /**
   * 眼球テクスチャの一辺（テクセル）。**左右で同じ**。
   *
   * **画像から読めるものをあえて申告する** — zip に入る画像の形を manifest だけで検査
   * できる状態を保つため。
   */
  readonly eye_texture_size: number;
  /** UV の原点（`UV_ORIGIN`）。 */
  readonly uv_origin: string;
  /** zip に入るテクスチャの色空間（`COLOR_SPACE`）。 */
  readonly color_space: string;
  /** 書き出した時刻。ISO8601。 */
  readonly captured_at: string;
}

/** キーの並び（= 契約の並び）。`fromJson` の照合と JSON の出力順の正本。 */
const MANIFEST_KEYS: readonly (keyof GuestManifest)[] = [
  'format_version',
  'exporter_version',
  'gnm_version',
  'gnm_variant',
  'identity_count',
  'identity',
  'atlas_size',
  'eye_texture_size',
  'uv_origin',
  'color_space',
  'captured_at',
];

/**
 * 自分で決まるフィールドを埋めて manifest を作る。
 *
 * `format_version` / `exporter_version` / `identity_count` / `uv_origin` / `color_space` は
 * 呼び出し側が決める余地が無い（この実装が何であるかで決まる）。引数に出すと呼び出し側ごとに
 * 違う値を入れられてしまう。
 *
 * 眼球テクスチャは**一辺だけを受け取る**（左右で同じことは呼び出し側の成果物の型が保証する）。
 */
export function createGuestManifest(input: {
  identity: Float64Array | readonly number[];
  gnmVersion: string;
  gnmVariant: string;
  atlasSize: number;
  eyeTextureSize: number;
  capturedAt: Date;
  exporterVersion: string;
}): GuestManifest {
  const identity = Array.from(input.identity, (value) => Number(value));
  const manifest: GuestManifest = {
    format_version: FORMAT_VERSION,
    exporter_version: input.exporterVersion,
    gnm_version: input.gnmVersion,
    gnm_variant: input.gnmVariant,
    identity_count: identity.length,
    identity,
    atlas_size: input.atlasSize,
    eye_texture_size: input.eyeTextureSize,
    uv_origin: UV_ORIGIN,
    color_space: COLOR_SPACE,
    captured_at: isoWithSeconds(input.capturedAt),
  };
  validateGuestManifest(manifest);
  return manifest;
}

/** ローカル時刻のオフセット付き ISO8601（秒まで）。Python の `isoformat(timespec="seconds")`。 */
function isoWithSeconds(value: Date): string {
  const pad = (number: number): string => String(Math.floor(Math.abs(number))).padStart(2, '0');
  const offsetMinutes = -value.getTimezoneOffset();
  const sign = offsetMinutes >= 0 ? '+' : '-';
  return (
    `${value.getFullYear()}-${pad(value.getMonth() + 1)}-${pad(value.getDate())}` +
    `T${pad(value.getHours())}:${pad(value.getMinutes())}:${pad(value.getSeconds())}` +
    `${sign}${pad(offsetMinutes / 60)}:${pad(offsetMinutes % 60)}`
  );
}

export function validateGuestManifest(manifest: GuestManifest): void {
  // identity_count は identity の長さと同じことを言っている冗長なフィールド。冗長なものを
  // 持つなら、書く側が食い違いを通さないのが条件。
  if (manifest.identity_count !== manifest.identity.length) {
    throw new Error(
      `identity_count ${manifest.identity_count} が identity の個数` +
        ` ${manifest.identity.length} と食い違っている`,
    );
  }
  if (manifest.format_version !== FORMAT_VERSION) {
    throw new Error(
      `format_version が ${manifest.format_version}（この実装は ${FORMAT_VERSION}）`,
    );
  }
  if (manifest.identity.length === 0) {
    throw new Error('identity が空（フィットの結果が入っていない）');
  }
  if (!manifest.identity.every((value) => Number.isFinite(value))) {
    throw new Error('identity に有限でない値がある');
  }
  if (manifest.atlas_size <= 0) throw new Error(`atlas_size が正でない: ${manifest.atlas_size}`);
  if (manifest.eye_texture_size <= 0) {
    throw new Error(`eye_texture_size が正でない: ${manifest.eye_texture_size}`);
  }
  if (manifest.uv_origin !== UV_ORIGIN) {
    throw new Error(`uv_origin が ${manifest.uv_origin}（対応 ${UV_ORIGIN}）`);
  }
  // 色空間はこの実装が何であるかで決まる（呼び出し側に選ばせる余地が無い）ので、uv_origin と
  // 同じく 1 つの値だけを許す。
  if (manifest.color_space !== COLOR_SPACE) {
    throw new Error(`color_space が ${manifest.color_space}（対応 ${COLOR_SPACE}）`);
  }
  if (!manifest.gnm_version || !manifest.gnm_variant) {
    throw new Error('gnm_version / gnm_variant が空（Unity側のGNMと照合できない）');
  }
  // captured_at は zip のファイル名の元になるので、ここで読めることを確かめる。
  capturedAtDate(manifest);
}

/** `captured_at` を Date に戻す。zip のファイル名を作るのに使う。 */
export function capturedAtDate(manifest: GuestManifest): Date {
  const parsed = new Date(manifest.captured_at);
  if (Number.isNaN(parsed.getTime())) {
    throw new Error(`captured_at が ISO8601 でない: ${manifest.captured_at}`);
  }
  return parsed;
}

/** JSON に落とせる形にする。キーの並びは宣言順（= 契約の並び）。 */
export function manifestToJson(manifest: GuestManifest): string {
  const ordered: Record<string, unknown> = {};
  for (const key of MANIFEST_KEYS) ordered[key] = manifest[key];
  return JSON.stringify(ordered, null, 2);
}

/**
 * guest.json から読み戻す。未知のキー・欠けたキーは例外。
 *
 * 読み戻しを持つのは、書いたものが読めることをテストで往復させるため。**版を先に見る** —
 * 版が違えばキーの集合も違うのが普通なので、キーから先に調べると原因から 1 段離れた診断が
 * 出る。
 */
export function manifestFromJson(values: Record<string, unknown>): GuestManifest {
  const foundVersion = values['format_version'];
  if (foundVersion !== FORMAT_VERSION) {
    throw new Error(
      `guest.json の format_version が ${String(foundVersion)}` +
        `（この実装は ${FORMAT_VERSION}）`,
    );
  }
  const expected = new Set<string>(MANIFEST_KEYS as readonly string[]);
  const actual = new Set(Object.keys(values));
  const missing = [...expected].filter((key) => !actual.has(key)).sort();
  const unknown = [...actual].filter((key) => !expected.has(key)).sort();
  if (missing.length > 0 || unknown.length > 0) {
    throw new Error(`guest.json のキーが合わない: 欠け=${missing} 余り=${unknown}`);
  }
  const manifest = {
    ...(values as unknown as GuestManifest),
    identity: (values['identity'] as number[]).map((value) => Number(value)),
  };
  validateGuestManifest(manifest);
  return manifest;
}

/**
 * 髪シェルのジオメトリ（GNM 空間・右手系・メートル）。
 *
 * `positions` は GNM 空間そのまま（座標変換はしない。左手系化は消費側）。`uvs` は
 * `UV_ORIGIN` の向き（v = 0 が画像の下端）。`triangles` は外向き法線
 * `(v1 − v0) × (v2 − v0)`。
 */
export interface HairShell {
  readonly positions: Float32Array;
  readonly uvs: Float32Array;
  readonly triangles: Uint32Array;
  readonly vertexCount: number;
  readonly triangleCount: number;
}

/**
 * UV が `domain/field` の画像 UV 空間（v 下向き）の値から作る。
 *
 * v を反転する。写真をそのまま JPEG に保存すると、消費側が UV (0, 0) で引くのは画像の**下**
 * 端（`UV_ORIGIN` の向き）なので、画像 UV 空間の v をそのまま渡すとテクスチャが上下逆に
 * 貼られる。反転をここに1箇所だけ置く。
 */
export function hairShellFromImageUv(
  positions: Float32Array,
  imageUvs: Float32Array,
  triangles: Uint32Array,
): HairShell {
  if (imageUvs.length % 2 !== 0) throw new Error(`imageUvs の形が (Nv, 2) ではない`);
  const flipped = new Float32Array(imageUvs.length);
  for (let vertex = 0; vertex < imageUvs.length / 2; vertex++) {
    flipped[vertex * 2] = imageUvs[vertex * 2];
    flipped[vertex * 2 + 1] = 1 - imageUvs[vertex * 2 + 1];
  }
  return makeHairShell(positions, flipped, triangles);
}

export function makeHairShell(
  positions: Float32Array,
  uvs: Float32Array,
  triangles: Uint32Array,
): HairShell {
  if (positions.length % 3 !== 0) throw new Error('positions の形が (Nv, 3) ではない');
  const vertexCount = positions.length / 3;
  if (uvs.length !== vertexCount * 2) {
    throw new Error(`uvs の形が (${vertexCount}, 2) ではない: ${uvs.length}`);
  }
  if (triangles.length % 3 !== 0) throw new Error('triangles の形が (Nt, 3) ではない');
  const triangleCount = triangles.length / 3;
  if (vertexCount === 0 || triangleCount === 0) {
    throw new Error(
      `空の髪シェル（頂点 ${vertexCount} / 三角形 ${triangleCount}）。` +
        ' 髪が無い写真では HairShell を作らず null にすること',
    );
  }
  for (const index of triangles) {
    if (index >= vertexCount) {
      throw new Error(`triangles が頂点数 ${vertexCount} の範囲外を指している`);
    }
  }
  for (const value of positions) {
    if (!Number.isFinite(value)) throw new Error('positions に有限でない値がある');
  }
  for (const value of uvs) {
    if (!Number.isFinite(value)) throw new Error('uvs に有限でない値がある');
  }
  return { positions, uvs, triangles, vertexCount, triangleCount };
}

/** 単一チャンネルの uint8 画像（`hair_alpha`）。 */
export interface AlphaImage {
  readonly data: Uint8Array;
  readonly width: number;
  readonly height: number;
}

/** RGB の uint8 画像（`hair_albedo`）。 */
export interface RgbImage {
  readonly data: Uint8Array;
  readonly width: number;
  readonly height: number;
}

/**
 * guest zip に入る全成果物。ファイルにする前の値。
 *
 * 髪の 3 つは **3 つ揃うか 3 つとも無いかの二択**。1 つでも欠けた組を許すと、消費側が
 * 「シェルはあるがテクスチャが無い」状態を扱う分岐を持つことになる。
 *
 * **眼球テクスチャは髪と違って欠けを許さない。** 任意にすると消費側が起こり得ない分岐を
 * 持つことになる。**左右を 2 つのフィールドに分けずに側で引ける形で持つ** — zip へ書く側も
 * 検査も側で回せる。
 */
export interface GuestArtifacts {
  readonly manifest: GuestManifest;
  /** (atlas_size, atlas_size, 3) sRGB。 */
  readonly skinAlbedo: Uint8Array;
  /** 側 → (eye_texture_size, eye_texture_size, 3) sRGB。**どちらも必ず入る**。 */
  readonly eyeAlbedos: Readonly<Record<EyeSide, Uint8Array>>;
  /** 髪シェル。髪が写っていない写真では null。 */
  readonly hair: HairShell | null;
  readonly hairAlbedo: RgbImage | null;
  readonly hairAlpha: AlphaImage | null;
}

export function makeGuestArtifacts(artifacts: GuestArtifacts): GuestArtifacts {
  const size = artifacts.manifest.atlas_size;
  if (artifacts.skinAlbedo.length !== size * size * 3) {
    throw new Error(
      `skin_albedo は uint8 の (${size}, ${size}, 3): length=${artifacts.skinAlbedo.length}`,
    );
  }
  // 正方形であること自体を別に検査しない。極座標のレイアウトは中心からの半径で決まっていて、
  // 縦横比が 1 でなければ円が楕円に潰れるので、一辺を 2 度書いた形の比較がそのまま正方形の
  // 要求になる。
  const eyeSize = artifacts.manifest.eye_texture_size;
  for (const side of EYE_SIDES) {
    const image = artifacts.eyeAlbedos[side];
    if (!image || image.length !== eyeSize * eyeSize * 3) {
      throw new Error(
        `${side} の眼球テクスチャは uint8 の (${eyeSize}, ${eyeSize}, 3):` +
          ` length=${image ? image.length : 'なし'}`,
      );
    }
  }

  const present = (
    [
      ['hair', artifacts.hair],
      ['hairAlbedo', artifacts.hairAlbedo],
      ['hairAlpha', artifacts.hairAlpha],
    ] as const
  )
    .filter(([, value]) => value !== null)
    .map(([name]) => name);
  if (present.length !== 0 && present.length !== 3) {
    throw new Error(`髪の 3 つは揃うか全て無いかの二択。今あるのは ${present} だけ`);
  }
  if (present.length === 3) {
    const albedo = artifacts.hairAlbedo as RgbImage;
    const alpha = artifacts.hairAlpha as AlphaImage;
    if (albedo.data.length !== albedo.width * albedo.height * 3) {
      throw new Error(`hair_albedo は uint8 の (H, W, 3): length=${albedo.data.length}`);
    }
    if (alpha.data.length !== alpha.width * alpha.height) {
      throw new Error(`hair_alpha は uint8 の (H, W): length=${alpha.data.length}`);
    }
    if (alpha.width !== albedo.width || alpha.height !== albedo.height) {
      throw new Error(
        `hair_alpha の形 ${alpha.width}x${alpha.height} が hair_albedo` +
          ` ${albedo.width}x${albedo.height} と揃っていない`,
      );
    }
  }
  return artifacts;
}

export function hasHair(artifacts: GuestArtifacts): boolean {
  return artifacts.hair !== null;
}

/** この成果物が zip に作るエントリ名。髪が無ければ髪系 3 つは入らない。 */
export function entryNames(artifacts: GuestArtifacts): readonly string[] {
  return hasHair(artifacts) ? [...ALWAYS_ENTRY_NAMES, ...HAIR_ENTRY_NAMES] : ALWAYS_ENTRY_NAMES;
}
