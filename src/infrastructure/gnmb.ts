// GNMB コンテナの読み書き。
//
// GNMB は "GNM Binary" — **1-10/2608_Obayashi_GNMHeadExporter が決めた形式で、公式 GNM の用語では
// ない**。やっていることは「numpy の `.npz` を C# / JS から読める形にしたもの」で、目次を JSON にして
// 本体を生の配列にしただけ。消費側の読み込みは「JSON を parse して `offset` から `byteLength` だけ
// コピー」の2手で済み、デコードも解凍も要らない。
//
// バイト配置:
//
//     magic  "GNMB"  4 bytes ASCII
//     uint32 headerLen                    little endian
//     JSON   header  headerLen bytes      UTF-8
//     payload
//
// **ヘッダの契約**（Unity 側の reader が依存する。変えることは出力仕様を変えること）:
//
//     dtype 表記        int16 / f32 / u16 / u32 / i32 / u8
//     header 直下       format / version / content / uv_origin / arrays
//     arrays の各値     offset / byteLength / dtype / shape
//
// offset は payload 先頭からの相対バイト数で、4 byte 境界に揃える。
//
// **頂点数・三角形数・成分数を申告するフィールドは持たない。** 正本は配列の shape 1つ。申告値と
// shape が食い違ったときに「どちらを信じるか」の分岐が生まれ、短い方に切って黙って壊れる余地が
// できるため。
//
// 座標変換はしない。GNM 空間（右手系 / X=解剖学的左 / Y=上 / Z=前 / メートル）のまま扱う。左手系
// への変換は消費側（Unity）の責務。
//
// 2 つの content を扱う:
//
//     head_asset  ブラウザが読む頭部アセット（`tools/export_gnm_assets.py` が書く）
//     hair_shell  guest zip に入る髪シェル（`infrastructure/packaging` が書く）
//
// **`head_asset` は web だから増えた content。** デスクトップ側は npz を直接読めるので持たない。

export const MAGIC = 'GNMB';
export const GNMB_FORMAT = 'GNMB';
export const GNMB_FORMAT_VERSION = 1;
export const PAYLOAD_ALIGNMENT = 4;

export const GNMB_CONTENT_HAIR_SHELL = 'hair_shell';
export const GNMB_CONTENT_HEAD_ASSET = 'head_asset';

/** UV の原点。v=0 が下（3D の慣習）。Unity のテクスチャ UV と同じ向き。 */
export const UV_ORIGIN = 'bottom-left';

/** dtype トークン ↔ TypedArray。header の "dtype" に出る文字列の契約の正本。 */
type DtypeToken = 'f32' | 'int16' | 'u16' | 'u32' | 'i32' | 'u8';

interface TypedArrayConstructorLike {
  new (buffer: ArrayBuffer, byteOffset: number, length: number): ArrayBufferView;
  readonly BYTES_PER_ELEMENT: number;
}

const TOKEN_TO_ARRAY: Readonly<Record<DtypeToken, TypedArrayConstructorLike>> = {
  f32: Float32Array,
  int16: Int16Array,
  u16: Uint16Array,
  u32: Uint32Array,
  i32: Int32Array,
  u8: Uint8Array,
};

const ARRAY_TO_TOKEN = new Map<Function, DtypeToken>([
  [Float32Array, 'f32'],
  [Int16Array, 'int16'],
  [Uint16Array, 'u16'],
  [Uint32Array, 'u32'],
  [Int32Array, 'i32'],
  [Uint8Array, 'u8'],
]);

export interface GnmbArrayEntry {
  readonly offset: number;
  readonly byteLength: number;
  readonly dtype: DtypeToken;
  readonly shape: readonly number[];
}

export interface GnmbHeader {
  readonly format: string;
  readonly version: number;
  readonly content: string;
  readonly uv_origin: string;
  readonly arrays: Readonly<Record<string, GnmbArrayEntry>>;
  readonly [key: string]: unknown;
}

export interface GnmbContainer {
  readonly header: GnmbHeader;
  readonly arrays: ReadonlyMap<string, ArrayBufferView>;
}

/**
 * 書き出す配列と、header に申告する shape。
 *
 * **shape を呼び出し側が渡す。** 平坦化した TypedArray からは `(Nv, 3)` と `(3Nv,)` の区別が付かない
 * が、header の shape は Unity 側の reader が読む契約なので、`(Nv, 3)` と書けなければならない。
 */
export interface GnmbArray {
  readonly array: ArrayBufferView;
  readonly shape: readonly number[];
}

/** コンテナが自分で書くヘッダのキー。種別ごとの metadata では上書きさせない。 */
const RESERVED_HEADER_KEYS = new Set(['format', 'version', 'content', 'uv_origin', 'arrays']);

/**
 * GNMB bin の全バイトを作る。
 *
 * `arrays` の並びがそのまま payload の並びになる。`metadata` は種別ごとの付加情報で、コンテナ自身が
 * 書くキーは上書きできない。
 */
export function buildGnmbContainerBytes(
  content: string,
  arrays: ReadonlyMap<string, GnmbArray>,
  metadata: Readonly<Record<string, unknown>> = {},
): Uint8Array {
  for (const key of Object.keys(metadata)) {
    if (RESERVED_HEADER_KEYS.has(key)) {
      throw new Error(`metadata がコンテナ側のキーを上書きしようとしている: ${key}`);
    }
  }
  const entries: Record<string, GnmbArrayEntry> = {};
  const chunks: Uint8Array[] = [];
  let offset = 0;
  for (const [name, { array, shape }] of arrays) {
    const token = ARRAY_TO_TOKEN.get(array.constructor);
    if (token === undefined) {
      throw new Error(`${name} の型 ${array.constructor.name} は GNMB で表せない`);
    }
    const elements = array.byteLength / TOKEN_TO_ARRAY[token].BYTES_PER_ELEMENT;
    const declared = shape.reduce((product, extent) => product * extent, 1);
    if (declared !== elements) {
      throw new Error(
        `${name} の shape ${shape.join('x')} が要素数 ${elements} と食い違っている`,
      );
    }
    const padding = ((-offset % PAYLOAD_ALIGNMENT) + PAYLOAD_ALIGNMENT) % PAYLOAD_ALIGNMENT;
    if (padding > 0) {
      chunks.push(new Uint8Array(padding));
      offset += padding;
    }
    const bytes = new Uint8Array(
      array.buffer.slice(array.byteOffset, array.byteOffset + array.byteLength),
    );
    entries[name] = { offset, byteLength: bytes.byteLength, dtype: token, shape: [...shape] };
    chunks.push(bytes);
    offset += bytes.byteLength;
  }

  const header = {
    format: GNMB_FORMAT,
    version: GNMB_FORMAT_VERSION,
    content,
    uv_origin: UV_ORIGIN,
    ...metadata,
    arrays: entries,
  };
  const headerBytes = new TextEncoder().encode(JSON.stringify(header));
  const total = 4 + 4 + headerBytes.byteLength + offset;
  const out = new Uint8Array(total);
  out.set(new TextEncoder().encode(MAGIC), 0);
  new DataView(out.buffer).setUint32(4, headerBytes.byteLength, true);
  out.set(headerBytes, 8);
  let cursor = 8 + headerBytes.byteLength;
  for (const chunk of chunks) {
    out.set(chunk, cursor);
    cursor += chunk.byteLength;
  }
  return out;
}

/** GNMB bin を header と配列に分解する。種別は呼び手が解釈する。 */
export function readGnmbContainer(
  buffer: ArrayBuffer,
  expectedContent?: string,
): GnmbContainer {
  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 4));
  if (magic !== MAGIC) throw new Error(`GNMB の magic ではない: ${magic}`);
  const headerLength = new DataView(buffer).getUint32(4, true);
  const header = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buffer, 8, headerLength)),
  ) as GnmbHeader;
  if (header.format !== GNMB_FORMAT) {
    throw new Error(`header の format が ${header.format}（期待 ${GNMB_FORMAT}）`);
  }
  if (header.version !== GNMB_FORMAT_VERSION) {
    throw new Error(`未対応の GNMB version: ${header.version}（対応 ${GNMB_FORMAT_VERSION}）`);
  }
  if (expectedContent !== undefined && header.content !== expectedContent) {
    throw new Error(`GNMB の content が ${header.content}（期待 ${expectedContent}）`);
  }
  if (header.uv_origin !== UV_ORIGIN) {
    throw new Error(`GNMB の uv_origin が ${header.uv_origin}（対応 ${UV_ORIGIN}）`);
  }

  // **payload を 1 回だけ写す。** `offset` は payload 先頭からの相対値で 4 byte 境界に揃っているが、
  // ファイル先頭からの絶対 offset は header の長さぶんずれるので TypedArray の要求（要素サイズに
  // 揃った byteOffset）を満たさない。payload だけを新しい ArrayBuffer へ写せば相対 offset がそのまま
  // 使える（Python 側は `np.frombuffer(blob[start:stop])` で同じことをしている）。
  const payload = buffer.slice(8 + headerLength);
  const arrays = new Map<string, ArrayBufferView>();
  for (const [name, entry] of Object.entries(header.arrays)) {
    const constructor = TOKEN_TO_ARRAY[entry.dtype];
    if (constructor === undefined) throw new Error(`未知の dtype: ${entry.dtype}`);
    const elements = entry.byteLength / constructor.BYTES_PER_ELEMENT;
    if (!Number.isInteger(elements)) {
      throw new Error(`${name} の byteLength ${entry.byteLength} が要素サイズで割り切れない`);
    }
    if (entry.offset % PAYLOAD_ALIGNMENT !== 0) {
      throw new Error(`${name} の offset ${entry.offset} が ${PAYLOAD_ALIGNMENT} byte 境界でない`);
    }
    arrays.set(name, new constructor(payload, entry.offset, elements));
  }
  return { header, arrays };
}

/** 指定した名前の配列を型ごと取り出す（無ければ落とす）。 */
export function requireArray<T extends ArrayBufferView>(
  container: GnmbContainer,
  name: string,
  constructor: new (...args: never[]) => T,
): T {
  const array = container.arrays.get(name);
  if (array === undefined) throw new Error(`GNMB に配列 ${name} が無い`);
  if (!(array instanceof constructor)) {
    throw new Error(`${name} の型が ${array.constructor.name}（期待 ${constructor.name}）`);
  }
  return array;
}
