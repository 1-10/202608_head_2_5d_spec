// guest zip の書き出し。
//
// 出力契約の値を `guest_<YYYYMMDDhhmmss>.zip` にまとめる:
//
//     guest.json / skin_albedo.jpg / left_eye_albedo.png / right_eye_albedo.png
//     hair_shell.bin / hair_albedo.jpg / hair_alpha.png（髪が写っていれば）
//
// Unity 側が名前で引くので、エントリ名は契約そのもの（名前の正本は `domain/contract`）。
//
// **一時ファイル → rename はブラウザには無い。** デスクトップ側は書き出し中に落ちたときに壊れた zip が
// 残らないよう一時名で書いてから差し替えるが、ブラウザは Blob を作り終えてから初めてダウンロードが
// 始まるので、**「最終名のファイルが現れた時点で必ず完成している」が構造的に満たされる**。web だから
// 消える段。

import {
  EYE_ALBEDO_NAMES,
  GuestArtifacts,
  HAIR_ALBEDO_NAME,
  HAIR_ALPHA_NAME,
  HAIR_SHELL_NAME,
  MANIFEST_NAME,
  SKIN_ALBEDO_NAME,
  UV_ORIGIN as CONTRACT_UV_ORIGIN,
  entryNames,
  manifestToJson,
  zipNameOf,
} from '../domain/contract';
import { EYE_SIDES } from '../domain/eyes/layout';
import {
  GNMB_CONTENT_HAIR_SHELL,
  GnmbArray,
  UV_ORIGIN,
  buildGnmbContainerBytes,
} from './gnmb';
import { encodeAlphaPng, encodeJpeg, encodePng } from './imaging';

/**
 * 中身の性質で圧縮を選ぶ。
 *
 * JPEG / PNG は既に圧縮済みで deflate が効かない（実測で 1% 未満）ので、CPU を使うだけになる。
 * JSON と GNMB の float 配列は効く。眼球テクスチャも PNG なので既定（無圧縮）側。
 */
const DEFLATED_ENTRIES = new Set<string>([MANIFEST_NAME, HAIR_SHELL_NAME]);

export interface GuestZipEntry {
  readonly name: string;
  readonly bytes: Uint8Array;
}

/**
 * zip に入れる `(エントリ名, バイト列)` を入れる順で返す。
 *
 * zip を書く手前で切っているのは、書いたものを読み直さずに中身を検証できるようにするため
 * （テストが zip の I/O を経由せずに済む）。
 */
export async function buildGuestEntries(
  artifacts: GuestArtifacts,
): Promise<readonly GuestZipEntry[]> {
  // UV の原点は guest.json と GNMB ヘッダの両方に出る。片方だけ変えたらここで落ちる。
  if (UV_ORIGIN !== CONTRACT_UV_ORIGIN) {
    throw new Error(
      `GNMB コンテナの uv_origin ${UV_ORIGIN} が出力契約の ${CONTRACT_UV_ORIGIN} と食い違っている`,
    );
  }

  const size = artifacts.manifest.atlas_size;
  const entries: GuestZipEntry[] = [
    { name: MANIFEST_NAME, bytes: new TextEncoder().encode(manifestToJson(artifacts.manifest)) },
    {
      name: SKIN_ALBEDO_NAME,
      bytes: await encodeJpeg({ data: artifacts.skinAlbedo, width: size, height: size }),
    },
  ];
  // 眼球は PNG（可逆）。JPEG にすると虹彩の模様と瞳孔の縁で量子化がリンギングを出す。並びは
  // `EYE_SIDES` の順（`ALWAYS_ENTRY_NAMES` もそれで組まれているので、食い違えば下の検査が落ちる）。
  const eyeSize = artifacts.manifest.eye_texture_size;
  for (const side of EYE_SIDES) {
    entries.push({
      name: EYE_ALBEDO_NAMES[side],
      bytes: await encodePng({
        data: artifacts.eyeAlbedos[side],
        width: eyeSize,
        height: eyeSize,
      }),
    });
  }

  if (artifacts.hair !== null) {
    if (artifacts.hairAlbedo === null || artifacts.hairAlpha === null) {
      throw new Error('髪シェルがあるのにテクスチャが無い');
    }
    const { vertexCount, triangleCount } = artifacts.hair;
    const arrays = new Map<string, GnmbArray>([
      ['positions', { array: artifacts.hair.positions, shape: [vertexCount, 3] }],
      ['uvs', { array: artifacts.hair.uvs, shape: [vertexCount, 2] }],
      ['triangles', { array: artifacts.hair.triangles, shape: [triangleCount, 3] }],
    ]);
    entries.push(
      {
        name: HAIR_SHELL_NAME,
        bytes: buildGnmbContainerBytes(GNMB_CONTENT_HAIR_SHELL, arrays),
      },
      { name: HAIR_ALBEDO_NAME, bytes: await encodeJpeg(artifacts.hairAlbedo) },
      { name: HAIR_ALPHA_NAME, bytes: await encodeAlphaPng(artifacts.hairAlpha) },
    );
  }

  const expected = entryNames(artifacts);
  const actual = entries.map((entry) => entry.name);
  if (actual.length !== expected.length || actual.some((name, index) => name !== expected[index])) {
    throw new Error(
      `書き出すエントリが entryNames() と食い違っている: ${actual} / ${expected}`,
    );
  }
  return entries;
}

/** guest zip の Blob とファイル名を作る。 */
export async function buildGuestZip(
  artifacts: GuestArtifacts,
): Promise<{ blob: Blob; filename: string }> {
  const entries = await buildGuestEntries(artifacts);
  const { zipSync } = await import('fflate');
  const files: Record<string, [Uint8Array, { level: 0 | 6 }]> = {};
  for (const entry of entries) {
    files[entry.name] = [entry.bytes, { level: DEFLATED_ENTRIES.has(entry.name) ? 6 : 0 }];
  }
  const zipped = zipSync(files);
  return {
    blob: new Blob([zipped as BlobPart], { type: 'application/zip' }),
    filename: zipNameOf(artifacts.manifest),
  };
}
