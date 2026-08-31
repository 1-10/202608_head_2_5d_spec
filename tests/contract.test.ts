// 出力契約（`domain/contract`）の検査。
//
// **契約は「消費側が実際に読むもの」だけを載せる**という約束を、型が自分で守っていることを見る。
// キーの集合・版・向き・色空間は Unity 側の reader が依存するので、ここが落ちたら出力仕様が変わって
// いる。

import { describe, expect, it } from 'vitest';
import {
  COLOR_SPACE,
  FORMAT_VERSION,
  UV_ORIGIN,
  atlasRowColToUv,
  createGuestManifest,
  entryNames,
  hairShellFromImageUv,
  makeGuestArtifacts,
  manifestFromJson,
  manifestToJson,
  zipNameOf,
} from '../src/domain/contract';

function manifest(identityCount = 4): ReturnType<typeof createGuestManifest> {
  return createGuestManifest({
    identity: new Float64Array(identityCount).fill(0.5),
    gnmVersion: '3.0',
    gnmVariant: 'head',
    atlasSize: 8,
    eyeTextureSize: 4,
    capturedAt: new Date(2026, 7, 31, 12, 34, 56),
    exporterVersion: '0.2.0',
  });
}

function artifacts(): ReturnType<typeof makeGuestArtifacts> {
  const value = manifest();
  return makeGuestArtifacts({
    manifest: value,
    skinAlbedo: new Uint8Array(8 * 8 * 3),
    eyeAlbedos: { left: new Uint8Array(4 * 4 * 3), right: new Uint8Array(4 * 4 * 3) },
    hair: null,
    hairAlbedo: null,
    hairAlpha: null,
  });
}

describe('guest.json', () => {
  it('自分で決まるフィールドは呼び出し側が選べない', () => {
    const value = manifest();
    expect(value.format_version).toBe(FORMAT_VERSION);
    expect(value.uv_origin).toBe(UV_ORIGIN);
    expect(value.color_space).toBe(COLOR_SPACE);
    expect(value.identity_count).toBe(value.identity.length);
  });

  it('JSON へ書いて読み戻せる（キーの並びは契約の並び）', () => {
    const value = manifest();
    const text = manifestToJson(value);
    expect(Object.keys(JSON.parse(text) as Record<string, unknown>)).toEqual([
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
    ]);
    expect(manifestFromJson(JSON.parse(text) as Record<string, unknown>)).toEqual(value);
  });

  it('版が違う guest.json は「版が違う」と言って落ちる（キーの余りではなく）', () => {
    const values = JSON.parse(manifestToJson(manifest())) as Record<string, unknown>;
    values['format_version'] = 1;
    expect(() => manifestFromJson(values)).toThrow(/format_version/);
  });

  it('キーが欠けても余っても落ちる', () => {
    const values = JSON.parse(manifestToJson(manifest())) as Record<string, unknown>;
    delete values['atlas_size'];
    expect(() => manifestFromJson(values)).toThrow(/キーが合わない/);
    const extra = JSON.parse(manifestToJson(manifest())) as Record<string, unknown>;
    extra['skin_base_color'] = [0.5, 0.4, 0.3];
    expect(() => manifestFromJson(extra)).toThrow(/キーが合わない/);
  });

  it('identity が空だと落ちる（フィットの結果が入っていない）', () => {
    expect(() =>
      createGuestManifest({
        identity: new Float64Array(0),
        gnmVersion: '3.0',
        gnmVariant: 'head',
        atlasSize: 8,
        eyeTextureSize: 4,
        capturedAt: new Date(),
        exporterVersion: '0.2.0',
      }),
    ).toThrow(/identity が空/);
  });

  it('zip の名前は captured_at から作る（別に時計を読まない）', () => {
    expect(zipNameOf(manifest())).toBe('guest_20260831123456.zip');
  });
});

describe('skin_albedo の行と UV の対応', () => {
  it('行 0 が v = 1 側', () => {
    expect(atlasRowColToUv(0, 0, 4)).toEqual([0.125, 0.875]);
    expect(atlasRowColToUv(3, 3, 4)).toEqual([0.875, 0.125]);
  });
});

describe('GuestArtifacts', () => {
  it('髪が無ければ髪系 3 つは zip に入らない', () => {
    expect(entryNames(artifacts())).toEqual([
      'guest.json',
      'skin_albedo.jpg',
      'left_eye_albedo.png',
      'right_eye_albedo.png',
    ]);
  });

  it('髪の 3 つは揃うか全て無いかの二択', () => {
    const shell = hairShellFromImageUv(
      Float32Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      Float32Array.from([0, 0, 1, 0, 0, 1]),
      Uint32Array.from([0, 1, 2]),
    );
    expect(() =>
      makeGuestArtifacts({
        ...artifacts(),
        hair: shell,
        hairAlbedo: null,
        hairAlpha: null,
      }),
    ).toThrow(/揃うか全て無いか/);
  });

  it('眼球テクスチャの一辺が manifest と食い違ったら落ちる', () => {
    expect(() =>
      makeGuestArtifacts({
        ...artifacts(),
        eyeAlbedos: { left: new Uint8Array(3 * 3 * 3), right: new Uint8Array(4 * 4 * 3) },
      }),
    ).toThrow(/眼球テクスチャ/);
  });
});

describe('hair_shell の UV', () => {
  it('画像 UV 空間（v 下向き）から bottom-left へ v を反転する', () => {
    const shell = hairShellFromImageUv(
      Float32Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      Float32Array.from([0.25, 0.25, 1, 0, 0, 1]),
      Uint32Array.from([0, 1, 2]),
    );
    expect(Array.from(shell.uvs)).toEqual([0.25, 0.75, 1, 1, 0, 0]);
  });

  it('空の髪シェルは作れない（髪が無い写真では null にする）', () => {
    expect(() =>
      hairShellFromImageUv(new Float32Array(0), new Float32Array(0), new Uint32Array(0)),
    ).toThrow(/空の髪シェル/);
  });
});
