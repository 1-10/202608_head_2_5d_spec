// 3D ビュー用のデータと計算の検査。
//
// **Unity 側（1-10/2607_Obayashi_Avatar_Mockup_3DGS の `Assets/Sandbox/Ooba/GNM`）が正本**で、あちらの
// README が実データで確認済みだと書いている前提を、こちらの実アセットでも測る:
//
// - 既定の領域設定で全三角形が漏れなく割り当たる（角膜を除外した残りに未割り当てゼロ）
// - `mouth_sock` は `skin` の部分集合なので、順序を逆にすると Skin が全部取る
// - 重み和はちょうど 1 / 影響ボーンは最大 2 本
// - ジョイント位置は identity で動く（実測 7mm 程度）
//
// 姿勢と表情の計算そのものは純粋関数なので、アセット無しでも回る形にしてある。

import { describe, expect, it } from 'vitest';
import { loadBundle, loadPreview } from './asset';
import {
  EXCLUDED_SELECTOR,
  GnmPreviewAsset,
  PREVIEW_REGIONS,
  UNASSIGNED_REGION,
  classifyTriangles,
  evaluateSelector,
} from '../src/domain/preview/asset';
import {
  BLINK_PRESET_NAMES,
  FADE_SECONDS,
  HOLD_SECONDS,
  addExpression,
  advanceBlink,
  advancePlayback,
  envelope,
  startBlink,
  weightsFor,
} from '../src/domain/preview/expression';
import {
  GAZE_LIMIT_DEGREES,
  NECK_SHARE,
  PITCH_LIMIT_DEGREES,
  YAW_LIMIT_DEGREES,
  clampPose,
  followPointerPose,
  jointLocalRotations,
  jointRestPositions,
  jointSkinMatrices,
  skinVertices,
} from '../src/domain/preview/pose';
import {
  buildPreviewScene,
  keepRealNormalMask,
} from '../src/domain/preview/scene';
import {
  flattenNormals,
  planNormals,
  recalculateNormals,
  skinNormals,
} from '../src/domain/preview/normals';
import { unsplitVertexCount, verticesOf } from '../src/domain/gnm/model';

/** 全頂点で実法線を求める plan（絞り込み前の答え合わせ用）。 */
function allReal(vertexCount: number): Uint8Array {
  return new Uint8Array(vertexCount).fill(1);
}

function countMask(mask: Uint8Array): number {
  let total = 0;
  for (const value of mask) if (value !== 0) total++;
  return total;
}

describe('vertex group と領域分け', () => {
  it('必要な group が全部ある', () => {
    const preview = loadPreview();
    const required = [
      ...PREVIEW_REGIONS.flatMap((region) => region.selector),
      ...EXCLUDED_SELECTOR,
    ].map((token) => token.replace(/^[|&-]/, '').replace(/^~/, ''));
    for (const name of required) {
      expect(preview.vertexGroupNames, name).toContain(name);
    }
  });

  it('mouth_sock は skin の部分集合（だから Skin より前に置く）', () => {
    const preview = loadPreview();
    const sock = evaluateSelector(preview, ['mouth_sock']);
    const skin = evaluateSelector(preview, ['skin']);
    for (let vertex = 0; vertex < preview.vertexCount; vertex++) {
      if (sock[vertex] !== 0) expect(skin[vertex], `頂点 ${vertex}`).not.toBe(0);
    }
    expect(countMask(sock)).toBeGreaterThan(0);
  });

  it('全三角形が漏れなく割り当たる（角膜を除いた残りに未割り当てゼロ）', () => {
    const { asset, preview } = loadBundle();
    const result = classifyTriangles(preview, asset.mesh.triangles, asset.mesh.triangleCount);
    expect(result.regions.map((region) => region.name)).toEqual(
      PREVIEW_REGIONS.map((region) => region.name),
    );
    expect(result.regions).not.toContain(UNASSIGNED_REGION);
    expect(result.excludedCount).toBeGreaterThan(0);
    const assigned = result.perRegion.reduce((total, list) => total + list.length, 0);
    expect(assigned + result.excludedCount).toBe(asset.mesh.triangleCount);
  });

  it('Skin を MouthSock より前に置くと口腔壁が Skin に飲まれる', () => {
    const { asset, preview } = loadBundle();
    const sock = PREVIEW_REGIONS.find((region) => region.name === 'MouthSock');
    const skin = PREVIEW_REGIONS.find((region) => region.name === 'Skin');
    if (sock === undefined || skin === undefined) throw new Error('領域の定義が変わっている');
    const correct = classifyTriangles(preview, asset.mesh.triangles, asset.mesh.triangleCount, [
      sock,
      skin,
    ]);
    const swapped = classifyTriangles(preview, asset.mesh.triangles, asset.mesh.triangleCount, [
      skin,
      sock,
    ]);
    expect(correct.perRegion[0].length).toBeGreaterThan(0);
    // 逆順では MouthSock（2 番目）に 1 つも入らない。
    expect(swapped.perRegion[1].length).toBe(0);
  });

  it('口腔内は写真テクスチャを読まず、Unity の Material と同じ固定色で塗る', () => {
    const colors = new Map(
      PREVIEW_REGIONS.map((region) => [region.name, region.color.join(',')]),
    );
    // 正本: MT_GnmMouthSock / MT_GnmTeeth / MT_GnmGums / MT_GnmTongue の _BaseColor。
    expect(colors.get('MouthSock')).toBe('80,37,37');
    expect(colors.get('Teeth')).toBe('190,164,164');
    expect(colors.get('Gums')).toBe('114,53,53');
    expect(colors.get('Tongue')).toBe('114,53,53');
    for (const region of PREVIEW_REGIONS) {
      if (region.name === 'Skin') expect(region.kind).toBe('skin_texture');
      else if (region.name.startsWith('Eye')) expect(region.kind).not.toBe('flat_color');
      else expect(region.kind).toBe('flat_color');
    }
  });
});

describe('スキニングとジョイント', () => {
  it('重み和はちょうど 1 で、影響ボーンは 2 本以内', () => {
    const preview = loadPreview();
    for (let vertex = 0; vertex < preview.vertexCount; vertex++) {
      const sum = preview.skinJointWeights[vertex * 2] + preview.skinJointWeights[vertex * 2 + 1];
      expect(sum, `頂点 ${vertex}`).toBeCloseTo(1, 5);
    }
  });

  it('ジョイント位置は identity で動く（テンプレのままだと bind pose がずれる）', () => {
    const { asset, preview } = loadBundle();
    const identity = new Float64Array(asset.vertexIdentityBasis.componentCount);
    identity[0] = 3;
    const rest = jointRestPositions(preview, identity);
    let moved = 0;
    for (let index = 0; index < rest.length; index++) {
      moved = Math.max(moved, Math.abs(rest[index] - preview.templateJointPositions[index]));
    }
    expect(moved).toBeGreaterThan(0);
    // 平均顔ならテンプレートそのまま。
    const zero = jointRestPositions(preview, new Float64Array(identity.length));
    for (let index = 0; index < zero.length; index++) {
      expect(zero[index]).toBeCloseTo(preview.templateJointPositions[index], 10);
    }
  });

  it('無回転のスキニングは恒等（pose correctives が全ゼロという前提の裏返し）', () => {
    const { asset, preview } = loadBundle();
    const identity = new Float64Array(asset.vertexIdentityBasis.componentCount);
    const vertices = verticesOf(asset, identity);
    const rest = jointRestPositions(preview, identity);
    const matrices = jointSkinMatrices(
      preview,
      rest,
      jointLocalRotations(preview, {
        headYawDegrees: 0,
        headPitchDegrees: 0,
        gazeYawDegrees: 0,
        gazePitchDegrees: 0,
      }),
    );
    const skinned = skinVertices(preview, vertices, matrices);
    let maximum = 0;
    for (let index = 0; index < vertices.length; index++) {
      maximum = Math.max(maximum, Math.abs(skinned[index] - vertices[index]));
    }
    // 厳密な 0 にはならない。重みは float32 なので、2 本の和を float64 で足すと 1 から
    // 1e-8 程度ずれる。頭部の寸法 0.3m に掛かって数 nm 残る（実測 6.4nm）。
    expect(maximum).toBeLessThan(1e-7);
  });

  it('首を右へ振ると鼻先が解剖学的な右（GNM の -X）へ動く', () => {
    const { asset, preview } = loadBundle();
    const identity = new Float64Array(asset.vertexIdentityBasis.componentCount);
    const vertices = verticesOf(asset, identity);
    const rest = jointRestPositions(preview, identity);
    // 最も前（+Z）にある頂点を鼻先の代わりに使う。
    let nose = 0;
    for (let vertex = 1; vertex < preview.vertexCount; vertex++) {
      if (vertices[vertex * 3 + 2] > vertices[nose * 3 + 2]) nose = vertex;
    }
    const turned = skinVertices(
      preview,
      vertices,
      jointSkinMatrices(
        preview,
        rest,
        jointLocalRotations(preview, {
          headYawDegrees: YAW_LIMIT_DEGREES,
          headPitchDegrees: 0,
          gazeYawDegrees: 0,
          gazePitchDegrees: 0,
        }),
      ),
    );
    expect(turned[nose * 3]).toBeLessThan(vertices[nose * 3] - 1e-3);

    const down = skinVertices(
      preview,
      vertices,
      jointSkinMatrices(
        preview,
        rest,
        jointLocalRotations(preview, {
          headYawDegrees: 0,
          headPitchDegrees: PITCH_LIMIT_DEGREES,
          gazeYawDegrees: 0,
          gazePitchDegrees: 0,
        }),
      ),
    );
    // 正の pitch で下を向く（Unity 側の Quaternion.Euler(pitch, yaw, 0) と同じ向き）。
    expect(down[nose * 3 + 1]).toBeLessThan(vertices[nose * 3 + 1] - 1e-3);
  });

  it('首と頭で角度を分ける（配分は Unity 側と同じ既定）', () => {
    const preview = loadPreview();
    const rotations = jointLocalRotations(preview, {
      headYawDegrees: 10,
      headPitchDegrees: 0,
      gazeYawDegrees: 0,
      gazePitchDegrees: 0,
    });
    const angleOf = (joint: number): number =>
      Math.abs(Math.atan2(rotations[joint * 9 + 2], rotations[joint * 9]));
    const neck = preview.jointNames.indexOf('neck');
    const head = preview.jointNames.indexOf('head');
    expect(angleOf(neck)).toBeCloseTo((10 * NECK_SHARE * Math.PI) / 180, 8);
    expect(angleOf(head)).toBeCloseTo((10 * (1 - NECK_SHARE) * Math.PI) / 180, 8);
  });

  it('可動域でクランプする', () => {
    const clamped = clampPose({
      headYawDegrees: 90,
      headPitchDegrees: -90,
      gazeYawDegrees: 90,
      gazePitchDegrees: -90,
    });
    expect(clamped.headYawDegrees).toBe(YAW_LIMIT_DEGREES);
    expect(clamped.headPitchDegrees).toBe(-PITCH_LIMIT_DEGREES);
    expect(clamped.gazeYawDegrees).toBe(GAZE_LIMIT_DEGREES);
    expect(clamped.gazePitchDegrees).toBe(-GAZE_LIMIT_DEGREES);
  });

  it('マウス追従は視線が首より速く動く（首だけ平滑化する）', () => {
    const start = clampPose({
      headYawDegrees: 0,
      headPitchDegrees: 0,
      gazeYawDegrees: 0,
      gazePitchDegrees: 0,
    });
    const next = followPointerPose(start, 1, 0, 1 / 60);
    expect(next.gazeYawDegrees).toBe(GAZE_LIMIT_DEGREES);
    expect(next.headYawDegrees).toBeGreaterThan(0);
    expect(next.headYawDegrees).toBeLessThan(YAW_LIMIT_DEGREES);
  });
});

describe('表情プリセット', () => {
  it('Unity 側と同じ 20 本で、まばたきに使う名前がある', () => {
    const preview = loadPreview();
    expect(preview.presetCount).toBe(20);
    for (const name of BLINK_PRESET_NAMES) {
      expect(preview.expressionPresetNames).toContain(name);
    }
  });

  it('重みを立てると顔が動き、0 に戻すと元へ戻る', () => {
    const { asset, preview } = loadBundle();
    const identity = new Float64Array(asset.vertexIdentityBasis.componentCount);
    const base = verticesOf(asset, identity);
    const moved = Float64Array.from(base);
    addExpression(preview, moved, weightsFor(preview, [['smile_wide', 1]]));
    let maximum = 0;
    for (let index = 0; index < base.length; index++) {
      maximum = Math.max(maximum, Math.abs(moved[index] - base[index]));
    }
    // Unity 側の実測（weight 1.0 で最大 1.2cm 程度）と同じ桁。
    expect(maximum).toBeGreaterThan(0.002);
    expect(maximum).toBeLessThan(0.03);

    const zeroed = Float64Array.from(base);
    addExpression(preview, zeroed, new Float64Array(preview.presetCount));
    expect(zeroed).toEqual(base);
  });

  it('知らないプリセット名は落とす（黙って無表情にしない）', () => {
    const preview = loadPreview();
    expect(() => weightsFor(preview, [['grin', 1]])).toThrow(/grin/);
  });

  it('台形エンベロープは 0 → 1 → 0', () => {
    const cycle = FADE_SECONDS * 2 + HOLD_SECONDS;
    expect(envelope(0)).toBeCloseTo(0, 10);
    expect(envelope(FADE_SECONDS)).toBeCloseTo(1, 10);
    expect(envelope(FADE_SECONDS + HOLD_SECONDS / 2)).toBeCloseTo(1, 10);
    expect(envelope(cycle)).toBeCloseTo(0, 10);
  });

  it('自動再生は 1 本ずつで、順番モードは申告順に回る', () => {
    let playback = { index: -1, elapsedSeconds: 0 };
    const seen: number[] = [];
    const cycle = FADE_SECONDS * 2 + HOLD_SECONDS;
    for (let step = 0; step < 5; step++) {
      const result = advancePlayback(playback, 'sequence', 3, cycle);
      playback = result.playback;
      seen.push(result.index);
    }
    expect(seen).toEqual([0, 1, 2, 0, 1]);
  });

  it('ランダムモードは直前と同じものを引かない', () => {
    let playback = { index: -1, elapsedSeconds: 0 };
    const cycle = FADE_SECONDS * 2 + HOLD_SECONDS;
    let previous = -1;
    // 乱数を全域に振って、どの引きでも直前と同じにならないことを見る。
    for (let step = 0; step < 40; step++) {
      const result = advancePlayback(playback, 'random', 4, cycle, () => (step % 20) / 20);
      playback = result.playback;
      if (previous >= 0) expect(result.index).not.toBe(previous);
      previous = result.index;
    }
  });

  it('off では何も立てない', () => {
    const result = advancePlayback({ index: 2, elapsedSeconds: 0.1 }, 'off', 20, 0.016);
    expect(result.index).toBe(-1);
    expect(result.weight).toBe(0);
  });

  it('まばたきは待ってから閉じ、閉じ切りで留まらない', () => {
    let state = startBlink(() => 0);
    let closed = 0;
    let peak = 0;
    // 周期の下限 3 秒 + まばたき 1 回ぶんを 1/60 秒刻みで回す。
    for (let step = 0; step < 60 * 4; step++) {
      const result = advanceBlink(state, 1 / 60, () => 0);
      state = result.state;
      if (result.weight > 0) closed++;
      peak = Math.max(peak, result.weight);
    }
    expect(closed).toBeGreaterThan(0);
    expect(peak).toBeGreaterThan(0.9);
    expect(peak).toBeLessThanOrEqual(1);
  });
});

describe('確認用シーン', () => {
  it('領域ごとにメッシュが立ち、口腔壁は写真テクスチャを持たない', () => {
    const { asset, preview } = loadBundle();
    const identity = new Float64Array(asset.vertexIdentityBasis.componentCount);
    const vertices = verticesOf(asset, identity);
    const scene = buildPreviewScene({
      vertices,
      headMesh: asset.mesh,
      preview,
      skinAlbedo: { data: new Uint8Array(4 * 4 * 3), width: 4, height: 4 },
      eyeAlbedos: {
        left: { data: new Uint8Array(4 * 4 * 3), width: 4, height: 4 },
        right: { data: new Uint8Array(4 * 4 * 3), width: 4, height: 4 },
      },
      hair: null,
      hairAlbedo: null,
      hairAlpha: null,
    });
    expect(scene.unassignedTriangleCount).toBe(0);
    expect(scene.excludedTriangleCount).toBeGreaterThan(0);
    const byName = new Map(scene.meshes.map((mesh) => [mesh.name, mesh]));
    expect([...byName.keys()]).toEqual(PREVIEW_REGIONS.map((region) => region.name));
    expect(byName.get('Skin')?.texture).not.toBeNull();
    expect(byName.get('MouthSock')?.texture).toBeNull();
    expect(byName.get('Teeth')?.texture).toBeNull();
    expect(byName.get('EyeLeft')?.texture).not.toBeNull();
    // 頭部の各メッシュは split 頂点配列への index を持つ（毎フレームここから集める）。
    for (const mesh of scene.meshes) {
      expect(mesh.sourceVertices).not.toBeNull();
      expect(mesh.sourceVertices?.length).toBe(mesh.restPositions.length / 3);
    }
  });

  it('角膜は描かない（Unity 側と同じ除外設定）', () => {
    const { asset, preview } = loadBundle();
    const excluded = evaluateSelector(preview, EXCLUDED_SELECTOR);
    expect(countMask(excluded)).toBeGreaterThan(0);
    const result = classifyTriangles(preview, asset.mesh.triangles, asset.mesh.triangleCount);
    for (const list of result.perRegion) {
      for (const triangle of list) {
        const all =
          excluded[asset.mesh.triangles[triangle * 3]] !== 0 &&
          excluded[asset.mesh.triangles[triangle * 3 + 1]] !== 0 &&
          excluded[asset.mesh.triangles[triangle * 3 + 2]] !== 0;
        expect(all).toBe(false);
      }
    }
  });
});

describe('selector の構文', () => {
  const preview = (): GnmPreviewAsset => loadPreview();

  it('& は積、- は差', () => {
    const asset = preview();
    const interiors = evaluateSelector(asset, ['eye_interiors']);
    const left = evaluateSelector(asset, ['left_eye']);
    const intersection = evaluateSelector(asset, ['eye_interiors', '&left_eye']);
    const difference = evaluateSelector(asset, ['eye_interiors', '-left_eye']);
    for (let vertex = 0; vertex < asset.vertexCount; vertex++) {
      const both = interiors[vertex] !== 0 && left[vertex] !== 0;
      expect(intersection[vertex] !== 0).toBe(both);
      expect(difference[vertex] !== 0).toBe(interiors[vertex] !== 0 && left[vertex] === 0);
    }
    expect(countMask(intersection)).toBeGreaterThan(0);
    expect(countMask(difference)).toBeGreaterThan(0);
  });

  it('知らない group 名は落とす', () => {
    expect(() => evaluateSelector(preview(), ['eyebrows'])).toThrow(/eyebrows/);
  });
});

// 正本は Unity 側 `GnmHeadInstance.FlattenNormals`。あちらのセッション（ccdesk 868eacc0）に確認した
// 切り分けと同じであることを実アセットで測る:
//
// | 部位 | リアル表示の法線 |
// |:--|:--|
// | 肌・眼球・髪 | +Z 固定 |
// | 歯・歯茎・舌 | 実法線 |
// | 口腔壁の奥 | 実法線 |
// | 口腔壁のうち肌と共有する頂点（唇の内縁） | **+Z 固定** |
describe('法線の切り分け', () => {
  function masks(): {
    keepReal: Uint8Array;
    vertexOf: (name: string) => Set<number>;
  } {
    const { asset, preview } = loadBundle();
    const classification = classifyTriangles(
      preview,
      asset.mesh.triangles,
      asset.mesh.triangleCount,
    );
    const keepReal = keepRealNormalMask(
      asset.mesh.vertexCount,
      asset.mesh.triangles,
      classification,
    );
    const vertexOf = (name: string): Set<number> => {
      const index = classification.regions.findIndex((region) => region.name === name);
      if (index < 0) throw new Error(`領域 ${name} が無い`);
      const set = new Set<number>();
      for (const triangle of classification.perRegion[index]) {
        for (let corner = 0; corner < 3; corner++) {
          set.add(asset.mesh.triangles[triangle * 3 + corner]);
        }
      }
      return set;
    };
    return { keepReal, vertexOf };
  }

  it('歯・歯茎・舌は全頂点が実法線', () => {
    const { keepReal, vertexOf } = masks();
    for (const name of ['Teeth', 'Gums', 'Tongue']) {
      for (const vertex of vertexOf(name)) {
        expect(keepReal[vertex], `${name} の頂点 ${vertex}`).toBe(1);
      }
    }
  });

  it('肌・眼球は全頂点が +Z 固定', () => {
    const { keepReal, vertexOf } = masks();
    for (const name of ['Skin', 'EyeLeft', 'EyeRight']) {
      for (const vertex of vertexOf(name)) {
        expect(keepReal[vertex], `${name} の頂点 ${vertex}`).toBe(0);
      }
    }
  });

  it('口腔壁は奥が実法線・唇の内縁だけ +Z 固定（肌と共有する頂点が後勝ちで落ちる）', () => {
    const { keepReal, vertexOf } = masks();
    const sock = [...vertexOf('MouthSock')];
    const real = sock.filter((vertex) => keepReal[vertex] === 1);
    const flat = sock.filter((vertex) => keepReal[vertex] === 0);
    // 両方が非空でなければ「二段マーク」が効いていない。
    expect(real.length).toBeGreaterThan(0);
    expect(flat.length).toBeGreaterThan(0);
    // 落ちるのは境界の一周ぶんなので、奥の方が多い。
    expect(real.length).toBeGreaterThan(flat.length);
  });

  it('実法線は外向き（巻き順が合っている）', () => {
    const { asset, preview } = loadBundle();
    const identity = new Float64Array(asset.vertexIdentityBasis.componentCount);
    const vertices = verticesOf(asset, identity);
    const normals = new Float32Array(vertices.length);
    const undetermined = recalculateNormals(
      vertices,
      asset.mesh.triangles,
      asset.mesh.uvSplitSource,
      planNormals(asset.mesh.triangles, allReal(asset.mesh.vertexCount), unsplitVertexCount(asset.mesh)),
      normals,
    );
    expect(undetermined).toBe(0);
    // 単位長であること（面積重みだとここが 1e-5 まで落ちて潰れる）。
    for (let vertex = 0; vertex < preview.vertexCount; vertex++) {
      const length = Math.hypot(
        normals[vertex * 3],
        normals[vertex * 3 + 1],
        normals[vertex * 3 + 2],
      );
      expect(length, `頂点 ${vertex}`).toBeCloseTo(1, 4);
    }
    // 最も前（+Z）の頂点は前を向く。裏返っていたら符号が反転する。
    let front = 0;
    let back = 0;
    for (let vertex = 1; vertex < preview.vertexCount; vertex++) {
      if (vertices[vertex * 3 + 2] > vertices[front * 3 + 2]) front = vertex;
      if (vertices[vertex * 3 + 2] < vertices[back * 3 + 2]) back = vertex;
    }
    expect(normals[front * 3 + 2]).toBeGreaterThan(0.5);
    expect(normals[back * 3 + 2]).toBeLessThan(-0.5);
  });

  it('+Z へ落とした後も口腔内だけは実法線が残る', () => {
    const { asset, preview } = loadBundle();
    const identity = new Float64Array(asset.vertexIdentityBasis.componentCount);
    const vertices = verticesOf(asset, identity);
    const keepReal = keepRealNormalMask(
      asset.mesh.vertexCount,
      asset.mesh.triangles,
      classifyTriangles(preview, asset.mesh.triangles, asset.mesh.triangleCount),
    );
    // 本番と同じ順序: +Z で埋めてから、plan が挙げた頂点だけ実法線で上書きし、最後に落とす。
    const normals = new Float32Array(vertices.length);
    for (let vertex = 0; vertex < preview.vertexCount; vertex++) normals[vertex * 3 + 2] = 1;
    recalculateNormals(
      vertices,
      asset.mesh.triangles,
      asset.mesh.uvSplitSource,
      planNormals(asset.mesh.triangles, keepReal, unsplitVertexCount(asset.mesh)),
      normals,
    );
    flattenNormals(normals, keepReal);
    let flatCount = 0;
    let tilted = 0;
    for (let vertex = 0; vertex < preview.vertexCount; vertex++) {
      const isFlat =
        normals[vertex * 3] === 0 && normals[vertex * 3 + 1] === 0 && normals[vertex * 3 + 2] === 1;
      if (keepReal[vertex] === 0) {
        expect(isFlat, `頂点 ${vertex} は +Z のはず`).toBe(true);
        flatCount++;
      } else if (!isFlat) {
        tilted++;
      }
    }
    expect(flatCount).toBeGreaterThan(0);
    // 口腔内の大半は +Z を向いていない（向いていたら実法線を残す意味が無い）。
    expect(tilted).toBeGreaterThan(0);
  });

  it('法線もスキニングで回る（首を回すと陰影が動く）', () => {
    const { asset, preview } = loadBundle();
    const identity = new Float64Array(asset.vertexIdentityBasis.componentCount);
    const vertices = verticesOf(asset, identity);
    const normals = new Float32Array(vertices.length);
    recalculateNormals(
      vertices,
      asset.mesh.triangles,
      asset.mesh.uvSplitSource,
      planNormals(asset.mesh.triangles, allReal(asset.mesh.vertexCount), unsplitVertexCount(asset.mesh)),
      normals,
    );
    const rest = jointRestPositions(preview, identity);
    const out = new Float32Array(vertices.length);
    skinNormals(
      preview,
      normals,
      jointSkinMatrices(
        preview,
        rest,
        jointLocalRotations(preview, {
          headYawDegrees: YAW_LIMIT_DEGREES,
          headPitchDegrees: 0,
          gazeYawDegrees: 0,
          gazePitchDegrees: 0,
        }),
      ),
      out,
    );
    let moved = 0;
    let maximumLengthError = 0;
    for (let vertex = 0; vertex < preview.vertexCount; vertex++) {
      const length = Math.hypot(out[vertex * 3], out[vertex * 3 + 1], out[vertex * 3 + 2]);
      maximumLengthError = Math.max(maximumLengthError, Math.abs(length - 1));
      if (Math.abs(out[vertex * 3] - normals[vertex * 3]) > 1e-3) moved++;
    }
    expect(moved).toBeGreaterThan(preview.vertexCount / 2);
    // 重みで混ぜた回転は直交にならないので、掛けた後に正規化し直している。
    expect(maximumLengthError).toBeLessThan(1e-5);
  });
});

describe('法線の計算を口腔内へ絞る', () => {
  it('絞っても結果が変わらない（+Z へ落とす頂点は上書きされるので計算しない）', () => {
    const { asset, preview } = loadBundle();
    const identity = new Float64Array(asset.vertexIdentityBasis.componentCount);
    const vertices = verticesOf(asset, identity);
    const welded = unsplitVertexCount(asset.mesh);
    const keepReal = keepRealNormalMask(
      asset.mesh.vertexCount,
      asset.mesh.triangles,
      classifyTriangles(preview, asset.mesh.triangles, asset.mesh.triangleCount),
    );

    const full = new Float32Array(vertices.length);
    recalculateNormals(
      vertices,
      asset.mesh.triangles,
      asset.mesh.uvSplitSource,
      planNormals(asset.mesh.triangles, allReal(asset.mesh.vertexCount), welded),
      full,
    );
    flattenNormals(full, keepReal);

    const plan = planNormals(asset.mesh.triangles, keepReal, welded);
    const narrow = new Float32Array(vertices.length);
    for (let vertex = 0; vertex < preview.vertexCount; vertex++) narrow[vertex * 3 + 2] = 1;
    recalculateNormals(vertices, asset.mesh.triangles, asset.mesh.uvSplitSource, plan, narrow);
    flattenNormals(narrow, keepReal);

    expect(narrow).toEqual(full);
    // 絞れていなければ速くならない。口腔内とその周りは全体の一部（実測で 4 分の 1 弱）。
    expect(plan.triangles.length).toBeLessThan(asset.mesh.triangleCount / 3);
    expect(plan.triangles.length).toBeGreaterThan(0);
  });
});
