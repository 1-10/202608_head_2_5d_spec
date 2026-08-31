// 6 段を通して guest zip の成果物ができることを、**推論なしで**確かめる。
//
// 推論は Port 経由なので、偽の Port を渡せばフィットとアトラスと髪シェルをそのまま回せる（これが
// `application/ports` を持つ理由そのもの）。ここが通れば、段の合成・座標系の受け渡し・契約の検査が
// 全部繋がっている。
//
// **写真は合成**（平均顔を相似変換で写した位置に、顔の肌・体の肌・髪の場を置いたもの）。実写での
// 見た目はブラウザで確認する — ここで見るのは「段が繋がっていること」と「契約を満たす値が出ること」。

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { DepthNormalEstimator, FaceLandmarkDetector, PersonSegmenter } from '../src/application/ports';
import { DEFAULT_SETTINGS } from '../src/application/settings';
import { exportGuest } from '../src/application/exportGuest';
import { entryNames } from '../src/domain/contract';
import {
  DepthNormalResult,
  PersonSegmentation,
  fieldOverFullImage,
  makeField,
  rectFromPixels,
} from '../src/domain/field';
import { EYE_SIDES } from '../src/domain/eyes/layout';
import {
  MEDIAPIPE_FACE_MESH_COUNT,
  MEDIAPIPE_LANDMARK_COUNT,
  Similarity2d,
  buildDenseLandmarkModel,
  evaluateModel,
  xyOf,
} from '../src/domain/gnm/fit';
import { GnmHeadAsset } from '../src/domain/gnm/model';
import { meanChinHeight } from '../src/domain/gnm/crop';
import { PhotoRgb } from '../src/domain/photo';
import { CachingAtlasBaker } from '../src/infrastructure/atlasBaker';
import { DomainHairImageProcessor } from '../src/infrastructure/hairImage';
import { loadAsset } from './asset';

const WIDTH = 480;
const HEIGHT = 600;

/**
 * 平均形状が写真に収まる相似変換と、そこから決まる領域の境目。
 *
 * **写真の額縁ではなくメッシュの投影域から決める**（`domain/gnm/crop` が切り出しでそうしているのと
 * 同じ理由 — 数値を決め打ちすると、アセットが変わったときに顔が枠から出る）。
 */
function layout(asset: GnmHeadAsset): {
  similarity: Similarity2d;
  hairBottomRow: number;
  faceBottomRow: number;
} {
  const mesh = asset.mesh;
  let lowX = Infinity;
  let lowY = Infinity;
  let highX = -Infinity;
  let highY = -Infinity;
  for (let vertex = 0; vertex < mesh.vertexCount; vertex++) {
    lowX = Math.min(lowX, mesh.templateVertexPositions[vertex * 3]);
    highX = Math.max(highX, mesh.templateVertexPositions[vertex * 3]);
    lowY = Math.min(lowY, mesh.templateVertexPositions[vertex * 3 + 1]);
    highY = Math.max(highY, mesh.templateVertexPositions[vertex * 3 + 1]);
  }
  const scale = Math.min((WIDTH * 0.9) / (highX - lowX), (HEIGHT * 0.9) / (highY - lowY));
  // GNM の +Y が上、画像の行が下なので鏡映を含む。
  const similarity = new Similarity2d(
    Float64Array.from([scale, 0, 0, -scale]),
    Float64Array.from([
      WIDTH / 2 - (scale * (lowX + highX)) / 2,
      HEIGHT / 2 + (scale * (lowY + highY)) / 2,
    ]),
  );
  const model = buildDenseLandmarkModel(asset, asset.dense);
  const chinRow = similarity.applyPoint(0, meanChinHeight(model))[1];
  const topRow = similarity.applyPoint(0, highY)[1];
  return {
    similarity,
    // 髪は頭頂から顎までの上 3 割。顔はそこから顎まで。顎より下は体の肌。
    hairBottomRow: topRow + (chinRow - topRow) * 0.3,
    faceBottomRow: chinRow,
  };
}

/** 平均顔を写した 478 点。虹彩 10 点は眼球コンポーネントの重心へ置く。 */
function syntheticLandmarks(asset: GnmHeadAsset): Float64Array {
  const { similarity } = layout(asset);
  const model = buildDenseLandmarkModel(asset, asset.dense);
  const projected = similarity.apply(
    xyOf(evaluateModel(model, new Float64Array(model.identityComponentCount))),
  );
  const landmarks = new Float64Array(MEDIAPIPE_LANDMARK_COUNT * 2);
  for (let point = 0; point < model.pointCount; point++) {
    const target = model.photoIndices[point];
    landmarks[target * 2] = projected[point * 2];
    landmarks[target * 2 + 1] = projected[point * 2 + 1];
  }
  // 虹彩の 5 点（中心 + 縁 4 点）。並びは検出器の出力と同じで、先頭 5 点が片目・次の 5 点が
  // もう片方（どちらが解剖学的にどちらかは `assignEyeSides` が決める）。
  const centroids: [number, number][] = [];
  for (const componentName of ['right_eye', 'left_eye']) {
    const componentIndex = asset.mesh.componentNames.indexOf(componentName);
    let count = 0;
    let totalX = 0;
    let totalY = 0;
    for (let vertex = 0; vertex < asset.mesh.vertexCount; vertex++) {
      if (asset.mesh.componentId[vertex] !== componentIndex) continue;
      count++;
      totalX += asset.mesh.templateVertexPositions[vertex * 3];
      totalY += asset.mesh.templateVertexPositions[vertex * 3 + 1];
    }
    centroids.push(similarity.applyPoint(totalX / count, totalY / count));
  }
  // limbus 半径の実測（約 5.9mm）を写真の画素へ。
  const irisRadius = 0.0059 * similarity.scale;
  centroids.forEach(([x, y], group) => {
    const base = MEDIAPIPE_FACE_MESH_COUNT + group * 5;
    const offsets: [number, number][] = [
      [0, 0],
      [irisRadius, 0],
      [-irisRadius, 0],
      [0, irisRadius],
      [0, -irisRadius],
    ];
    offsets.forEach(([dx, dy], slot) => {
      landmarks[(base + slot) * 2] = x + dx;
      landmarks[(base + slot) * 2 + 1] = y + dy;
    });
  });
  return landmarks;
}

/** 顔・体・髪を写した合成写真（色は領域ごとに変えて、焼けた場所が読めるようにする）。 */
function syntheticPhoto(asset: GnmHeadAsset): PhotoRgb {
  const { hairBottomRow } = layout(asset);
  const data = new Uint8Array(WIDTH * HEIGHT * 3);
  for (let row = 0; row < HEIGHT; row++) {
    for (let column = 0; column < WIDTH; column++) {
      const pixel = (row * WIDTH + column) * 3;
      const hair = row < hairBottomRow;
      data[pixel] = hair ? 60 : 205;
      data[pixel + 1] = hair ? 45 : 155;
      data[pixel + 2] = hair ? 40 : 135;
    }
  }
  return { data, width: WIDTH, height: HEIGHT };
}

class FakeLandmarkDetector implements FaceLandmarkDetector {
  constructor(private readonly asset: GnmHeadAsset) {}
  async detect(): Promise<Float64Array> {
    return syntheticLandmarks(this.asset);
  }
}

class FakeSegmenter implements PersonSegmenter {
  constructor(private readonly asset: GnmHeadAsset) {}
  async segment(photo: PhotoRgb): Promise<PersonSegmentation> {
    const { hairBottomRow, faceBottomRow } = layout(this.asset);
    const area = photo.width * photo.height;
    const hair = new Float32Array(area);
    const faceSkin = new Float32Array(area);
    const bodySkin = new Float32Array(area);
    for (let row = 0; row < photo.height; row++) {
      for (let column = 0; column < photo.width; column++) {
        const pixel = row * photo.width + column;
        if (row < hairBottomRow) hair[pixel] = 1;
        else if (row < faceBottomRow) faceSkin[pixel] = 1;
        else bodySkin[pixel] = 1;
      }
    }
    return {
      hair: fieldOverFullImage(hair, photo.width, photo.height),
      accessory: fieldOverFullImage(new Float32Array(area), photo.width, photo.height),
      faceSkin: fieldOverFullImage(faceSkin, photo.width, photo.height),
      bodySkin: fieldOverFullImage(bodySkin, photo.width, photo.height),
    };
  }
}

/**
 * 深度は画像の行に沿う傾斜、法線は +Z、前景は全面 1。
 *
 * 深度が行の傾斜なのは、`fitDepthToLandmarkZ` が「Depth と GNM の z の 1 次式」を解けるだけの分散を
 * 要求するため（値そのものの正しさはここでは見ない）。
 */
class FakeDepthNormal implements DepthNormalEstimator {
  async estimateSquare(
    photo: PhotoRgb,
    square: { x: number; y: number; size: number },
  ): Promise<DepthNormalResult> {
    const resolution = 128;
    const rect = rectFromPixels(
      square.x,
      square.y,
      square.size,
      square.size,
      photo.width,
      photo.height,
    );
    const depth = new Float32Array(resolution * resolution);
    const normal = new Float32Array(3 * resolution * resolution);
    const foreground = new Float32Array(resolution * resolution).fill(1);
    for (let row = 0; row < resolution; row++) {
      for (let column = 0; column < resolution; column++) {
        const index = row * resolution + column;
        depth[index] = 1 - row / (resolution - 1);
        normal[2 * resolution * resolution + index] = 1;
      }
    }
    return {
      depth: makeField(depth, resolution, resolution, rect),
      normal,
      foreground: makeField(foreground, resolution, resolution, rect),
    };
  }
}

/** 合成入力で 6 段を通す（`tools/golden_export_guest.py` と同じ設定）。 */
async function runPipeline(
  asset: GnmHeadAsset,
  stages: string[] = [],
): Promise<Awaited<ReturnType<typeof exportGuest>>> {
  const settings = {
    ...DEFAULT_SETTINGS,
    skinAtlasSize: 512,
    eyeTextureSize: 128,
    hairTextureSize: 512,
  };
  return exportGuest({
    photo: syntheticPhoto(asset),
    asset,
    landmarkDetector: new FakeLandmarkDetector(asset),
    segmenter: new FakeSegmenter(asset),
    depthNormal: new FakeDepthNormal(),
    atlasBaker: new CachingAtlasBaker(),
    hairImageProcessor: new DomainHairImageProcessor(),
    settings,
    exporterVersion: '0.0.0-test',
    onStage: (stage) => stages.push(stage),
  });
}

describe('exportGuest', () => {
  it('6 段を通して契約を満たす成果物が出る', async () => {
    const asset = loadAsset();
    const stages: string[] = [];
    const outcome = await runPipeline(asset, stages);

    // 段は宣言した順に走る。
    expect(stages).toEqual(['推論', 'フィット', '眼球', 'アトラス', '髪シェル', '組み立て']);

    const manifest = outcome.artifacts.manifest;
    expect(manifest.format_version).toBe(2);
    expect(manifest.atlas_size).toBe(512);
    expect(manifest.eye_texture_size).toBe(128);
    expect(manifest.identity_count).toBe(asset.vertexIdentityBasis.componentCount);
    expect(manifest.gnm_version).toBe('3.0');
    expect(manifest.exporter_version).toBe('0.0.0-test');

    // 平均顔をそのまま写した写真なので identity はほぼ 0（シルエットフィットの分だけ動く）。
    for (const value of manifest.identity) expect(Math.abs(value)).toBeLessThan(1);

    expect(outcome.artifacts.skinAlbedo.length).toBe(512 * 512 * 3);
    for (const side of EYE_SIDES) {
      expect(outcome.artifacts.eyeAlbedos[side].length).toBe(128 * 128 * 3);
    }

    // 髪の場を置いたので髪シェルが出て、髪系 3 つが zip に入る。
    expect(outcome.artifacts.hair).not.toBeNull();
    expect(entryNames(outcome.artifacts)).toEqual([
      'guest.json',
      'skin_albedo.jpg',
      'left_eye_albedo.png',
      'right_eye_albedo.png',
      'hair_shell.bin',
      'hair_albedo.jpg',
      'hair_alpha.png',
    ]);
    // 髪テクスチャと alpha は同じ形（契約が要求する）。
    expect(outcome.artifacts.hairAlbedo?.width).toBe(outcome.artifacts.hairAlpha?.width);
    expect(outcome.artifacts.hairAlbedo?.height).toBe(outcome.artifacts.hairAlpha?.height);

    // 検査画像は段ごとに出る。
    expect(outcome.inspection.photoLandmarks).toBeDefined();
    expect(outcome.inspection.landmarkFit).toBeDefined();
    expect(outcome.inspection.leftEyeAlbedo).toBeDefined();
    expect(outcome.inspection.atlasAlbedo).toBeDefined();
    expect(outcome.inspection.hairShellWire).toBeDefined();

    // 眼球は左右とも「その側の写真」から焼かれている（片方をもう片方で代用しない）。
    expect(outcome.eyeAlbedos.left.side).toBe('left');
    expect(outcome.eyeAlbedos.right.side).toBe('right');
    // 虹彩の大きさは公式へ揃えない（比が 1 に潰れていない、あるいは偶然 1 でも潰した結果ではない）。
    expect(outcome.eyeAlbedos.left.limbusRadiusPx).toBeGreaterThan(0);
  }, 300_000);
});

/** 配列の要約（`tools/golden_export_guest.py` の `summary` と同じ形）。 */
function summary(values: ArrayLike<number>): {
  count: number;
  sum: number;
  min: number;
  max: number;
  mean: number;
} {
  let sum = 0;
  let minimum = Infinity;
  let maximum = -Infinity;
  for (let index = 0; index < values.length; index++) {
    const value = values[index];
    sum += value;
    if (value < minimum) minimum = value;
    if (value > maximum) maximum = value;
  }
  return {
    count: values.length,
    sum,
    min: values.length === 0 ? 0 : minimum,
    max: values.length === 0 ? 0 : maximum,
    mean: values.length === 0 ? 0 : sum / values.length,
  };
}

describe('正本との突き合わせ', () => {
  /**
   * **正本の domain をそのまま動かして作った基準値**と、同じ合成入力での移植の結果を比べる。
   *
   * 基準値は `tools/golden_export_guest.py` が作る（あちらの `domain` は numpy 以外を import しない
   * ので、そのまま呼べる）。**定数の一致はテキストで確かめられるが、アルゴリズムの一致は数値でしか
   * 確かめられない。**
   *
   * 許容差は「同じ計算を別の順序でやったときに出る差」の桁に置く。**桁で外れたら移植が違う。**
   */
  it('identity・アトラス・眼球・髪シェルが基準値と一致する', async () => {
    const golden = JSON.parse(
      readFileSync(resolve(__dirname, 'golden', 'exportGuest.json'), 'utf-8'),
    ) as Record<string, never>;
    const asset = loadAsset();
    const outcome = await runPipeline(asset);
    const deltas: string[] = [];
    const measured = new Map<string, number>();
    const close = (name: string, actual: number, expected: number, tolerance: number): void => {
      const difference = Math.abs(actual - expected);
      const scale = Math.max(1, Math.abs(expected));
      measured.set(name, difference / scale);
      if (difference / scale > tolerance) {
        deltas.push(`${name}: 移植 ${actual} / 正本 ${expected}（相対差 ${difference / scale}）`);
      }
    };
    const closeSummary = (
      name: string,
      actual: ReturnType<typeof summary>,
      expected: Record<string, number>,
      tolerance: number,
    ): void => {
      expect(actual.count, `${name}.count`).toBe(expected['count']);
      for (const key of ['sum', 'min', 'max', 'mean'] as const) {
        close(`${name}.${key}`, actual[key], expected[key], tolerance);
      }
    };

    // 密対応（`tools/export_gnm_assets.py` が作ったものと、あちらが npz から作るものの一致）。
    const dense = golden['dense'] as unknown as Record<string, never>;
    expect(asset.dense.pointCount).toBe(dense['point_count'] as unknown as number);
    close('dense.edge_meters', asset.dense.edgeMeters, dense['edge_meters'] as unknown as number, 1e-9);
    closeSummary(
      'dense.residual_meters',
      summary(asset.dense.residualMeters),
      dense['residual_meters'] as unknown as Record<string, number>,
      1e-6,
    );

    // identity 係数（フィットの唯一の成果物）。
    //
    // **ここだけ許容差が緩いのは int16 量子化のぶん**（`tools/export_gnm_assets.py`。web だから
    // 増えた差分）。あちらは identity 基底を float32 のまま持つので、フィットの設計行列がわずかに
    // 違う。あちらが量子化をやめたときに測った係数の差が 7e-4 で、実測はその内側に収まる。
    // **桁で外れたら量子化ではなくアルゴリズムが違う。**
    const IDENTITY_TOLERANCE = 1e-3;
    closeSummary(
      'identity',
      summary(outcome.artifacts.manifest.identity),
      golden['identity'] as unknown as Record<string, number>,
      IDENTITY_TOLERANCE,
    );
    const head = golden['identity_head'] as unknown as number[];
    head.forEach((expected, index) => {
      close(
        `identity[${index}]`,
        outcome.artifacts.manifest.identity[index],
        expected,
        IDENTITY_TOLERANCE,
      );
    });

    // 肌アトラスと眼球テクスチャ（sRGB uint8 なので、丸めで ±1 階調は動く）。
    closeSummary(
      'skin_albedo',
      summary(outcome.artifacts.skinAlbedo),
      golden['skin_albedo'] as unknown as Record<string, number>,
      2e-3,
    );
    closeSummary(
      'eye_albedo_left',
      summary(outcome.artifacts.eyeAlbedos.left),
      golden['eye_albedo_left'] as unknown as Record<string, number>,
      2e-3,
    );
    closeSummary(
      'eye_albedo_right',
      summary(outcome.artifacts.eyeAlbedos.right),
      golden['eye_albedo_right'] as unknown as Record<string, number>,
      2e-3,
    );
    close(
      'eye_left_limbus_px',
      outcome.eyeAlbedos.left.limbusRadiusPx,
      golden['eye_left_limbus_px'] as unknown as number,
      1e-5,
    );
    close(
      'eye_left_iris_px',
      outcome.eyeAlbedos.left.irisRadiusPx,
      golden['eye_left_iris_px'] as unknown as number,
      1e-9,
    );

    // 髪シェル（格子・厚み・法線融合・三角形の採用がすべて効く）。
    const hair = golden['hair'] as unknown as Record<string, never>;
    expect(outcome.artifacts.hair).not.toBeNull();
    const shell = outcome.artifacts.hair as NonNullable<typeof outcome.artifacts.hair>;
    expect(shell.vertexCount, 'hair.vertex_count').toBe(hair['vertex_count'] as unknown as number);
    expect(shell.triangleCount, 'hair.triangle_count').toBe(
      hair['triangle_count'] as unknown as number,
    );
    closeSummary(
      'hair.positions',
      summary(shell.positions),
      hair['positions'] as unknown as Record<string, number>,
      1e-4,
    );
    closeSummary(
      'hair.uvs',
      summary(shell.uvs),
      hair['uvs'] as unknown as Record<string, number>,
      1e-5,
    );
    closeSummary(
      'hair_alpha',
      summary((outcome.artifacts.hairAlpha as NonNullable<typeof outcome.artifacts.hairAlpha>).data),
      golden['hair_alpha'] as unknown as Record<string, number>,
      5e-3,
    );
    closeSummary(
      'hair_albedo',
      summary(
        (outcome.artifacts.hairAlbedo as NonNullable<typeof outcome.artifacts.hairAlbedo>).data,
      ),
      golden['hair_albedo'] as unknown as Record<string, number>,
      5e-3,
    );

    if (deltas.length > 0) throw new Error('正本と差がある:\n' + deltas.join('\n'));
  }, 300_000);
});
