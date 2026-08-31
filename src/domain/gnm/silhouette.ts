// 顔の外にある耳・首を、写真の体肌シルエットへ追加フィットする。
//
// MediaPipe FaceLandmarker の点は顔面に集中し、耳の外周と首を拘束しない。ここでは顔の密対応
// フィットを初期値にし、SelfieMulticlass の `bodySkin` 境界へ GNM の耳・首の外周点を対応付ける。
// 領域ごとに別の変換は持たず、全領域が同じ identity 係数と1つの相似変換を共有するため、顔と首の
// 境界でメッシュが裂けない。

import { ScalarField, resampleFieldToImage } from '../field';
import {
  HeadFit,
  LandmarkModel,
  Similarity2d,
  evaluateModel,
  regularizationScheduleFor,
  selectModelPoints,
  solveSimilarity2d,
  solveSymmetricPositiveDefinite,
  xyOf,
} from './fit';
import { GnmHeadAsset, gatherBasisAtVertices } from './model';

/** 耳・首の観測として採用する体肌クラス確信度の下限。 */
export const BODY_SKIN_THRESHOLD = 0.35;

/** GNM v3 head の平均形状上で、首の正面外周を取る高さ範囲（メートル）。 */
export const NECK_Y_RANGE: readonly [number, number] = [0.175, 0.225];

export const EAR_LEVELS = 9;
export const NECK_LEVELS = 7;
export const EAR_WEIGHT = 2.0;
export const NECK_WEIGHT = 3.0;
export const MAX_COEFFICIENT_STEP = 0.35;
export const MIN_REGION_CONSTRAINTS = 4;

/** 耳・首を追加したフィットと、検査画像へ描く最終対応点。 */
export interface RegionSilhouetteFit {
  readonly headFit: HeadFit;
  readonly observedPixels: Float64Array;
  readonly fittedPixels: Float64Array;
  readonly regions: readonly string[];
  readonly earConstraintCount: number;
  readonly neckConstraintCount: number;
}

/** 顔→耳→首→合同の順で、同じidentity係数を体肌輪郭へ寄せる。 */
export function refineEarNeckFit(input: {
  initial: HeadFit;
  photoLandmarks: Float64Array;
  landmarkModel: LandmarkModel;
  asset: GnmHeadAsset;
  faceSkin: ScalarField;
  bodySkin: ScalarField;
  imageSize: readonly [number, number];
  identityClip?: number | null;
}): RegionSilhouetteFit {
  const [width, height] = input.imageSize;
  const identityClip = input.identityClip ?? null;
  const bodyValues = resampleFieldToImage(input.bodySkin, width, height);
  const faceValues = resampleFieldToImage(input.faceSkin, width, height);

  const regionBoundaries = new Map<string, Float64Array>();
  for (const [region, values] of [
    [
      'ear',
      (() => {
        const combined = new Float32Array(bodyValues.length);
        for (let pixel = 0; pixel < combined.length; pixel++) {
          combined[pixel] = Math.min(1, Math.max(0, faceValues[pixel] + bodyValues[pixel]));
        }
        return combined;
      })(),
    ],
    ['neck', bodyValues],
  ] as const) {
    regionBoundaries.set(region, innerBoundaryPoints(values, width, height, BODY_SKIN_THRESHOLD));
  }
  const empty = new Float64Array(0);
  if (
    [...regionBoundaries.values()].every(
      (boundary) => boundary.length / 2 < MIN_REGION_CONSTRAINTS,
    )
  ) {
    return {
      headFit: input.initial,
      observedPixels: empty,
      fittedPixels: empty,
      regions: [],
      earConstraintCount: 0,
      neckConstraintCount: 0,
    };
  }

  const mesh = input.asset.mesh;
  const skin = new Uint8Array(mesh.vertexCount);
  for (let vertex = 0; vertex < mesh.vertexCount; vertex++) {
    skin[vertex] = mesh.componentId[vertex] === 0 ? 1 : 0;
  }
  const earAnchors = regionSideAnchorIndices(
    mesh.templateVertexPositions,
    mesh.earRegion,
    mesh.vertexCount,
    EAR_LEVELS,
  );
  const neckAnchors = sideAnchorIndices(
    mesh.templateVertexPositions,
    skin,
    mesh.vertexCount,
    NECK_Y_RANGE,
    NECK_LEVELS,
  );

  const photoPoints = selectModelPoints(input.photoLandmarks, input.landmarkModel);
  let centerTotal = 0;
  let lowX = Infinity;
  let highX = -Infinity;
  for (let point = 0; point < photoPoints.length / 2; point++) {
    const x = photoPoints[point * 2];
    centerTotal += x;
    lowX = Math.min(lowX, x);
    highX = Math.max(highX, x);
  }
  const faceCenterX = centerTotal / (photoPoints.length / 2);
  const faceWidth = highX - lowX;

  let identity = Float64Array.from(input.initial.identity);
  let similarity = input.initial.similarity;
  const residuals = [...input.initial.residualRmsPixels];

  let finalObserved = empty;
  let finalFitted = empty;
  let finalRegions: string[] = [];
  let earCount = 0;
  let neckCount = 0;

  const phases: readonly (readonly string[])[] = [['ear'], ['neck'], ['ear', 'neck']];
  for (const phaseRegions of phases) {
    const vertices = verticesFromIdentity(input.asset, identity);
    const selectedIndices: number[] = [];
    const selectedTargets: number[] = [];
    const selectedWeights: number[] = [];
    const selectedRegions: string[] = [];
    for (const region of phaseRegions) {
      const anchors = region === 'ear' ? earAnchors : neckAnchors;
      const projected = new Float64Array(anchors.length * 2);
      anchors.forEach((vertex, slot) => {
        const [x, y] = similarity.applyPoint(vertices[vertex * 3], vertices[vertex * 3 + 1]);
        projected[slot * 2] = x;
        projected[slot * 2 + 1] = y;
      });
      const boundary = regionBoundaries.get(region) as Float64Array;
      if (boundary.length / 2 < MIN_REGION_CONSTRAINTS) continue;
      const { rows, targets } = nearestBoundaryTargets(
        projected,
        boundary,
        faceCenterX,
        faceWidth,
        region,
      );
      if (rows.length < MIN_REGION_CONSTRAINTS) continue;
      for (let slot = 0; slot < rows.length; slot++) {
        selectedIndices.push(anchors[rows[slot]]);
        selectedTargets.push(targets[slot * 2], targets[slot * 2 + 1]);
        selectedWeights.push(region === 'ear' ? EAR_WEIGHT : NECK_WEIGHT);
        selectedRegions.push(region);
      }
    }
    if (selectedIndices.length === 0) continue;

    const anchorIndices = Int32Array.from(selectedIndices);
    const anchorTargets = Float64Array.from(selectedTargets);
    const anchorWeights = Float64Array.from(selectedWeights);
    identity = solveIdentity(
      input.landmarkModel,
      photoPoints,
      similarity,
      input.asset,
      anchorIndices,
      anchorTargets,
      anchorWeights,
      identity,
    );
    if (identityClip !== null) {
      for (let component = 0; component < identity.length; component++) {
        identity[component] = Math.min(identityClip, Math.max(-identityClip, identity[component]));
      }
    }
    similarity = solveSimilarity2d(xyOf(evaluateModel(input.landmarkModel, identity)), photoPoints);
    const fittedFace = similarity.apply(xyOf(evaluateModel(input.landmarkModel, identity)));
    let squared = 0;
    for (let point = 0; point < photoPoints.length / 2; point++) {
      const dx = photoPoints[point * 2] - fittedFace[point * 2];
      const dy = photoPoints[point * 2 + 1] - fittedFace[point * 2 + 1];
      squared += dx * dx + dy * dy;
    }
    residuals.push(Math.sqrt(squared / (photoPoints.length / 2)));

    const fittedVertices = verticesFromIdentity(input.asset, identity);
    const fitted = new Float64Array(anchorIndices.length * 2);
    anchorIndices.forEach((vertex, slot) => {
      const [x, y] = similarity.applyPoint(
        fittedVertices[vertex * 3],
        fittedVertices[vertex * 3 + 1],
      );
      fitted[slot * 2] = x;
      fitted[slot * 2 + 1] = y;
    });
    finalObserved = anchorTargets;
    finalFitted = fitted;
    finalRegions = selectedRegions;
    earCount = selectedRegions.filter((region) => region === 'ear').length;
    neckCount = selectedRegions.filter((region) => region === 'neck').length;
  }

  return {
    headFit: { identity, similarity, residualRmsPixels: residuals },
    observedPixels: finalObserved,
    fittedPixels: finalFitted,
    regions: finalRegions,
    earConstraintCount: earCount,
    neckConstraintCount: neckCount,
  };
}

/** identity 係数から split 空間の頂点を作る（`verticesOf` と同じ式）。 */
function verticesFromIdentity(asset: GnmHeadAsset, identity: Float64Array): Float64Array {
  const basis = asset.vertexIdentityBasis;
  const out = Float64Array.from(asset.mesh.templateVertexPositions);
  for (let component = 0; component < basis.componentCount; component++) {
    const coefficient = identity[component];
    if (coefficient === 0) continue;
    const factor = (coefficient * basis.scales[component]) / 32767;
    const base = component * basis.vertexCount * 3;
    for (let index = 0; index < out.length; index++) {
      out[index] += basis.quantized[base + index] * factor;
    }
  }
  return out;
}

/** 二値領域の内側1画素の輪郭。画像端も輪郭として残す。 */
function innerBoundaryPoints(
  values: Float32Array,
  width: number,
  height: number,
  threshold: number,
): Float64Array {
  const mask = new Uint8Array(width * height);
  for (let pixel = 0; pixel < mask.length; pixel++) mask[pixel] = values[pixel] >= threshold ? 1 : 0;
  const points: number[] = [];
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      if (mask[row * width + column] === 0) continue;
      const left = column > 0 ? mask[row * width + column - 1] : 0;
      const right = column < width - 1 ? mask[row * width + column + 1] : 0;
      const up = row > 0 ? mask[(row - 1) * width + column] : 0;
      const down = row < height - 1 ? mask[(row + 1) * width + column] : 0;
      const interior = left !== 0 && right !== 0 && up !== 0 && down !== 0;
      if (!interior) points.push(column, row);
    }
  }
  return Float64Array.from(points);
}

/** 各高さで左右端の頂点を1つずつ選ぶ。 */
function sideAnchorIndices(
  positions: Float32Array,
  region: Uint8Array,
  vertexCount: number,
  yRange: readonly [number, number],
  levels: number,
): Int32Array {
  const result = new Set<number>();
  const halfBand = (yRange[1] - yRange[0]) / Math.max(2 * (levels - 1), 1);
  for (let level = 0; level < levels; level++) {
    const y = yRange[0] + ((yRange[1] - yRange[0]) * level) / Math.max(levels - 1, 1);
    let lowest = -1;
    let highest = -1;
    let lowestX = Infinity;
    let highestX = -Infinity;
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      if (region[vertex] === 0) continue;
      if (Math.abs(positions[vertex * 3 + 1] - y) > halfBand) continue;
      const x = positions[vertex * 3];
      if (x < lowestX) {
        lowestX = x;
        lowest = vertex;
      }
      if (x > highestX) {
        highestX = x;
        highest = vertex;
      }
    }
    if (lowest >= 0) result.add(lowest);
    if (highest >= 0) result.add(highest);
  }
  return Int32Array.from([...result].sort((first, second) => first - second));
}

/** 解剖領域の全高から、各高さの左右外端を選ぶ。 */
function regionSideAnchorIndices(
  positions: Float32Array,
  region: Uint8Array,
  vertexCount: number,
  levels: number,
): Int32Array {
  let low = Infinity;
  let high = -Infinity;
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    if (region[vertex] === 0) continue;
    low = Math.min(low, positions[vertex * 3 + 1]);
    high = Math.max(high, positions[vertex * 3 + 1]);
  }
  if (!Number.isFinite(low)) return new Int32Array(0);
  return sideAnchorIndices(positions, region, vertexCount, [low, high], levels);
}

/** 初期投影の近傍にある同じ側の体肌輪郭を対応先にする。 */
function nearestBoundaryTargets(
  projected: Float64Array,
  boundaryXy: Float64Array,
  faceCenterX: number,
  faceWidth: number,
  region: string,
): { rows: number[]; targets: Float64Array } {
  const radiusX = faceWidth * (region === 'ear' ? 0.22 : 0.18);
  const radiusY = faceWidth * (region === 'ear' ? 0.14 : 0.1);
  const rows: number[] = [];
  const targets: number[] = [];
  const boundaryCount = boundaryXy.length / 2;
  for (let row = 0; row < projected.length / 2; row++) {
    const pointX = projected[row * 2];
    const pointY = projected[row * 2 + 1];
    const wantLeft = pointX <= faceCenterX;
    let bestCost = Infinity;
    let bestIndex = -1;
    for (let index = 0; index < boundaryCount; index++) {
      const boundaryX = boundaryXy[index * 2];
      const sameSide = wantLeft ? boundaryX <= faceCenterX : boundaryX >= faceCenterX;
      if (!sameSide) continue;
      const deltaX = boundaryX - pointX;
      if (Math.abs(deltaX) > radiusX) continue;
      const deltaY = boundaryXy[index * 2 + 1] - pointY;
      if (Math.abs(deltaY) > radiusY) continue;
      // 外周フィットでは高さを保ち、主に横方向へ輪郭を寄せる。
      const cost = (deltaX / radiusX) ** 2 + ((2 * deltaY) / radiusY) ** 2;
      if (cost < bestCost) {
        bestCost = cost;
        bestIndex = index;
      }
    }
    if (bestIndex < 0) continue;
    rows.push(row);
    targets.push(boundaryXy[bestIndex * 2], boundaryXy[bestIndex * 2 + 1]);
  }
  return { rows, targets: Float64Array.from(targets) };
}

function solveIdentity(
  model: LandmarkModel,
  photoPoints: Float64Array,
  similarity: Similarity2d,
  asset: GnmHeadAsset,
  anchorIndices: Int32Array,
  anchorTargets: Float64Array,
  anchorWeights: Float64Array,
  currentIdentity: Float64Array,
): Float64Array {
  const faceTarget = similarity.inverseApply(photoPoints);
  const anchorTarget = similarity.inverseApply(anchorTargets);
  const facePoints = model.pointCount;
  const anchorPoints = anchorIndices.length;
  const totalPoints = facePoints + anchorPoints;
  const componentCount = model.identityComponentCount;

  const anchorBasis = gatherBasisAtVertices(asset.vertexIdentityBasis, anchorIndices);
  const faceWidth = model.faceWidth;
  const observationCount = totalPoints * 2;
  const design = new Float64Array(observationCount * componentCount);
  const residual = new Float64Array(observationCount);

  const rootWeight = new Float64Array(totalPoints);
  for (let point = 0; point < facePoints; point++) rootWeight[point] = Math.sqrt(model.weights[point]);
  for (let slot = 0; slot < anchorPoints; slot++) {
    rootWeight[facePoints + slot] = Math.sqrt(anchorWeights[slot]);
  }

  for (let component = 0; component < componentCount; component++) {
    const faceBase = component * facePoints * 3;
    const anchorBase = component * anchorPoints * 3;
    for (let point = 0; point < facePoints; point++) {
      const weight = rootWeight[point] / faceWidth;
      design[point * 2 * componentCount + component] =
        model.identityBasis[faceBase + point * 3] * weight;
      design[(point * 2 + 1) * componentCount + component] =
        model.identityBasis[faceBase + point * 3 + 1] * weight;
    }
    for (let slot = 0; slot < anchorPoints; slot++) {
      const point = facePoints + slot;
      const weight = rootWeight[point] / faceWidth;
      design[point * 2 * componentCount + component] =
        anchorBasis[anchorBase + slot * 3] * weight;
      design[(point * 2 + 1) * componentCount + component] =
        anchorBasis[anchorBase + slot * 3 + 1] * weight;
    }
  }

  for (let point = 0; point < facePoints; point++) {
    const weight = rootWeight[point] / faceWidth;
    residual[point * 2] = (faceTarget[point * 2] - model.meanPositions[point * 3]) * weight;
    residual[point * 2 + 1] =
      (faceTarget[point * 2 + 1] - model.meanPositions[point * 3 + 1]) * weight;
  }
  for (let slot = 0; slot < anchorPoints; slot++) {
    const point = facePoints + slot;
    const vertex = anchorIndices[slot];
    const weight = rootWeight[point] / faceWidth;
    residual[point * 2] =
      (anchorTarget[slot * 2] - asset.mesh.templateVertexPositions[vertex * 3]) * weight;
    residual[point * 2 + 1] =
      (anchorTarget[slot * 2 + 1] - asset.mesh.templateVertexPositions[vertex * 3 + 1]) * weight;
  }

  const gram = new Float64Array(componentCount * componentCount);
  for (let row = 0; row < componentCount; row++) {
    for (let column = row; column < componentCount; column++) {
      let total = 0;
      for (let observation = 0; observation < observationCount; observation++) {
        total +=
          design[observation * componentCount + row] * design[observation * componentCount + column];
      }
      const value = total / observationCount;
      gram[row * componentCount + column] = value;
      gram[column * componentCount + row] = value;
    }
  }
  const rightHand = new Float64Array(componentCount);
  for (let component = 0; component < componentCount; component++) {
    let total = 0;
    for (let observation = 0; observation < observationCount; observation++) {
      total += design[observation * componentCount + component] * residual[observation];
    }
    rightHand[component] = total / observationCount;
  }
  const schedule = regularizationScheduleFor(model, 1);
  const lambda = schedule[schedule.length - 1];
  for (let component = 0; component < componentCount; component++) {
    gram[component * componentCount + component] += lambda;
  }
  const solved = solveSymmetricPositiveDefinite(gram, rightHand, componentCount);
  const out = new Float64Array(componentCount);
  for (let component = 0; component < componentCount; component++) {
    const step = Math.min(
      MAX_COEFFICIENT_STEP,
      Math.max(-MAX_COEFFICIENT_STEP, solved[component] - currentIdentity[component]),
    );
    out[component] = currentIdentity[component] + step;
  }
  return out;
}
