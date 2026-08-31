// 3D ビューの調整値。**書き出しには一切影響しない。**
//
// `application/settings` の `ExportSettings` と分けてあるのは、書き出しの再現性に関わる値と「見る
// ときの都合」を混ぜないため。保存も別のキーにしてある（`presentation/parameterStore`）。
//
// 既定値の正本は Unity 側（1-10/2607_Obayashi_Avatar_Mockup_3DGS の `Assets/Sandbox/Ooba/GNM`）で、
// カメラは `Scenes/Viewer.unity`、首と視線は `Viewer/GnmHeadPoseController`、表情の自動再生は
// `Viewer/GnmExpressionPlayer`。まばたきと背景色と FOV / 距離の調整は旧 web 版から残したもの。

import {
  FADE_SECONDS,
  ExpressionPlayMode,
  HOLD_SECONDS,
} from '../domain/preview/expression';
import { GAZE_LIMIT_DEGREES, NECK_SHARE, PITCH_LIMIT_DEGREES, YAW_LIMIT_DEGREES } from '../domain/preview/pose';
import {
  DEFAULT_BACKGROUND,
  DEFAULT_DISTANCE_METERS,
  DEFAULT_FOV_DEGREES,
  MAXIMUM_DISTANCE_METERS,
  MAXIMUM_FOV_DEGREES,
  MINIMUM_DISTANCE_METERS,
  MINIMUM_FOV_DEGREES,
} from './viewer';

export const MINIMUM_FADE_SECONDS = 0;
export const MAXIMUM_FADE_SECONDS = 2;
export const MINIMUM_HOLD_SECONDS = 0;
export const MAXIMUM_HOLD_SECONDS = 5;
export const MINIMUM_EXPRESSION_INTENSITY = 0;
export const MAXIMUM_EXPRESSION_INTENSITY = 2;

/** 自動再生の選べる値（GUI のドロップダウンの並び）。 */
export const PLAY_MODES: readonly ExpressionPlayMode[] = ['off', 'sequence', 'random'];

/** 自動再生の日本語ラベル。 */
export const PLAY_MODE_LABELS: Readonly<Record<ExpressionPlayMode, string>> = {
  off: '手動',
  sequence: '順番に',
  random: 'ランダム',
};

/** 3D ビューの調整値。 */
export interface ViewSettings {
  readonly fovDegrees: number;
  readonly distanceMeters: number;
  readonly background: string;
  readonly showWireframe: boolean;
  readonly headYawDegrees: number;
  readonly headPitchDegrees: number;
  readonly gazeYawDegrees: number;
  readonly gazePitchDegrees: number;
  readonly neckShare: number;
  readonly followPointer: boolean;
  readonly playMode: ExpressionPlayMode;
  readonly fadeSeconds: number;
  readonly holdSeconds: number;
  readonly expressionIntensity: number;
  readonly blinkEnabled: boolean;
}

export const DEFAULT_VIEW_SETTINGS: ViewSettings = {
  fovDegrees: DEFAULT_FOV_DEGREES,
  distanceMeters: DEFAULT_DISTANCE_METERS,
  background: DEFAULT_BACKGROUND,
  showWireframe: false,
  headYawDegrees: 0,
  headPitchDegrees: 0,
  gazeYawDegrees: 0,
  gazePitchDegrees: 0,
  neckShare: NECK_SHARE,
  followPointer: false,
  playMode: 'off',
  fadeSeconds: FADE_SECONDS,
  holdSeconds: HOLD_SECONDS,
  expressionIntensity: 1,
  blinkEnabled: true,
};

function clamped(value: unknown, low: number, high: number, fallback: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return fallback;
  return Math.min(high, Math.max(low, value));
}

function boolean(value: unknown, fallback: boolean): boolean {
  return typeof value === 'boolean' ? value : fallback;
}

/**
 * 読んだ値を使える形に直す。**落とさずに丸める**（`ExportSettings` とは扱いが違う）。
 *
 * 書き出しの値は範囲外なら使わずに既定へ戻す（結果が黙って変わるのを避ける）。ビューの値は結果に
 * 影響しないので、丸めて使う方が「保存したのに戻っている」より親切。
 */
export function normalizeViewSettings(values: unknown): ViewSettings {
  if (typeof values !== 'object' || values === null || Array.isArray(values)) {
    return DEFAULT_VIEW_SETTINGS;
  }
  const raw = values as Record<string, unknown>;
  const mode = PLAY_MODES.includes(raw.playMode as ExpressionPlayMode)
    ? (raw.playMode as ExpressionPlayMode)
    : DEFAULT_VIEW_SETTINGS.playMode;
  return {
    fovDegrees: clamped(
      raw.fovDegrees,
      MINIMUM_FOV_DEGREES,
      MAXIMUM_FOV_DEGREES,
      DEFAULT_VIEW_SETTINGS.fovDegrees,
    ),
    distanceMeters: clamped(
      raw.distanceMeters,
      MINIMUM_DISTANCE_METERS,
      MAXIMUM_DISTANCE_METERS,
      DEFAULT_VIEW_SETTINGS.distanceMeters,
    ),
    background:
      typeof raw.background === 'string' && /^#[0-9a-fA-F]{6}$/.test(raw.background)
        ? raw.background
        : DEFAULT_VIEW_SETTINGS.background,
    showWireframe: boolean(raw.showWireframe, DEFAULT_VIEW_SETTINGS.showWireframe),
    headYawDegrees: clamped(raw.headYawDegrees, -YAW_LIMIT_DEGREES, YAW_LIMIT_DEGREES, 0),
    headPitchDegrees: clamped(raw.headPitchDegrees, -PITCH_LIMIT_DEGREES, PITCH_LIMIT_DEGREES, 0),
    gazeYawDegrees: clamped(raw.gazeYawDegrees, -GAZE_LIMIT_DEGREES, GAZE_LIMIT_DEGREES, 0),
    gazePitchDegrees: clamped(raw.gazePitchDegrees, -GAZE_LIMIT_DEGREES, GAZE_LIMIT_DEGREES, 0),
    neckShare: clamped(raw.neckShare, 0, 1, DEFAULT_VIEW_SETTINGS.neckShare),
    followPointer: boolean(raw.followPointer, DEFAULT_VIEW_SETTINGS.followPointer),
    playMode: mode,
    fadeSeconds: clamped(
      raw.fadeSeconds,
      MINIMUM_FADE_SECONDS,
      MAXIMUM_FADE_SECONDS,
      DEFAULT_VIEW_SETTINGS.fadeSeconds,
    ),
    holdSeconds: clamped(
      raw.holdSeconds,
      MINIMUM_HOLD_SECONDS,
      MAXIMUM_HOLD_SECONDS,
      DEFAULT_VIEW_SETTINGS.holdSeconds,
    ),
    expressionIntensity: clamped(
      raw.expressionIntensity,
      MINIMUM_EXPRESSION_INTENSITY,
      MAXIMUM_EXPRESSION_INTENSITY,
      DEFAULT_VIEW_SETTINGS.expressionIntensity,
    ),
    blinkEnabled: boolean(raw.blinkEnabled, DEFAULT_VIEW_SETTINGS.blinkEnabled),
  };
}
