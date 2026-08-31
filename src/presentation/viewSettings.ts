// 3D ビューの調整値。**書き出しには一切影響しない。**
//
// `application/settings` の `ExportSettings` と分けてあるのは、書き出しの再現性に関わる値と「見る
// ときの都合」を混ぜないため。**どちらも保存しない**（毎回この既定から始める）。
//
// 既定値の正本は Unity 側（1-10/2607_Obayashi_Avatar_Mockup_3DGS の `Assets/Sandbox/Ooba/GNM`）で、
// カメラは `Scenes/Viewer.unity`、首と視線は `Viewer/GnmHeadPoseController`、表情の自動再生は
// `Viewer/GnmExpressionPlayer`。まばたきと背景色と FOV / 距離の調整は旧 web 版から残したもの。

import {
  FADE_SECONDS,
  ExpressionPlayMode,
  HOLD_SECONDS,
} from '../domain/preview/expression';
import { NECK_SHARE } from '../domain/preview/pose';
import {
  DEFAULT_BACKGROUND,
  DEFAULT_DISTANCE_METERS,
  DEFAULT_FOV_DEGREES,
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
