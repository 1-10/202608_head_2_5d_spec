// 3D ビューの調整値。**書き出しには一切影響しない。**
//
// `application/settings` の `ExportSettings` と分けてあるのは、書き出しの再現性に関わる値と「見る
// ときの都合」を混ぜないため。**どちらも保存しない**（毎回この既定から始める）。
//
// 既定値の正本は Unity 側（1-10/2607_Obayashi_Avatar_Mockup_3DGS の `Assets/Sandbox/Ooba/GNM`）で、
// カメラは `Scenes/Viewer.unity`、首と視線は `Viewer/GnmHeadPoseController`、表情の自動再生は
// `Viewer/GnmExpressionPlayer`。まばたきと背景色と FOV の調整は旧 web 版から残したもの。

import {
  DEFAULT_FOV_DEGREES,
  TARGET_HEIGHT_METERS,
  cameraPoseAt,
} from '../domain/preview/camera';
import {
  FADE_SECONDS,
  ExpressionPlayMode,
  HOLD_SECONDS,
} from '../domain/preview/expression';
import { NECK_SHARE } from '../domain/preview/pose';
import { VISEME_FADE_SECONDS, VISEME_HOLD_SECONDS } from '../domain/preview/viseme';
import {
  AMBIENT_LIGHT,
  DEFAULT_AMBIENT_COLOR,
  DEFAULT_BACKGROUND,
  DEFAULT_LIGHT_COLOR,
  DEFAULT_LIGHT_INTENSITY,
} from './viewer';

export const MINIMUM_FADE_SECONDS = 0;
export const MAXIMUM_FADE_SECONDS = 2;
export const MINIMUM_HOLD_SECONDS = 0;
export const MAXIMUM_HOLD_SECONDS = 5;
export const MINIMUM_EXPRESSION_INTENSITY = 0;
export const MAXIMUM_EXPRESSION_INTENSITY = 2;

// 口形の連続再生は 1 音ぶんが短いので、表情の 0〜2 秒 / 0〜5 秒とは範囲を分ける（同じ範囲だと
// スライダーの端 1/10 でしか触れない）。
export const MINIMUM_VISEME_FADE_SECONDS = 0;
export const MAXIMUM_VISEME_FADE_SECONDS = 0.6;
export const MINIMUM_VISEME_HOLD_SECONDS = 0;
export const MAXIMUM_VISEME_HOLD_SECONDS = 0.6;

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
  // カメラの位置と回転（ワールド）。**周回半径はここに持たない** — パネルに出ない値で、正本は
  // `Viewer.cameraPose`（ホイールと「注視点を見る」だけが変える）。
  readonly cameraPositionX: number;
  readonly cameraPositionY: number;
  readonly cameraPositionZ: number;
  readonly cameraPitchDegrees: number;
  readonly cameraYawDegrees: number;
  readonly background: string;
  /** 平行光の色（CSS の色表記）。 */
  readonly lightColor: string;
  readonly lightIntensity: number;
  readonly ambientColor: string;
  readonly ambient: number;
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
  // 口形は**焼いたプリセット 1 本**なので、手で立てる量は表情のスライダーと同じ入れ物
  // （`PanelState.expressions`）に入る。ここに持つのは連続再生の速さとループだけ。
  //
  // **「今 連続再生しているか」もここに持たない。** それは値ではなく駆動源の状態で、正本は
  // `VisemeDriver`（再生ボタンが押した / 終端で止まった、をあちらが知っている）。ビューの調整値
  // として持つと、終端で止まったときに設定の側が古くなる。
  readonly visemeFadeSeconds: number;
  readonly visemeHoldSeconds: number;
  /** 連続再生を お の次に あ へ戻すか。 */
  readonly visemeLoop: boolean;
}

// シーンを読む前の姿勢（注視点は眼の高さの既定）。シーンがあるときは頭部の中心を見るので、
// `Viewer.resetView` が同じ形で作り直す。**ここに座標を書き写さない。**
const DEFAULT_CAMERA_POSE = cameraPoseAt([0, TARGET_HEIGHT_METERS, 0]);

export const DEFAULT_VIEW_SETTINGS: ViewSettings = {
  fovDegrees: DEFAULT_FOV_DEGREES,
  cameraPositionX: DEFAULT_CAMERA_POSE.position[0],
  cameraPositionY: DEFAULT_CAMERA_POSE.position[1],
  cameraPositionZ: DEFAULT_CAMERA_POSE.position[2],
  cameraPitchDegrees: DEFAULT_CAMERA_POSE.pitchDegrees,
  cameraYawDegrees: DEFAULT_CAMERA_POSE.yawDegrees,
  background: DEFAULT_BACKGROUND,
  lightColor: DEFAULT_LIGHT_COLOR,
  lightIntensity: DEFAULT_LIGHT_INTENSITY,
  ambientColor: DEFAULT_AMBIENT_COLOR,
  ambient: AMBIENT_LIGHT,
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
  visemeFadeSeconds: VISEME_FADE_SECONDS,
  visemeHoldSeconds: VISEME_HOLD_SECONDS,
  visemeLoop: true,
};
