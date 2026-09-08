// 調整パラメータと 3D ビューの操作パネル。
//
// **表示の都合しか持たない。** 書き出しの既定値・範囲・選べる値は `application/settings` が持ち、
// 3D ビューの既定値と範囲は `presentation/viewSettings` が持つ（入口ごとに違う既定を持つと、どちらで
// 動かしたかで結果が変わる）。ここがするのは、その値を lil-gui のコントロールへ結ぶことだけ。
//
// デスクトップ側は CLI と GUI の 2 入口を持ち、`--help` がパラメータの一覧を出す。ブラウザは入口が
// 1 つなので、**このパネルが一覧そのもの**になる。
//
// **パネルは 2 枚に分ける（書き出し = 左 / 3D ビュー = 右）。** 片方は書き出す zip の中身を変え、
// もう片方は見え方しか変えない。同じ列に混ざっていると「今どちらを触ったか」が分からず、書き出しの
// 値をビューの値だと思って動かす事故が起きる。
//
// 「首と視線」「表情」の節は Unity 側の Viewer パネル（`Viewer/GnmViewerUi`）と同じ並びにしてある。
// 同じものを同じ順で触れる方が、web と Unity を見比べるときに迷わない。「口形（あいうえお）」は
// あちらに無い節なので、その後ろへ独立して置く（表情の末尾へ足すと表情の一種に見える）。

import GUI from 'lil-gui';
import {
  DEFAULT_IDENTITY_CLIP,
  DEFAULT_SETTINGS,
  EYE_TEXTURE_SIZE_CHOICES,
  ExportSettings,
  MAXIMUM_ATLAS_FOREGROUND_EXPONENT,
  MAXIMUM_ATLAS_FOREGROUND_THRESHOLD,
  MAXIMUM_ATLAS_HARMONIC_SCREENING,
  MAXIMUM_DISAGREEMENT_SCALE,
  MAXIMUM_HAIR_LIFT_MM,
  MAXIMUM_HAIR_ROLLOFF_MM,
  MAXIMUM_IDENTITY_CLIP,
  MINIMUM_ATLAS_FOREGROUND_EXPONENT,
  MINIMUM_ATLAS_FOREGROUND_THRESHOLD,
  MINIMUM_ATLAS_HARMONIC_SCREENING,
  MINIMUM_DISAGREEMENT_SCALE,
  MINIMUM_HAIR_LIFT_MM,
  MINIMUM_HAIR_ROLLOFF_MM,
  MINIMUM_IDENTITY_CLIP,
  TEXTURE_SIZE_CHOICES,
} from '../application/settings';
import { LAYER_ORDER } from '../domain/preview/asset';
import {
  CameraPose,
  MAXIMUM_FOV_DEGREES,
  MAXIMUM_PITCH_DEGREES,
  MAXIMUM_YAW_DEGREES,
  MINIMUM_FOV_DEGREES,
} from '../domain/preview/camera';
import { ExpressionPlayMode } from '../domain/preview/expression';
import { isVisemePreset, visemeLabel } from '../domain/preview/viseme';
import {
  GAZE_LIMIT_DEGREES,
  HeadPose,
  PITCH_LIMIT_DEGREES,
  YAW_LIMIT_DEGREES,
} from '../domain/preview/pose';
import {
  ALL_TEXTURES_KEY,
  LAYER_KEYS,
  MAXIMUM_AMBIENT,
  MAXIMUM_LIGHT_INTENSITY,
  MINIMUM_AMBIENT,
  MINIMUM_LIGHT_INTENSITY,
  RESET_KEY,
  TEXTURE_KEYS,
  WIREFRAME_KEY,
} from './viewer';
import {
  DEFAULT_VIEW_SETTINGS,
  MAXIMUM_EXPRESSION_INTENSITY,
  MAXIMUM_FADE_SECONDS,
  MAXIMUM_HOLD_SECONDS,
  MAXIMUM_VISEME_FADE_SECONDS,
  MAXIMUM_VISEME_HOLD_SECONDS,
  MINIMUM_EXPRESSION_INTENSITY,
  MINIMUM_FADE_SECONDS,
  MINIMUM_HOLD_SECONDS,
  MINIMUM_VISEME_FADE_SECONDS,
  MINIMUM_VISEME_HOLD_SECONDS,
  PLAY_MODES,
  PLAY_MODE_LABELS,
  ViewSettings,
} from './viewSettings';

/** 層の日本語ラベル（デスクトップ側の `LAYER_LABELS` と同じ）。 */
export const LAYER_LABELS: Readonly<Record<string, string>> = {
  skin: '肌',
  eyes: '眼球',
  mouth: '口腔内',
  hair: '髪シェル',
};

/** パネルが編集する状態（`ExportSettings` + `ViewSettings` + 表示切り替え + 表情の重み）。 */
export interface PanelState {
  settings: {
    disagreementScale: number;
    /** identity 係数の上限を置くか。**置かない**のが既定（公式 GNM も置いていない）。 */
    clipEnabled: boolean;
    identityClip: number;
    skinAtlasSize: number;
    eyeTextureSize: number;
    hairTextureSize: number;
    atlasForegroundThreshold: number;
    atlasForegroundExponent: number;
    atlasHarmonicScreening: number;
    hairLiftMm: number;
    hairRolloffMm: number;
  };
  /** 3D ビューの調整値（書き出しには影響しない）。 */
  view: {
    fovDegrees: number;
    cameraPositionX: number;
    cameraPositionY: number;
    cameraPositionZ: number;
    cameraPitchDegrees: number;
    cameraYawDegrees: number;
    background: string;
    lightColor: string;
    lightIntensity: number;
    ambientColor: string;
    ambient: number;
    showWireframe: boolean;
    headYawDegrees: number;
    headPitchDegrees: number;
    gazeYawDegrees: number;
    gazePitchDegrees: number;
    neckShare: number;
    followPointer: boolean;
    playMode: ExpressionPlayMode;
    fadeSeconds: number;
    holdSeconds: number;
    expressionIntensity: number;
    blinkEnabled: boolean;
    visemeFadeSeconds: number;
    visemeHoldSeconds: number;
    visemeLoop: boolean;
    recordingLoop: boolean;
  };
  /**
   * 手で立てるプリセットの重み（プリセット名 → 0〜1）。アセットを読むまで空。
   *
   * **表情と口形を分けて持たない。** どちらもアセットの同じ並びの 1 本なので、入れ物を分けると
   * 「どちらに入っているか」を持ち回ることになる。分けるのはスライダーを置くフォルダだけ。
   */
  expressions: Record<string, number>;
  /** 層ごとの表示。 */
  visibleLayers: Record<string, boolean>;
  /** 層ごとのテクスチャ。OFF では `baseColor` と陰影だけになる。 */
  texturedLayers: Record<string, boolean>;
}

export function createPanelState(
  settings: ExportSettings = DEFAULT_SETTINGS,
  view: ViewSettings = DEFAULT_VIEW_SETTINGS,
): PanelState {
  const visibleLayers: Record<string, boolean> = {};
  const texturedLayers: Record<string, boolean> = {};
  for (const layer of LAYER_ORDER) {
    visibleLayers[layer] = true;
    texturedLayers[layer] = true;
  }
  return {
    settings: {
      disagreementScale: settings.disagreementScale,
      clipEnabled: settings.identityClip !== null,
      identityClip: settings.identityClip ?? DEFAULT_IDENTITY_CLIP,
      skinAtlasSize: settings.skinAtlasSize,
      eyeTextureSize: settings.eyeTextureSize,
      hairTextureSize: settings.hairTextureSize,
      atlasForegroundThreshold: settings.atlasForegroundThreshold,
      atlasForegroundExponent: settings.atlasForegroundExponent,
      atlasHarmonicScreening: settings.atlasHarmonicScreening,
      hairLiftMm: settings.hairLiftMm,
      hairRolloffMm: settings.hairRolloffMm,
    },
    view: { ...view },
    expressions: {},
    visibleLayers,
    texturedLayers,
  };
}

/** パネルの状態を `ExportSettings` へ移す。 */
export function toExportSettings(state: PanelState): ExportSettings {
  const { settings } = state;
  return {
    disagreementScale: settings.disagreementScale,
    identityClip: settings.clipEnabled ? settings.identityClip : null,
    skinAtlasSize: settings.skinAtlasSize,
    eyeTextureSize: settings.eyeTextureSize,
    hairTextureSize: settings.hairTextureSize,
    atlasForegroundThreshold: settings.atlasForegroundThreshold,
    atlasForegroundExponent: settings.atlasForegroundExponent,
    atlasHarmonicScreening: settings.atlasHarmonicScreening,
    hairLiftMm: settings.hairLiftMm,
    hairRolloffMm: settings.hairRolloffMm,
  };
}

/** パネルの状態を `ViewSettings` へ移す。 */
export function toViewSettings(state: PanelState): ViewSettings {
  return { ...state.view };
}

/** キーコードを人に見せる短い表記（`KeyA` → `A` / `Digit1` → `1`）。 */
function keyLabel(code: string): string {
  return code.replace(/^Key/, '').replace(/^Digit/, '');
}

export interface GuiCallbacks {
  onLayerVisibilityChanged: (layer: string, visible: boolean) => void;
  onLayerTextureChanged: (layer: string, enabled: boolean) => void;
  onAllTexturesToggled: () => void;
  onResetView: () => void;
  /** カメラを注視点（頭部の中心）へ向け直す。 */
  onLookAtTarget: () => void;
  /** カメラ（位置・回転・周回半径・画角）だけを既定へ戻す。 */
  onResetCamera: () => void;
  /** ビューの値が変わった（まとめて適用する）。 */
  onViewSettingsChanged: (view: ViewSettings) => void;
  /** 手で立てるプリセット（表情・口形とも）の重みが変わった。 */
  onExpressionChanged: (name: string, weight: number) => void;
  /** 口形の連続再生の再生 / 停止が押された。 */
  onVisemePlayToggled: () => void;
  /** 収録した表情アニメーションの再生 / 停止が押された。 */
  onRecordingPlayToggled: () => void;
  /** 収録した表情アニメーションの読み込みが押された。 */
  onRecordingLoad: () => void;
}

export interface GuiHandle {
  /** ビュー側で状態が変わったとき、パネルのチェックを合わせる。 */
  syncViewControls(
    layerStates: readonly [string, boolean][],
    textureStates: readonly [string, boolean][],
  ): void;
  /** ドラッグやマウス追従で動いた首と視線をスライダーへ戻す。 */
  syncHeadPose(pose: HeadPose): void;
  /** ドラッグ・ホイール・「注視点を見る」で動いたカメラを入力欄へ戻す。 */
  syncCameraPose(pose: CameraPose): void;
  /**
   * 口形の連続再生のボタンを今の状態に合わせる。
   *
   * **押した側で切り替えない。** ループ無しの連続再生は終端で自分から止まるので、押した回数で
   * ラベルを決めると止まった後も「停止」のまま残る。正本は `VisemeDriver.isPlaying`。
   */
  syncVisemePlayback(playing: boolean): void;
  /**
   * 収録した表情アニメーションのボタンと状態表示を合わせる。
   *
   * `syncVisemePlayback` と同じ理由で押した側が持たない — 終端で自分から止まるので、正本は
   * `RecordingPlayer.isPlaying`。`summary` は「未収録 / 3.4 秒・204 フレーム」のような 1 行。
   */
  syncRecording(playing: boolean, summary: string): void;
  /**
   * アセットを読んだ後にプリセットのスライダーを作る（名前はアセットが正本）。
   *
   * 表情と口形の振り分けは**名前**で決める（`domain/preview/viseme`）。一覧を受け取る側で持つと、
   * 焼くプリセットが増減したとき黙って古くなる。
   */
  setExpressionPresets(names: readonly string[]): void;
}

/** パネルを置く先。左が書き出し、右が 3D ビュー。 */
export interface GuiContainers {
  readonly exportPanel: HTMLElement;
  readonly viewPanel: HTMLElement;
}

export function setupGui(
  containers: GuiContainers,
  state: PanelState,
  callbacks: GuiCallbacks,
): GuiHandle {
  const gui = new GUI({
    container: containers.exportPanel,
    title: '書き出しパラメータ',
    width: 280,
  });
  const pushView = (): void => callbacks.onViewSettingsChanged(toViewSettings(state));

  const fit = gui.addFolder('フィット');
  fit
    .add(
      state.settings,
      'disagreementScale',
      MINIMUM_DISAGREEMENT_SCALE,
      MAXIMUM_DISAGREEMENT_SCALE,
      0.05,
    )
    .name('事前分布の倍率');
  fit.add(state.settings, 'clipEnabled').name('係数の上限を置く');
  fit
    .add(state.settings, 'identityClip', MINIMUM_IDENTITY_CLIP, MAXIMUM_IDENTITY_CLIP, 0.1)
    .name('identity 係数の上限');

  const texture = gui.addFolder('テクスチャ');
  texture.add(state.settings, 'skinAtlasSize', [...TEXTURE_SIZE_CHOICES]).name('肌アトラスの一辺');
  texture
    .add(state.settings, 'eyeTextureSize', [...EYE_TEXTURE_SIZE_CHOICES])
    .name('眼球テクスチャの一辺');
  texture
    .add(state.settings, 'hairTextureSize', [...TEXTURE_SIZE_CHOICES])
    .name('髪テクスチャの長辺');

  const atlas = gui.addFolder('アトラス');
  atlas
    .add(
      state.settings,
      'atlasForegroundThreshold',
      MINIMUM_ATLAS_FOREGROUND_THRESHOLD,
      MAXIMUM_ATLAS_FOREGROUND_THRESHOLD,
      0.01,
    )
    .name('前景しきい値');
  atlas
    .add(
      state.settings,
      'atlasForegroundExponent',
      MINIMUM_ATLAS_FOREGROUND_EXPONENT,
      MAXIMUM_ATLAS_FOREGROUND_EXPONENT,
      0.5,
    )
    .name('前景の指数');
  atlas
    .add(
      state.settings,
      'atlasHarmonicScreening',
      MINIMUM_ATLAS_HARMONIC_SCREENING,
      MAXIMUM_ATLAS_HARMONIC_SCREENING,
      0.05,
    )
    .name('harmonic screening');

  const hair = gui.addFolder('髪シェル');
  hair
    .add(state.settings, 'hairLiftMm', MINIMUM_HAIR_LIFT_MM, MAXIMUM_HAIR_LIFT_MM, 0.1)
    .name('持ち上げ (mm)');
  hair
    .add(state.settings, 'hairRolloffMm', MINIMUM_HAIR_ROLLOFF_MM, MAXIMUM_HAIR_ROLLOFF_MM, 0.1)
    .name('巻き込み (mm)');

  const view = new GUI({ container: containers.viewPanel, title: '3D ビュー', width: 280 });

  // カメラはフリーカメラ。**ワールドの transform をそのまま出して、そのまま打てる**（周回半径は
  // 出さない — ホイールと「注視点を見る」だけが変える導出向けの値で、パネルに出すと位置・回転と
  // 同じことを二重に持つことになる）。可動域と既定は `domain/preview/camera`。
  const camera = view.addFolder('カメラ');
  camera
    .add(state.view, 'fovDegrees', MINIMUM_FOV_DEGREES, MAXIMUM_FOV_DEGREES, 1)
    .name('画角 (°)')
    .onChange(pushView);
  const cameraControllers = [
    camera.add(state.view, 'cameraPositionX').step(0.001).name('位置 X (m)').onChange(pushView),
    camera.add(state.view, 'cameraPositionY').step(0.001).name('位置 Y (m)').onChange(pushView),
    camera.add(state.view, 'cameraPositionZ').step(0.001).name('位置 Z (m)').onChange(pushView),
    camera
      .add(state.view, 'cameraPitchDegrees', -MAXIMUM_PITCH_DEGREES, MAXIMUM_PITCH_DEGREES, 0.1)
      .name('回転 X (°)')
      .onChange(pushView),
    camera
      .add(state.view, 'cameraYawDegrees', -MAXIMUM_YAW_DEGREES, MAXIMUM_YAW_DEGREES, 0.1)
      .name('回転 Y (°)')
      .onChange(pushView),
  ];
  camera
    .add({ 注視点: callbacks.onLookAtTarget }, '注視点')
    .name('注視点（頭部中心）を見る');
  // **`R`（正面・無表情に戻す）とは別に置く。** あちらは光も首も表情もまとめて戻すので、打ち込んだ
  // transform を戻したいだけのときに使うと他の調整まで巻き添えになる。
  camera.add({ 初期値: callbacks.onResetCamera }, '初期値').name('カメラを初期値に戻す');
  camera.addColor(state.view, 'background').name('背景色').onChange(pushView);

  // 既定は Unity 側 `DirectionalLight` の `m_Color` / `m_Intensity`。環境光の量は旧 web 版の
  // `AmbientLight` に合わせてある（あちらは skybox の SH なので単色では合わせられない）。
  const light = view.addFolder('ライト');
  light.addColor(state.view, 'lightColor').name('平行光の色').onChange(pushView);
  light
    .add(
      state.view,
      'lightIntensity',
      MINIMUM_LIGHT_INTENSITY,
      MAXIMUM_LIGHT_INTENSITY,
      0.05,
    )
    .name('平行光の強さ')
    .onChange(pushView);
  light.addColor(state.view, 'ambientColor').name('環境光の色').onChange(pushView);
  light
    .add(state.view, 'ambient', MINIMUM_AMBIENT, MAXIMUM_AMBIENT, 0.01)
    .name('環境光の量')
    .onChange(pushView);
  const wireframeController = camera
    .add(state.view, 'showWireframe')
    .name(`ワイヤーフレーム   [${keyLabel(WIREFRAME_KEY)}]`)
    .onChange(pushView);

  // 可動域の上限は `domain/preview/pose` が持つ（Unity 側と同じ値）。
  const pose = view.addFolder('首と視線');
  const poseControllers = [
    pose
      .add(state.view, 'headYawDegrees', -YAW_LIMIT_DEGREES, YAW_LIMIT_DEGREES, 0.5)
      .name('首 yaw (°)')
      .onChange(pushView),
    pose
      .add(state.view, 'headPitchDegrees', -PITCH_LIMIT_DEGREES, PITCH_LIMIT_DEGREES, 0.5)
      .name('首 pitch (°)')
      .onChange(pushView),
    pose
      .add(state.view, 'gazeYawDegrees', -GAZE_LIMIT_DEGREES, GAZE_LIMIT_DEGREES, 0.5)
      .name('視線 yaw (°)')
      .onChange(pushView),
    pose
      .add(state.view, 'gazePitchDegrees', -GAZE_LIMIT_DEGREES, GAZE_LIMIT_DEGREES, 0.5)
      .name('視線 pitch (°)')
      .onChange(pushView),
  ];
  pose.add(state.view, 'neckShare', 0, 1, 0.05).name('首へ配る割合').onChange(pushView);
  pose.add(state.view, 'followPointer').name('マウス追従').onChange(pushView);

  const expression = view.addFolder('表情');
  // ラベルは `PLAY_MODE_LABELS` から作る（一覧をここに書き写すと増減で黙って古くなる）。
  const playModeChoices: Record<string, ExpressionPlayMode> = {};
  for (const mode of PLAY_MODES) playModeChoices[PLAY_MODE_LABELS[mode]] = mode;
  expression.add(state.view, 'playMode', playModeChoices).name('自動再生').onChange(pushView);
  expression
    .add(state.view, 'fadeSeconds', MINIMUM_FADE_SECONDS, MAXIMUM_FADE_SECONDS, 0.05)
    .name('立ち上がり (秒)')
    .onChange(pushView);
  expression
    .add(state.view, 'holdSeconds', MINIMUM_HOLD_SECONDS, MAXIMUM_HOLD_SECONDS, 0.05)
    .name('保持 (秒)')
    .onChange(pushView);
  expression
    .add(
      state.view,
      'expressionIntensity',
      MINIMUM_EXPRESSION_INTENSITY,
      MAXIMUM_EXPRESSION_INTENSITY,
      0.05,
    )
    .name('強さ')
    .onChange(pushView);
  expression.add(state.view, 'blinkEnabled').name('自動まばたき').onChange(pushView);
  // プリセットの名前はアセットが正本なので、ここに一覧を書かない（増減で黙って古くなる）。
  const presets = expression.addFolder('プリセット');
  presets.hide();

  // **「表情」とは別の節にする。** 口形はアセットへ焼いた 1 本のプリセットだが、駆動する理由が
  // 表情とは別（言葉を作る / 感情を作る）で、速さの既定も桁が違う。同じ節に混ぜると、20 本の
  // 表情の末尾に あいうえお が並んで「表情の一種」に見える。
  const viseme = view.addFolder('口形（あいうえお）');
  // **操作は全部出しっぱなしにする。** 出し入れすると、口形を触るたびにパネルの行数が変わって他の
  // 節の位置が動く（探し直しになる）。連続再生中に手のスライダーが効かないのは表情の自動再生と
  // 同じ扱いで、そちらもスライダーを隠していない。
  //
  // **「駆動」の選択は持たない。** 手で立てるかどうかはスライダーを動かすかどうかで決まり、連続
  // 再生かどうかは下のボタンで決まる。同じことを言う選択肢を別に置くと、ボタンと食い違ったときに
  // どちらが本当か分からなくなる。
  //
  // スライダーはアセットを読んでから作る（`setExpressionPresets`）ので、それまでは空の節を出さない。
  const visemePresets = viseme.addFolder('手動');
  visemePresets.hide();
  const play = viseme.addFolder('連続再生');

  // ラベルの正本は `VisemeDriver`（ループ無しなら終端で自分から止まるので、押した回数では決まらない）。
  const playbackController = play
    .add({ 再生: (): void => callbacks.onVisemePlayToggled() }, '再生')
    .name('再生');
  play
    .add(
      state.view,
      'visemeFadeSeconds',
      MINIMUM_VISEME_FADE_SECONDS,
      MAXIMUM_VISEME_FADE_SECONDS,
      0.01,
    )
    .name('立ち上がり (秒)')
    .onChange(pushView);
  play
    .add(
      state.view,
      'visemeHoldSeconds',
      MINIMUM_VISEME_HOLD_SECONDS,
      MAXIMUM_VISEME_HOLD_SECONDS,
      0.01,
    )
    .name('保持 (秒)')
    .onChange(pushView);
  play.add(state.view, 'visemeLoop').name('ループ').onChange(pushView);

  // **Webカメラのパネルには置かない。** 読み込みと再生は `<video>` を一切見ない（カメラが繋がって
  // いなくても動く）ので、カメラの箱に置くと「カメラの機能」に見える。録るのはあちら、持つのと
  // 再生するのはここ、という分け方。
  const recording = view.addFolder('表情アニメーション');
  const recordingSummary = { 状態: '未収録' };
  const recordingSummaryController = recording.add(recordingSummary, '状態').disable();
  const recordingPlayController = recording
    .add({ 再生: (): void => callbacks.onRecordingPlayToggled() }, '再生')
    .name('再生');
  recording.add(state.view, 'recordingLoop').name('ループ').onChange(pushView);
  recording.add({ 読み込み: (): void => callbacks.onRecordingLoad() }, '読み込み');

  const layerControllers = new Map<string, ReturnType<typeof view.add>>();
  const textureControllers = new Map<string, ReturnType<typeof view.add>>();
  const layerKeyOf = (layer: string): string =>
    Object.entries(LAYER_KEYS).find(([, value]) => value === layer)?.[0] ?? '';
  const textureKeyOf = (layer: string): string =>
    Object.entries(TEXTURE_KEYS).find(([, value]) => value === layer)?.[0] ?? '';

  const layers = view.addFolder('表示する層');
  for (const layer of LAYER_ORDER) {
    layerControllers.set(
      layer,
      layers
        .add(state.visibleLayers, layer)
        .name(`${LAYER_LABELS[layer] ?? layer}   [${keyLabel(layerKeyOf(layer))}]`)
        .onChange((value: boolean) => callbacks.onLayerVisibilityChanged(layer, value)),
    );
  }
  const textures = view.addFolder('テクスチャを貼る層');
  for (const layer of LAYER_ORDER) {
    textureControllers.set(
      layer,
      textures
        .add(state.texturedLayers, layer)
        .name(`${LAYER_LABELS[layer] ?? layer}   [${keyLabel(textureKeyOf(layer))}]`)
        .onChange((value: boolean) => callbacks.onLayerTextureChanged(layer, value)),
    );
  }
  const viewActions = {
    全テクスチャ: callbacks.onAllTexturesToggled,
    視点: callbacks.onResetView,
  };
  view.add(viewActions, '全テクスチャ').name(`全テクスチャを切り替え   [${keyLabel(ALL_TEXTURES_KEY)}]`);
  view.add(viewActions, '視点').name(`正面・無表情に戻す   [${keyLabel(RESET_KEY)}]`);

  return {
    syncViewControls(layerStates, textureStates) {
      for (const [layer, visible] of layerStates) {
        state.visibleLayers[layer] = visible;
        layerControllers.get(layer)?.updateDisplay();
      }
      for (const [layer, enabled] of textureStates) {
        state.texturedLayers[layer] = enabled;
        textureControllers.get(layer)?.updateDisplay();
      }
      wireframeController.updateDisplay();
    },
    syncVisemePlayback(playing) {
      playbackController.name(playing ? '停止' : '再生');
    },
    syncCameraPose(pose_) {
      state.view.cameraPositionX = pose_.position[0];
      state.view.cameraPositionY = pose_.position[1];
      state.view.cameraPositionZ = pose_.position[2];
      state.view.cameraPitchDegrees = pose_.pitchDegrees;
      state.view.cameraYawDegrees = pose_.yawDegrees;
      for (const controller of cameraControllers) controller.updateDisplay();
    },
    syncRecording(playing, summary) {
      recordingPlayController.name(playing ? '停止' : '再生');
      recordingSummary.状態 = summary;
      recordingSummaryController.updateDisplay();
    },
    syncHeadPose(pose_) {
      state.view.headYawDegrees = pose_.headYawDegrees;
      state.view.headPitchDegrees = pose_.headPitchDegrees;
      state.view.gazeYawDegrees = pose_.gazeYawDegrees;
      state.view.gazePitchDegrees = pose_.gazePitchDegrees;
      for (const controller of poseControllers) controller.updateDisplay();
    },
    setExpressionPresets(names) {
      presets.children.slice().forEach((child) => child.destroy());
      visemePresets.children.slice().forEach((child) => child.destroy());
      let expressionCount = 0;
      let visemeCount = 0;
      for (const name of names) {
        if (state.expressions[name] === undefined) state.expressions[name] = 0;
        const onChange = (value: number): void => callbacks.onExpressionChanged(name, value);
        if (isVisemePreset(name)) {
          visemePresets
            .add(state.expressions, name, 0, 1, 0.01)
            .name(visemeLabel(name))
            .onChange(onChange);
          visemeCount++;
        } else {
          presets.add(state.expressions, name, 0, 1, 0.01).onChange(onChange);
          expressionCount++;
        }
      }
      // 出すのは中身があるときだけ。空の節を出しても触れるものが無い（駆動では出し入れしない）。
      if (expressionCount > 0) presets.show();
      if (visemeCount > 0) visemePresets.show();
    },
  };
}
