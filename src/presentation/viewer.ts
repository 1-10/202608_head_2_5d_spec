// 3D 確認ビュー（three.js）。
//
// **絵の正本は Unity 側（1-10/2607_Obayashi_Avatar_Mockup_3DGS の `Assets/Sandbox/Ooba/GNM`）**で、
// デスクトップ側（1-10/2608_Obayashi_GNMHeadExporter）の 3D ビューではない。書き出した guest を実際に
// 組み立てて画にするのは Unity なので、ここの絵が違うと「web で見て良かったが Unity で崩れる」が起きる。
//
// | | 値 | 正本 |
// |:--|:--|:--|
// | 投影 | 透視 FOV 20° / 距離 1.3m / 注視点 y=0.297m | `Scenes/Viewer.unity` の `MainCamera` |
// | 光 | 平行光 1 灯 + 環境光 | 同 `DirectionalLight` |
// | 背景 | `#26292e` | 同 `MainCamera` の `m_BackGroundColor` |
// | 法線 | 写真を貼る領域は **+Z 固定**、口腔内は**実法線** | `GnmHeadInstance.FlattenNormals` |
// | 髪 | alpha clip 0.3 | `MT_GnmHairTransparent` の `_Cutoff` |
//
// **肌・眼球・髪の法線を実法線にしない。** 写真の陰影と陰影が二重に掛かる。逆に**口腔内は実法線**で
// 描く — 写真を持たない単色なので二重に掛かる影が無く、+Z 固定だと開口時に歯・歯茎・舌が真っ平らな
// 切り絵に見える。切り分けの正本は `domain/preview/normals` と Unity 側 `FlattenNormals`。
//
// 法線はメッシュ空間で作って**位置と同じ LBS の回転を掛ける**。シェーダ側で +Z の定数に置き換える形は
// 使えない（首を回しても陰影が動かなくなる）。
//
// **光はワールド固定。** Unity 側も旧 web 版もワールドに置いた平行光で、カメラを動かしても顔に当たる
// 向きは変わらない。オブジェクト行列は単位なので、頂点法線がそのままワールド法線になる — 視点座標へ
// 移す変換は要らない（`normalMatrix` を掛けると光が画面に貼り付いて、周回しても陰影が動かなくなる）。
//
// 旧 web 版から残したもの: 首と視線のドラッグ操作（0.25°/px）・自動まばたき・ワイヤーフレーム・
// 背景色・FOV と距離の調整。Unity 側に無いが、写真 1 枚から起こした頭を確認するのに効く。
//
// 不透明を先に、半透明を後に描く。半透明は深度を読むが書かず、重なりは**三角形を奥→手前に並べ替えて**
// 色を決める（深度書き込みを止めて描くので、描く順がそのまま色になる）。
//
// **層ごとにテクスチャを外せる。** テクスチャを消すと `baseColor` と陰影だけが残るので、絵の中の
// 暗い所が「写真にそう写っていた」のか「面が無い・法線が逆・前後関係が壊れている」のかを分けられる。

import * as THREE from 'three';
import { AlphaImage, RgbImage } from '../domain/contract';
import { GnmPreviewAsset, LAYER_ORDER } from '../domain/preview/asset';
import {
  BLINK_PRESET_NAMES,
  BlinkState,
  ExpressionPlayMode,
  ExpressionPlayback,
  FADE_SECONDS,
  HOLD_SECONDS,
  IDLE_PLAYBACK,
  addExpression,
  advanceBlink,
  advancePlayback,
  startBlink,
} from '../domain/preview/expression';
import {
  DEGREES_PER_PIXEL,
  HeadPose,
  NECK_SHARE,
  NEUTRAL_POSE,
  applyRigidTransform,
  clampPose,
  followPointerPose,
  jointLocalRotations,
  jointRestPositions,
  jointSkinMatrices,
  rigidTransformFor,
  skinVertices,
} from '../domain/preview/pose';
import {
  flattenNormals,
  recalculateNormals,
  rotateNormals,
  skinNormals,
} from '../domain/preview/normals';
import {
  PreviewMesh,
  PreviewScene,
  drawPasses,
  gatherVertexVectors,
  isTransparent,
  sceneLayerNames,
} from '../domain/preview/scene';

/** 画角（度）。正本は Unity 側 `MainCamera` の `field of view`。 */
export const DEFAULT_FOV_DEGREES = 20;
export const MINIMUM_FOV_DEGREES = 10;
export const MAXIMUM_FOV_DEGREES = 60;

/** 注視点からの距離（メートル）。正本は同カメラの z。 */
export const DEFAULT_DISTANCE_METERS = 1.3;
export const MINIMUM_DISTANCE_METERS = 0.35;
export const MAXIMUM_DISTANCE_METERS = 3;

/** 注視点の高さ（メートル）。正本は同カメラの y（GNM の眼の高さ）。 */
export const TARGET_HEIGHT_METERS = 0.297;

/** near / far。正本は同カメラ。 */
const NEAR_PLANE = 0.01;
const FAR_PLANE = 20;

export const MAXIMUM_ZOOM = 5.0;
export const MINIMUM_ZOOM = 0.3;

/**
 * 環境光の量。
 *
 * Unity 側は skybox の SH を固定方向で引くので解析的には合わせられない。旧 web 版の
 * `AmbientLight 0.65` に合わせてある（下げると影側が締まるが、写真に焼き込まれた陰影の上へさらに
 * 陰影が乗るので、**写真をそのまま見たいならここは高い方が正しい**）。
 */
export const AMBIENT_LIGHT = 0.65;

/**
 * 光の向き（**GNM 空間**・光源へ向かうベクトル）。
 *
 * Unity 側 `DirectionalLight` の向きを GNM 空間へ写した値（Unity 空間は GNM 空間の X を反転した
 * 左手系なので x の符号が入れ替わる）。上・前・**被写体から見て右**から当たる。
 */
export const LIGHT_DIRECTION: readonly [number, number, number] = [-0.2802, 0.5736, 0.7698];

/** 背景色。正本は Unity 側 `MainCamera` の `m_BackGroundColor`。 */
export const DEFAULT_BACKGROUND = '#26292e';

/** カメラ周回のドラッグ 1 画素あたりの回転（ラジアン）。 */
const RADIANS_PER_PIXEL = 0.01;

/** 周回の pitch の上限（ラジアン）。真上・真下を越えて回さない。 */
const ORBIT_PITCH_LIMIT = 1.45;

/** 髪の裾を描かない閾値。正本は `MT_GnmHairTransparent` の `_Cutoff`。 */
const HAIR_ALPHA_CUTOFF = 0.3;

/** ワイヤーフレームの線の色。 */
const WIREFRAME_COLOR = 0x66ff99;

/** 不透明メッシュは重心を持たない（並べ替えないので計算する理由が無い）。 */
const EMPTY_CENTROIDS = new Float32Array(0);

/**
 * 数字キー → 表示を切り替える層。並びは `LAYER_ORDER`。
 */
export const LAYER_KEYS: Readonly<Record<string, string>> = {
  Digit1: LAYER_ORDER[0],
  Digit2: LAYER_ORDER[1],
  Digit3: LAYER_ORDER[2],
  Digit4: LAYER_ORDER[3],
};

/**
 * キー → テクスチャを切り替える層。並びは `LAYER_ORDER`（数字キーと同じ順）。
 *
 * **修飾キー（Shift + 数字）にしない。** キーボード配列によって Shift + 1 が届く形が変わるので、
 * 配列に依らない単独キーにする。
 */
export const TEXTURE_KEYS: Readonly<Record<string, string>> = {
  KeyA: LAYER_ORDER[0],
  KeyS: LAYER_ORDER[1],
  KeyD: LAYER_ORDER[2],
  KeyF: LAYER_ORDER[3],
};

/** 全部のテクスチャをまとめて切り替えるキー。 */
export const ALL_TEXTURES_KEY = 'KeyT';

/** 正面・等倍・無表情に戻すキー。 */
export const RESET_KEY = 'KeyR';

/** ワイヤーフレームを切り替えるキー。 */
export const WIREFRAME_KEY = 'KeyW';

const VERTEX_SHADER = `
varying vec2 vUv;
varying vec3 vNormal;

void main() {
  // 光はワールド固定。オブジェクト行列は単位なので、これがそのままワールド法線。
  vNormal = normal;
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const FRAGMENT_SHADER = `
uniform sampler2D uTexture;
uniform bool uUseTexture;
uniform bool uAlphaTest;
uniform vec4 uBaseColor;
varying vec2 vUv;
varying vec3 vNormal;

const vec3 LIGHT_DIRECTION = normalize(vec3(${LIGHT_DIRECTION[0]}, ${LIGHT_DIRECTION[1]}, ${LIGHT_DIRECTION[2]}));
const float AMBIENT = ${AMBIENT_LIGHT};
const float ALPHA_CUTOFF = ${HAIR_ALPHA_CUTOFF};

void main() {
  vec4 albedo = uUseTexture ? texture2D(uTexture, vUv) : uBaseColor;
  // 髪の薄い裾は描かず、頭皮へ巻き込んだシェルの継ぎ目を隠す。
  if (uAlphaTest && albedo.a < ALPHA_CUTOFF) discard;
  // 裏面は法線が逆を向く。反転しないと環境光だけの黒になり、穴と見分けが付かない。
  vec3 normal = gl_FrontFacing ? normalize(vNormal) : -normalize(vNormal);
  float light = AMBIENT + (1.0 - AMBIENT) * max(dot(normal, LIGHT_DIRECTION), 0.0);
  gl_FragColor = vec4(albedo.rgb * light, albedo.a);
}
`;

/**
 * RGB + alpha を sRGB のテクスチャにする（`domain/contract.COLOR_SPACE` の申告と同じ）。
 *
 * **行の反転はここで済ませる。** アトラスの行 0 は v = 1 側（`domain/contract` の座標規約）で、GL の
 * UV は下が v = 0。`flipY` に任せると、`UNPACK_FLIP_Y_WEBGL` が生の配列に効くかどうかという実装差に
 * 結果が乗る（`DataTexture` は `flipY` の既定が false でもある）。**ここを間違えると顔が上下逆に
 * 貼られる**ので、配列を作るときに 1 回だけ反転して `flipY` は false のままにする。デスクトップ側も
 * 同じ理由で numpy 側（`[::-1]`）で反転している。
 */
function textureFrom(image: RgbImage, alpha: AlphaImage | null): THREE.DataTexture {
  const data = new Uint8Array(image.width * image.height * 4);
  for (let row = 0; row < image.height; row++) {
    const sourceRow = image.height - 1 - row;
    for (let column = 0; column < image.width; column++) {
      const source = sourceRow * image.width + column;
      const target = row * image.width + column;
      data[target * 4] = image.data[source * 3];
      data[target * 4 + 1] = image.data[source * 3 + 1];
      data[target * 4 + 2] = image.data[source * 3 + 2];
      data[target * 4 + 3] = alpha === null ? 255 : alpha.data[source];
    }
  }
  const texture = new THREE.DataTexture(data, image.width, image.height, THREE.RGBAFormat);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.flipY = false;
  texture.generateMipmaps = true;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.needsUpdate = true;
  return texture;
}

/** 首・視線・表情を動かすのに要る入力。シーンと一緒に渡す。 */
export interface PreviewAnimation {
  readonly preview: GnmPreviewAsset;
  /** bind 姿勢（identity を当てただけ・無表情）の頂点。(頂点数, 3) */
  readonly restVertices: Float64Array;
  /** フィットで求めた identity 係数（ジョイント位置を作り直すのに要る）。 */
  readonly identity: Float64Array;
  /** 法線を作り直すのに要るトポロジ（頭部メッシュのもの）。 */
  readonly triangles: Uint32Array;
  readonly uvSplitSource: Uint32Array;
}

interface GpuMesh {
  readonly source: PreviewMesh;
  readonly object: THREE.Mesh;
  readonly wireframe: THREE.Mesh;
  readonly wireframeMaterial: THREE.MeshBasicMaterial;
  readonly material: THREE.ShaderMaterial;
  readonly texture: THREE.DataTexture | null;
  readonly positionAttribute: THREE.BufferAttribute;
  readonly normalAttribute: THREE.BufferAttribute;
  /** 三角形ごとの重心（半透明の並べ替えに使う）。 */
  readonly centroids: Float32Array;
}

export class Viewer {
  private readonly scene = new THREE.Scene();
  private readonly camera: THREE.PerspectiveCamera;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly container: HTMLElement;
  private readonly meshes: GpuMesh[] = [];
  private previewScene: PreviewScene | null = null;
  private animation: PreviewAnimation | null = null;
  /** identity から作った bind 姿勢のジョイント位置（identity が変わるまで使い回す）。 */
  private jointRest: Float64Array | null = null;
  /** 髪シェルの bind 姿勢の位置（剛体変換の元）。 */
  private hairRest: Float32Array | null = null;
  private workingVertices: Float64Array | null = null;
  /** メッシュ空間の法線（実法線 → 写真領域を +Z へ落としたもの）。 */
  private restNormals: Float32Array | null = null;
  /** LBS を掛けた後の法線。 */
  private skinnedNormals: Float32Array | null = null;
  /** 内角重みの集約に使う作業領域（複製前の頂点数ぶん）。 */
  private normalScratch: Float64Array | null = null;
  /** 髪シェルのメッシュ空間の法線。 */
  private hairRestNormals: Float32Array | null = null;
  private expressionWeights: Float64Array | null = null;
  /**
   * 前のフレームで当てた重み。
   *
   * **「変わったか」は重みの前後比較で決める。** 立てたときだけ dirty にすると、まばたきや
   * 自動再生が 0 へ戻るフレームで作り直しが走らず、目が半分閉じたまま固まる。
   */
  private appliedWeights: Float64Array | null = null;
  private manualWeights: Float64Array | null = null;
  private blinkPresetIndices: number[] = [];

  private hidden = new Set<string>();
  /** テクスチャを**外している**層。`hidden` と同じ「除外集合」の持ち方に揃える。 */
  private untextured = new Set<string>();

  private playback: ExpressionPlayback = IDLE_PLAYBACK;
  private blink: BlinkState = startBlink();
  private pointerX = 0;
  private pointerY = 0;
  private lastFrameMs: number | null = null;
  private poseDirty = true;
  /** 同じ数を何度も console へ出さないための直近値。 */
  private reportedUndeterminedNormals = 0;

  /** カメラ周回の角度（ラジアン）。頭の向き（`headPose`）とは別。 */
  orbitYaw = 0;
  orbitPitch = 0;
  zoom = 1;
  pan: [number, number] = [0, 0];
  fovDegrees = DEFAULT_FOV_DEGREES;
  distanceMeters = DEFAULT_DISTANCE_METERS;

  /** 首と視線（度）。可動域は `domain/preview/pose` が持つ。 */
  headPose: HeadPose = NEUTRAL_POSE;
  /** マウス位置で首と視線を動かす。 */
  followPointer = false;
  /** 表情の自動再生。 */
  playMode: ExpressionPlayMode = 'off';
  /** 自動まばたき（旧 web 版から残した機能。Unity 側には無い）。 */
  blinkEnabled = true;
  /** 表情の強さ（プリセットの重みに掛ける）。 */
  expressionIntensity = 1;
  /** 首へ配る割合。残りが頭。正本は Unity 側 `GnmHeadPoseController._neckShare`。 */
  neckShare = NECK_SHARE;
  /** 自動再生の立ち上がり秒。正本は同 `GnmExpressionPlayer._fadeSeconds`。 */
  fadeSeconds = FADE_SECONDS;
  /** 自動再生の保持秒。正本は同 `_holdSeconds`。 */
  holdSeconds = HOLD_SECONDS;
  /** ワイヤーフレームの重畳。 */
  showWireframe = false;

  /** 今かかっている自動再生のプリセット名（無ければ null）。 */
  currentExpression: string | null = null;

  /** 視点や表示状態が変わったときに呼ばれる（UI の同期用）。 */
  onViewChanged: (() => void) | null = null;

  constructor(container: HTMLElement) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.setBackground(DEFAULT_BACKGROUND);
    // テクスチャは sRGB、出力も sRGB。
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera(DEFAULT_FOV_DEGREES, 1, NEAR_PLANE, FAR_PLANE);
    this.resize();
    // window の resize だけだとコンテナ単独のレイアウト変化を取りこぼす。
    new ResizeObserver(() => this.resize()).observe(container);
    this.attachInput();
  }

  /** 背景色を差し替える（CSS の色表記）。 */
  setBackground(color: string): void {
    this.renderer.setClearColor(new THREE.Color(color), 1);
  }

  /** シーンを差し替える。**表示状態と姿勢は初期化する**（層の集合がシーンで変わる）。 */
  setScene(scene: PreviewScene, animation: PreviewAnimation): void {
    this.dispose();
    this.previewScene = scene;
    this.animation = animation;
    this.jointRest = jointRestPositions(animation.preview, animation.identity);
    this.workingVertices = new Float64Array(animation.restVertices.length);
    this.restNormals = new Float32Array(animation.restVertices.length);
    this.skinnedNormals = new Float32Array(animation.restVertices.length);
    this.normalScratch = new Float64Array(scene.normalPlan.weldedCount * 3);
    // +Z 固定の頂点は `recalculateNormals` が触れないので、ここで一度だけ埋める。
    for (let vertex = 0; vertex < animation.preview.vertexCount; vertex++) {
      this.restNormals[vertex * 3 + 2] = 1;
    }
    this.expressionWeights = new Float64Array(animation.preview.presetCount);
    this.appliedWeights = new Float64Array(animation.preview.presetCount);
    this.manualWeights = new Float64Array(animation.preview.presetCount);
    this.blinkPresetIndices = BLINK_PRESET_NAMES.map((name) =>
      animation.preview.expressionPresetNames.indexOf(name),
    ).filter((index) => index >= 0);
    this.hidden = new Set();
    this.untextured = new Set();
    this.playback = IDLE_PLAYBACK;
    this.blink = startBlink();
    this.currentExpression = null;
    this.resetView();

    for (const mesh of scene.meshes) {
      if (mesh.sourceVertices === null) {
        this.hairRest = Float32Array.from(mesh.restPositions);
        // 髪は写真を貼るので +Z 固定（Unity 側もリアル表示では髪を平坦にする）。
        this.hairRestNormals = new Float32Array(mesh.restPositions.length);
        for (let vertex = 0; vertex < mesh.restPositions.length / 3; vertex++) {
          this.hairRestNormals[vertex * 3 + 2] = 1;
        }
      }
      const geometry = new THREE.BufferGeometry();
      const positionAttribute = new THREE.BufferAttribute(
        Float32Array.from(mesh.restPositions),
        3,
      );
      positionAttribute.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute('position', positionAttribute);
      const normalAttribute = new THREE.BufferAttribute(
        new Float32Array(mesh.restPositions.length),
        3,
      );
      normalAttribute.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute('normal', normalAttribute);
      geometry.setAttribute('uv', new THREE.BufferAttribute(mesh.uvs, 2));
      geometry.setIndex(new THREE.BufferAttribute(Uint32Array.from(mesh.triangles), 1));
      const transparent = isTransparent(mesh);
      const texture = mesh.texture === null ? null : textureFrom(mesh.texture, mesh.alpha);
      const material = new THREE.ShaderMaterial({
        vertexShader: VERTEX_SHADER,
        fragmentShader: FRAGMENT_SHADER,
        uniforms: {
          uTexture: { value: texture },
          uUseTexture: { value: texture !== null },
          uAlphaTest: { value: transparent },
          uBaseColor: {
            value: new THREE.Vector4(
              mesh.baseColor[0] / 255,
              mesh.baseColor[1] / 255,
              mesh.baseColor[2] / 255,
              1,
            ),
          },
        },
        // 開いた面を隠さない。
        side: THREE.DoubleSide,
        transparent,
        // 半透明は深度を読むが書かない（不透明が先に深度を埋めている）。
        depthWrite: !transparent,
        blending: transparent ? THREE.NormalBlending : THREE.NoBlending,
      });
      const object = new THREE.Mesh(geometry, material);
      object.name = mesh.name;
      object.matrixAutoUpdate = false;
      object.frustumCulled = false;
      // 不透明を先に、半透明を後に。
      object.renderOrder = transparent ? 1 : 0;
      this.scene.add(object);

      // ワイヤーフレームは**同じ geometry** に載せる（位置の更新が 1 回で済む）。
      const wireframeMaterial = new THREE.MeshBasicMaterial({
        color: WIREFRAME_COLOR,
        wireframe: true,
        transparent: true,
        opacity: 0.35,
        depthWrite: false,
      });
      const wireframe = new THREE.Mesh(geometry, wireframeMaterial);
      wireframe.matrixAutoUpdate = false;
      wireframe.frustumCulled = false;
      wireframe.renderOrder = 2;
      wireframe.visible = false;
      this.scene.add(wireframe);

      this.meshes.push({
        source: mesh,
        object,
        wireframe,
        wireframeMaterial,
        material,
        texture,
        positionAttribute,
        normalAttribute,
        centroids: transparent ? new Float32Array((mesh.triangles.length / 3) * 3) : EMPTY_CENTROIDS,
      });
    }
    this.applyVisibility();
    this.updateGeometry();
  }

  /** 正面・等倍・無表情に戻す。 */
  resetView(): void {
    this.orbitYaw = 0;
    this.orbitPitch = 0;
    this.zoom = 1;
    this.pan = [0, 0];
    this.fovDegrees = DEFAULT_FOV_DEGREES;
    this.distanceMeters = DEFAULT_DISTANCE_METERS;
    this.neckShare = NECK_SHARE;
    this.headPose = NEUTRAL_POSE;
    this.followPointer = false;
    this.manualWeights?.fill(0);
    this.playback = IDLE_PLAYBACK;
    this.currentExpression = null;
    this.poseDirty = true;
    this.onViewChanged?.();
  }

  /** 首と視線を指定する（マウス追従中は受け付けない。Unity 側と同じ扱い）。 */
  setHeadPose(pose: HeadPose): void {
    if (this.followPointer) return;
    this.headPose = clampPose(pose);
    this.poseDirty = true;
  }

  /** 手で立てる表情の重み。プリセット名で指定する。 */
  setManualExpression(name: string, weight: number): void {
    if (this.animation === null || this.manualWeights === null) return;
    const index = this.animation.preview.expressionPresetNames.indexOf(name);
    if (index < 0) return;
    this.manualWeights[index] = weight;
    this.poseDirty = true;
  }

  /** プリセット名の一覧（GUI がスライダーを作るのに使う）。 */
  expressionNames(): readonly string[] {
    return this.animation === null ? [] : this.animation.preview.expressionPresetNames;
  }

  /** 層の表示を切り替える。シーンに無い層でも状態は持つ（消えたら効かないだけ）。 */
  toggleLayer(layer: string): void {
    if (this.hidden.has(layer)) this.hidden.delete(layer);
    else this.hidden.add(layer);
    this.applyVisibility();
    this.onViewChanged?.();
  }

  setLayerVisible(layer: string, visible: boolean): void {
    if (visible) this.hidden.delete(layer);
    else this.hidden.add(layer);
    this.applyVisibility();
    this.onViewChanged?.();
  }

  /** その層のテクスチャを切り替える。OFF では `baseColor` と陰影だけになる。 */
  toggleLayerTexture(layer: string): void {
    if (this.untextured.has(layer)) this.untextured.delete(layer);
    else this.untextured.add(layer);
    this.applyVisibility();
    this.onViewChanged?.();
  }

  setLayerTextureEnabled(layer: string, enabled: boolean): void {
    if (enabled) this.untextured.delete(layer);
    else this.untextured.add(layer);
    this.applyVisibility();
    this.onViewChanged?.();
  }

  /** 全部まとめて切り替える。1 つでも貼っていれば全部外し、無ければ全部貼る。 */
  toggleAllTextures(): void {
    const layers = this.previewScene === null ? LAYER_ORDER : sceneLayerNames(this.previewScene);
    const anyTextured = layers.some((layer) => !this.untextured.has(layer));
    this.untextured = anyTextured ? new Set(layers) : new Set();
    this.applyVisibility();
    this.onViewChanged?.();
  }

  setWireframe(enabled: boolean): void {
    this.showWireframe = enabled;
    this.applyVisibility();
    this.onViewChanged?.();
  }

  /** (層の名前, 表示しているか) をシーンにある層について返す。 */
  layerStates(): [string, boolean][] {
    if (this.previewScene === null) return [];
    return sceneLayerNames(this.previewScene).map((layer) => [layer, !this.hidden.has(layer)]);
  }

  /** (層の名前, テクスチャを貼っているか) をシーンにある層について返す。 */
  textureStates(): [string, boolean][] {
    if (this.previewScene === null) return [];
    return sceneLayerNames(this.previewScene).map((layer) => [layer, !this.untextured.has(layer)]);
  }

  /** キー操作。層・テクスチャ・視点のリセットを扱い、それ以外は false を返す。 */
  handleKey(code: string): boolean {
    if (code in LAYER_KEYS) {
      this.toggleLayer(LAYER_KEYS[code]);
      return true;
    }
    if (code in TEXTURE_KEYS) {
      this.toggleLayerTexture(TEXTURE_KEYS[code]);
      return true;
    }
    if (code === ALL_TEXTURES_KEY) {
      this.toggleAllTextures();
      return true;
    }
    if (code === WIREFRAME_KEY) {
      this.setWireframe(!this.showWireframe);
      return true;
    }
    if (code === RESET_KEY) {
      this.resetView();
      return true;
    }
    return false;
  }

  private applyVisibility(): void {
    if (this.previewScene === null) return;
    const passes = drawPasses(this.previewScene, this.hidden);
    const visible = new Set([...passes.opaque, ...passes.transparent].map((mesh) => mesh.name));
    for (const gpu of this.meshes) {
      const shown = visible.has(gpu.source.name);
      gpu.object.visible = shown;
      gpu.wireframe.visible = shown && this.showWireframe;
      gpu.material.uniforms.uUseTexture.value =
        gpu.texture !== null && !this.untextured.has(gpu.source.layer);
    }
  }

  resize(): void {
    // pixelRatio は毎回読み直す — ページズーム変更で devicePixelRatio が変わるため。
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    const width = this.container.clientWidth || 1;
    const height = this.container.clientHeight || 1;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  /** 1 フレーム進めて描く。経過時間はここで測る（呼び側が時計を持たなくてよい形にする）。 */
  render(): void {
    const now = performance.now();
    const deltaSeconds =
      this.lastFrameMs === null ? 0 : Math.min(0.1, (now - this.lastFrameMs) / 1000);
    this.lastFrameMs = now;

    if (this.previewScene === null) {
      this.renderer.clear();
      return;
    }
    this.advanceAnimation(deltaSeconds);
    this.placeCamera();
    for (const gpu of this.meshes) {
      if (gpu.object.visible && isTransparent(gpu.source)) this.sortBackToFront(gpu);
    }
    this.renderer.render(this.scene, this.camera);
  }

  /** 自動再生・まばたき・マウス追従を進め、要るなら頂点を作り直す。 */
  private advanceAnimation(deltaSeconds: number): void {
    if (this.animation === null || this.expressionWeights === null) return;
    const preview = this.animation.preview;

    if (this.followPointer && deltaSeconds > 0) {
      this.headPose = followPointerPose(this.headPose, this.pointerX, this.pointerY, deltaSeconds);
      this.poseDirty = true;
    }

    const weights = this.expressionWeights;
    weights.fill(0);
    if (this.playMode === 'off') {
      if (this.manualWeights !== null) {
        for (let preset = 0; preset < weights.length; preset++) {
          weights[preset] = this.manualWeights[preset] * this.expressionIntensity;
        }
      }
      this.currentExpression = null;
    } else {
      // 自動再生中は手のスライダーを無視する。
      // **同時に立てるのは 1 本だけ**（加算変位なので重ねると顔が壊れる）。Unity 側と同じ扱い。
      const step = advancePlayback(
        this.playback,
        this.playMode,
        preview.presetCount,
        deltaSeconds,
        Math.random,
        this.fadeSeconds,
        this.holdSeconds,
      );
      this.playback = step.playback;
      if (step.index >= 0) {
        weights[step.index] = step.weight * this.expressionIntensity;
        this.currentExpression = preview.expressionPresetNames[step.index];
      }
    }

    if (this.blinkEnabled && this.blinkPresetIndices.length > 0) {
      const step = advanceBlink(this.blink, deltaSeconds);
      this.blink = step.state;
      for (const index of this.blinkPresetIndices) weights[index] += step.weight;
    }

    if (this.appliedWeights !== null) {
      for (let preset = 0; preset < weights.length; preset++) {
        if (weights[preset] !== this.appliedWeights[preset]) {
          this.poseDirty = true;
          break;
        }
      }
    }
    if (!this.poseDirty) return;
    this.poseDirty = false;
    this.appliedWeights?.set(weights);
    this.updateGeometry();
  }

  /** identity + 表情 + LBS を当てて GPU の位置と法線を書き換える。 */
  private updateGeometry(): void {
    if (
      this.animation === null ||
      this.previewScene === null ||
      this.workingVertices === null ||
      this.restNormals === null ||
      this.skinnedNormals === null ||
      this.expressionWeights === null ||
      this.jointRest === null
    ) {
      return;
    }
    const preview = this.animation.preview;
    this.workingVertices.set(this.animation.restVertices);
    addExpression(preview, this.workingVertices, this.expressionWeights);

    // 法線は**スキニングの前**のメッシュ空間で作る（Unity 側も Mesh に焼いて skinning へ通す）。
    const undetermined = recalculateNormals(
      this.workingVertices,
      this.animation.triangles,
      this.animation.uvSplitSource,
      this.previewScene.normalPlan,
      this.restNormals,
      this.normalScratch ?? undefined,
    );
    if (undetermined > 0 && undetermined !== this.reportedUndeterminedNormals) {
      this.reportedUndeterminedNormals = undetermined;
      console.warn(`面が打ち消して法線を決められない頂点が ${undetermined} 個あった`);
    }
    flattenNormals(this.restNormals, this.previewScene.normalPlan.keepReal);

    const skinMatrices = jointSkinMatrices(
      preview,
      this.jointRest,
      jointLocalRotations(preview, this.headPose, this.neckShare),
    );
    const skinned = skinVertices(preview, this.workingVertices, skinMatrices);
    skinNormals(preview, this.restNormals, skinMatrices, this.skinnedNormals);
    const hairTransform = rigidTransformFor(preview, skinMatrices, 'head');

    for (const gpu of this.meshes) {
      const target = gpu.positionAttribute.array as Float32Array;
      const normalTarget = gpu.normalAttribute.array as Float32Array;
      if (gpu.source.sourceVertices !== null) {
        gatherVertexVectors(gpu.source, skinned, target);
        gatherVertexVectors(gpu.source, this.skinnedNormals, normalTarget);
      } else if (this.hairRest !== null && this.hairRestNormals !== null) {
        target.set(applyRigidTransform(this.hairRest, hairTransform));
        normalTarget.set(rotateNormals(this.hairRestNormals, hairTransform));
      }
      gpu.positionAttribute.needsUpdate = true;
      gpu.normalAttribute.needsUpdate = true;
      // 重心は**半透明の並べ替えにしか使わない**。不透明まで毎フレーム計算すると、頭部の
      // 35,324 三角形ぶんが丸ごと無駄になる。
      if (!isTransparent(gpu.source)) continue;
      const triangleCount = gpu.source.triangles.length / 3;
      for (let triangle = 0; triangle < triangleCount; triangle++) {
        for (let axis = 0; axis < 3; axis++) {
          let total = 0;
          for (let corner = 0; corner < 3; corner++) {
            total += target[gpu.source.triangles[triangle * 3 + corner] * 3 + axis];
          }
          gpu.centroids[triangle * 3 + axis] = total / 3;
        }
      }
    }
  }

  /** 注視点のまわりを周回するカメラを置く。 */
  private placeCamera(): void {
    this.camera.fov = Math.min(
      MAXIMUM_FOV_DEGREES,
      Math.max(MINIMUM_FOV_DEGREES, this.fovDegrees),
    );
    this.camera.updateProjectionMatrix();
    const distance =
      Math.min(MAXIMUM_DISTANCE_METERS, Math.max(MINIMUM_DISTANCE_METERS, this.distanceMeters)) /
      this.zoom;
    const rotation = new THREE.Euler(this.orbitPitch, this.orbitYaw, 0, 'YXZ');
    const right = new THREE.Vector3(1, 0, 0).applyEuler(rotation);
    const up = new THREE.Vector3(0, 1, 0).applyEuler(rotation);
    const forward = new THREE.Vector3(0, 0, 1).applyEuler(rotation);
    // pan は画面内の平行移動。拡大したまま端を見るのに要る（拡大は中心のまま効くので、平行移動が
    // 無いと目・口の端が枠外に出て検査できない）。
    const halfHeight = distance * Math.tan((this.camera.fov * Math.PI) / 360);
    const target = new THREE.Vector3(0, TARGET_HEIGHT_METERS, 0);
    target.addScaledVector(right, -this.pan[0] * halfHeight * this.camera.aspect);
    target.addScaledVector(up, -this.pan[1] * halfHeight);
    this.camera.position.copy(target).addScaledVector(forward, distance);
    this.camera.up.copy(up);
    this.camera.lookAt(target);
    this.camera.updateMatrixWorld();
  }

  /**
   * 三角形を奥から手前の順に並べ替える。
   *
   * 半透明は深度書き込みを止めて描くので、重なり合う面の**描く順がそのまま色**になる。カメラから
   * 遠い順に描く。
   */
  private sortBackToFront(gpu: GpuMesh): void {
    const eye = this.camera.position;
    const triangleCount = gpu.centroids.length / 3;
    const depth = new Float64Array(triangleCount);
    for (let triangle = 0; triangle < triangleCount; triangle++) {
      const x = gpu.centroids[triangle * 3] - eye.x;
      const y = gpu.centroids[triangle * 3 + 1] - eye.y;
      const z = gpu.centroids[triangle * 3 + 2] - eye.z;
      depth[triangle] = -(x * x + y * y + z * z);
    }
    const order = Array.from({ length: triangleCount }, (_, index) => index).sort(
      (first, second) => depth[first] - depth[second],
    );
    const index = gpu.object.geometry.getIndex();
    if (index === null) return;
    const array = index.array as Uint32Array;
    for (let slot = 0; slot < triangleCount; slot++) {
      const triangle = order[slot];
      array[slot * 3] = gpu.source.triangles[triangle * 3];
      array[slot * 3 + 1] = gpu.source.triangles[triangle * 3 + 1];
      array[slot * 3 + 2] = gpu.source.triangles[triangle * 3 + 2];
    }
    index.needsUpdate = true;
  }

  private attachInput(): void {
    const element = this.renderer.domElement;
    let dragging: 'orbit' | 'pan' | 'head' | null = null;
    let lastX = 0;
    let lastY = 0;
    element.style.touchAction = 'none';
    element.addEventListener('pointerdown', (event) => {
      // Shift + 左 = 首と視線（旧 web 版のドラッグ操作）。左 = カメラ周回。右 = 平行移動。
      dragging = event.button === 0 ? (event.shiftKey ? 'head' : 'orbit') : 'pan';
      lastX = event.clientX;
      lastY = event.clientY;
      element.setPointerCapture(event.pointerId);
      this.container.classList.add('dragging');
    });
    element.addEventListener('pointermove', (event) => {
      const rect = element.getBoundingClientRect();
      this.pointerX = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * 2 - 1;
      this.pointerY = 1 - ((event.clientY - rect.top) / Math.max(rect.height, 1)) * 2;
      if (dragging === null) return;
      const deltaX = event.clientX - lastX;
      const deltaY = event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;
      if (dragging === 'orbit') {
        this.orbitYaw += deltaX * RADIANS_PER_PIXEL;
        this.orbitPitch = Math.min(
          ORBIT_PITCH_LIMIT,
          Math.max(-ORBIT_PITCH_LIMIT, this.orbitPitch + deltaY * RADIANS_PER_PIXEL),
        );
      } else if (dragging === 'head') {
        this.setHeadPose({
          headYawDegrees: this.headPose.headYawDegrees + deltaX * DEGREES_PER_PIXEL,
          // 画面を上へドラッグ（deltaY<0）すると見上げる。
          headPitchDegrees: this.headPose.headPitchDegrees - deltaY * DEGREES_PER_PIXEL,
          gazeYawDegrees: this.headPose.gazeYawDegrees,
          gazePitchDegrees: this.headPose.gazePitchDegrees,
        });
      } else {
        this.pan = [
          this.pan[0] + (2 * deltaX) / Math.max(this.container.clientWidth, 1),
          this.pan[1] - (2 * deltaY) / Math.max(this.container.clientHeight, 1),
        ];
      }
      this.onViewChanged?.();
    });
    const endDrag = (event: PointerEvent): void => {
      dragging = null;
      this.container.classList.remove('dragging');
      try {
        element.releasePointerCapture(event.pointerId);
      } catch {
        // ポインタが既に解放されている場合がある。
      }
    };
    element.addEventListener('pointerup', endDrag);
    element.addEventListener('pointercancel', endDrag);
    element.addEventListener(
      'wheel',
      (event) => {
        event.preventDefault();
        // ホイール 1 ノッチ = 120 なので、デスクトップ側と同じ `exp(delta / 1200)`。
        const factor = Math.exp(-event.deltaY / 1200);
        this.zoom = Math.min(MAXIMUM_ZOOM, Math.max(MINIMUM_ZOOM, this.zoom * factor));
        this.onViewChanged?.();
      },
      { passive: false },
    );
    // 右ドラッグでのコンテキストメニューを抑える（平行移動に使う）。
    element.addEventListener('contextmenu', (event) => event.preventDefault());
  }

  dispose(): void {
    for (const gpu of this.meshes) {
      this.scene.remove(gpu.object);
      this.scene.remove(gpu.wireframe);
      gpu.wireframeMaterial.dispose();
      gpu.object.geometry.dispose();
      gpu.texture?.dispose();
      gpu.material.dispose();
    }
    this.meshes.length = 0;
    this.previewScene = null;
    this.animation = null;
    this.jointRest = null;
    this.hairRest = null;
    this.workingVertices = null;
    this.restNormals = null;
    this.skinnedNormals = null;
    this.normalScratch = null;
    this.hairRestNormals = null;
    this.expressionWeights = null;
    this.appliedWeights = null;
    this.manualWeights = null;
  }
}
