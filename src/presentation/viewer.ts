// 3D 検査ビュー（書き出し結果の確認用）。
//
// **正射影・固定 +Z 法線。** 法線は幾何法線ではなく `vec3(0, 0, 1)` を視点の回転だけで回したもの。
// テクスチャは写真で、その人の陰影が既に焼き込まれている。幾何法線で陰影を作ると写真に無い影の帯が
// 乗り、継ぎ目の検査で「継ぎ目なのか照明なのか」が分からなくなる。回転で明るさが一様に変わるのは
// そのまま残す（形の当たりを見る手掛かり）。
//
// **拡大は x と y だけに掛ける。** 深度に掛けると、拡大したときに NDC z が [-1, 1] を越えて手前と
// 奥が near/far で切られる（正射影の前後関係は拡大と無関係なので、掛ける理由がそもそも無い）。
//
// **大きさの基準は境界球の半径**（箱の辺ではない）。どの向きへ回しても半径は変わらないので、
// `FRAME_FILL` と `DEPTH_FILL` が回転に関わらずそのまま余白の保証になる。
//
// 不透明を先に、半透明を後に描く。半透明は深度を読むが書かず、重なりは**三角形を奥→手前に並べ替えて**
// 色を決める（深度書き込みを止めて描くので、描く順がそのまま色になる）。
//
// **層ごとにテクスチャを外せる。** テクスチャを消すと `baseColor` と陰影だけが残るので、絵の中の
// 暗い所が「写真にそう写っていた」のか「面が無い・法線が逆・前後関係が壊れている」のかを分けられる。

import * as THREE from 'three';
import { AlphaImage, RgbImage } from '../domain/contract';
import {
  DebugMesh,
  DebugScene,
  LAYER_ORDER,
  drawPasses,
  isTransparent,
  sceneBounds,
  sceneLayerNames,
} from '../domain/debugScene';

/** zoom 1 のとき境界球が画面の短辺の半分に対して占める割合。残り 0.1 が余白。 */
export const FRAME_FILL = 0.9;

/** 境界球を NDC z の [-DEPTH_FILL, DEPTH_FILL] に収める。1 未満なら near/far に触れない。 */
export const DEPTH_FILL = 0.5;

export const MAXIMUM_ZOOM = 5.0;
export const MINIMUM_ZOOM = 0.3;

/** 影側の明るさ。真っ黒にしないのは、暗い面と穴を見分けるため。 */
export const AMBIENT_LIGHT = 0.35;

/** 背景色（デスクトップ側の `glClearColor` と同じ値）。 */
const BACKGROUND = new THREE.Color(37 / 255, 39 / 255, 43 / 255);

/** ドラッグ 1 画素あたりの回転（ラジアン）。 */
const RADIANS_PER_PIXEL = 0.01;

/** pitch の上限（ラジアン）。真上・真下を越えて回さない。 */
const PITCH_LIMIT = 1.45;

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

/**
 * 全部のテクスチャをまとめて切り替えるキー。
 *
 * 1 つでも貼っていれば全部外し、全部外れていれば全部貼る。「今どれが貼られているか」を数えずに
 * 押せる形にしてある。
 */
export const ALL_TEXTURES_KEY = 'KeyT';

/** 正面・等倍に戻すキー。 */
export const RESET_KEY = 'KeyR';

const VERTEX_SHADER = `
uniform mat4 uNormalRotation;
varying vec2 vUv;
varying vec3 vNormal;

// 法線は +Z 固定。テクスチャに焼き込まれた陰影の上へ幾何法線の影を重ねない。
const vec3 SURFACE_NORMAL = vec3(0.0, 0.0, 1.0);

void main() {
  vNormal = mat3(uNormalRotation) * SURFACE_NORMAL;
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

const vec3 LIGHT_DIRECTION = normalize(vec3(0.25, 0.55, 0.80));
const float AMBIENT = ${AMBIENT_LIGHT};

void main() {
  vec4 albedo = uUseTexture ? texture2D(uTexture, vUv) : uBaseColor;
  // 髪の薄い裾は描かず、頭皮へ巻き込んだシェルの継ぎ目を隠す。
  if (uAlphaTest && albedo.a < 0.3) discard;
  // 裏面は法線が逆を向く。反転しないと環境光だけの黒になり、穴と見分けが付かない。
  vec3 normal = gl_FrontFacing ? normalize(vNormal) : -normalize(vNormal);
  float light = AMBIENT + (1.0 - AMBIENT) * max(dot(normal, LIGHT_DIRECTION), 0.0);
  gl_FragColor = vec4(albedo.rgb * light, albedo.a);
}
`;

/**
 * GNM 空間 → クリップ空間の 4x4 を作る。
 *
 * @param pan 画面内の平行移動（NDC 単位）。**深度には掛けない**ので前後関係は動かない
 */
export function viewProjection(input: {
  center: readonly [number, number, number];
  radius: number;
  width: number;
  height: number;
  zoom: number;
  yaw: number;
  pitch: number;
  pan: readonly [number, number];
}): THREE.Matrix4 {
  if (!(input.radius > 0)) throw new Error(`境界球の半径が正でない: ${input.radius}`);
  const width = Math.max(input.width, 1);
  const height = Math.max(input.height, 1);
  const shortest = Math.min(width, height);
  const span = (FRAME_FILL * input.zoom) / input.radius;
  // z は符号を反転させる: GNM は +z が前（視点側）、NDC は +z が奥。
  const scale = new THREE.Matrix4().makeScale(
    (span * shortest) / width,
    (span * shortest) / height,
    -DEPTH_FILL / input.radius,
  );
  const rotation = new THREE.Matrix4().makeRotationFromEuler(
    // 縦軸で yaw、次に画面の横軸で pitch（「モデルを回してから傾ける」）。
    new THREE.Euler(input.pitch, input.yaw, 0, 'YXZ'),
  );
  const translation = new THREE.Matrix4().makeTranslation(
    -input.center[0],
    -input.center[1],
    -input.center[2],
  );
  const screen = new THREE.Matrix4().makeTranslation(input.pan[0], input.pan[1], 0);
  return screen.multiply(scale).multiply(rotation).multiply(translation);
}

/** 法線に掛ける回転（位置の行列は z を反転して拡大も掛けるので、法線には使えない）。 */
export function normalRotation(yaw: number, pitch: number): THREE.Matrix4 {
  return new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(pitch, yaw, 0, 'YXZ'));
}

/** RGB + alpha を sRGB のテクスチャにする（`domain/contract.COLOR_SPACE` の申告と同じ）。 */
function textureFrom(image: RgbImage, alpha: AlphaImage | null): THREE.DataTexture {
  const data = new Uint8Array(image.width * image.height * 4);
  for (let pixel = 0; pixel < image.width * image.height; pixel++) {
    data[pixel * 4] = image.data[pixel * 3];
    data[pixel * 4 + 1] = image.data[pixel * 3 + 1];
    data[pixel * 4 + 2] = image.data[pixel * 3 + 2];
    data[pixel * 4 + 3] = alpha === null ? 255 : alpha.data[pixel];
  }
  const texture = new THREE.DataTexture(data, image.width, image.height, THREE.RGBAFormat);
  texture.colorSpace = THREE.SRGBColorSpace;
  // アトラスの行 0 は v = 1 側（`domain/contract` の座標規約）。**ここを間違えると顔が上下逆に
  // 貼られる。** デスクトップ側は行を反転して GL へ上げているのと同じ向き。
  texture.flipY = true;
  texture.generateMipmaps = true;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.needsUpdate = true;
  return texture;
}

interface GpuMesh {
  readonly source: DebugMesh;
  readonly object: THREE.Mesh;
  readonly material: THREE.ShaderMaterial;
  readonly texture: THREE.DataTexture | null;
  /** 三角形ごとの重心（半透明の並べ替えに使う）。 */
  readonly centroids: Float32Array;
}

export class Viewer {
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.Camera();
  private readonly renderer: THREE.WebGLRenderer;
  private readonly container: HTMLElement;
  private readonly meshes: GpuMesh[] = [];
  private debugScene: DebugScene | null = null;
  private hidden = new Set<string>();
  /** テクスチャを**外している**層。`hidden` と同じ「除外集合」の持ち方に揃える。 */
  private untextured = new Set<string>();
  private center: [number, number, number] = [0, 0, 0];
  private radius = 1;

  yaw = 0;
  pitch = 0;
  zoom = 1;
  pan: [number, number] = [0, 0];

  /** 視点や表示状態が変わったときに呼ばれる（UI の同期用）。 */
  onViewChanged: (() => void) | null = null;

  constructor(container: HTMLElement) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setClearColor(BACKGROUND, 1);
    // テクスチャは sRGB、出力も sRGB。
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(this.renderer.domElement);
    this.camera.matrixAutoUpdate = false;
    this.camera.matrixWorld.identity();
    this.camera.matrixWorldInverse.identity();
    this.resize();
    // window の resize だけだとコンテナ単独のレイアウト変化を取りこぼす。
    new ResizeObserver(() => this.resize()).observe(container);
    this.attachInput();
  }

  /** シーンを差し替える。**表示状態は初期化する**（層の集合がシーンで変わる）。 */
  setScene(scene: DebugScene): void {
    this.dispose();
    this.debugScene = scene;
    const bounds = sceneBounds(scene);
    this.center = bounds.center;
    this.radius = bounds.radius;
    this.hidden = new Set();
    this.untextured = new Set();
    this.resetView();

    for (const mesh of scene.meshes) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(mesh.positions, 3));
      geometry.setAttribute('uv', new THREE.BufferAttribute(mesh.uvs, 2));
      geometry.setIndex(new THREE.BufferAttribute(Uint32Array.from(mesh.triangles), 1));
      const transparent = isTransparent(mesh);
      const texture = mesh.texture === null ? null : textureFrom(mesh.texture, mesh.alpha);
      const material = new THREE.ShaderMaterial({
        vertexShader: VERTEX_SHADER,
        fragmentShader: FRAGMENT_SHADER,
        uniforms: {
          uNormalRotation: { value: new THREE.Matrix4() },
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

      const triangleCount = mesh.triangles.length / 3;
      const centroids = new Float32Array(triangleCount * 3);
      for (let triangle = 0; triangle < triangleCount; triangle++) {
        for (let axis = 0; axis < 3; axis++) {
          let total = 0;
          for (let corner = 0; corner < 3; corner++) {
            total += mesh.positions[mesh.triangles[triangle * 3 + corner] * 3 + axis];
          }
          centroids[triangle * 3 + axis] = total / 3;
        }
      }
      this.meshes.push({ source: mesh, object, material, texture, centroids });
    }
    this.applyVisibility();
  }

  /** 正面・等倍に戻す。 */
  resetView(): void {
    this.yaw = 0;
    this.pitch = 0;
    this.zoom = 1;
    this.pan = [0, 0];
    this.onViewChanged?.();
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
    const layers = this.debugScene === null ? LAYER_ORDER : sceneLayerNames(this.debugScene);
    const anyTextured = layers.some((layer) => !this.untextured.has(layer));
    this.untextured = anyTextured ? new Set(layers) : new Set();
    this.applyVisibility();
    this.onViewChanged?.();
  }

  /** (層の名前, 表示しているか) をシーンにある層について返す。 */
  layerStates(): [string, boolean][] {
    if (this.debugScene === null) return [];
    return sceneLayerNames(this.debugScene).map((layer) => [layer, !this.hidden.has(layer)]);
  }

  /** (層の名前, テクスチャを貼っているか) をシーンにある層について返す。 */
  textureStates(): [string, boolean][] {
    if (this.debugScene === null) return [];
    return sceneLayerNames(this.debugScene).map((layer) => [layer, !this.untextured.has(layer)]);
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
    if (code === RESET_KEY) {
      this.resetView();
      return true;
    }
    return false;
  }

  private applyVisibility(): void {
    if (this.debugScene === null) return;
    const passes = drawPasses(this.debugScene, this.hidden);
    const visible = new Set([...passes.opaque, ...passes.transparent].map((mesh) => mesh.name));
    for (const gpu of this.meshes) {
      gpu.object.visible = visible.has(gpu.source.name);
      gpu.material.uniforms.uUseTexture.value =
        gpu.texture !== null && !this.untextured.has(gpu.source.layer);
    }
  }

  resize(): void {
    // pixelRatio は毎回読み直す — ページズーム変更で devicePixelRatio が変わるため。
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(this.container.clientWidth || 1, this.container.clientHeight || 1, false);
  }

  render(): void {
    if (this.debugScene === null) {
      this.renderer.clear();
      return;
    }
    const projection = viewProjection({
      center: this.center,
      radius: this.radius,
      width: this.container.clientWidth,
      height: this.container.clientHeight,
      zoom: this.zoom,
      yaw: this.yaw,
      pitch: this.pitch,
      pan: this.pan,
    });
    this.camera.projectionMatrix.copy(projection);
    this.camera.projectionMatrixInverse.copy(projection).invert();
    const rotation = normalRotation(this.yaw, this.pitch);
    for (const gpu of this.meshes) {
      gpu.material.uniforms.uNormalRotation.value = rotation;
      if (gpu.object.visible && isTransparent(gpu.source)) this.sortBackToFront(gpu, projection);
    }
    this.renderer.render(this.scene, this.camera);
  }

  /**
   * 三角形を奥から手前の順に並べ替える。
   *
   * 半透明は深度書き込みを止めて描くので、重なり合う面の**描く順がそのまま色**になる。NDC の z は
   * 奥ほど大きいので、昇順が奥→手前。
   */
  private sortBackToFront(gpu: GpuMesh, projection: THREE.Matrix4): void {
    const elements = projection.elements; // column-major
    const triangleCount = gpu.centroids.length / 3;
    const depth = new Float64Array(triangleCount);
    for (let triangle = 0; triangle < triangleCount; triangle++) {
      const x = gpu.centroids[triangle * 3];
      const y = gpu.centroids[triangle * 3 + 1];
      const z = gpu.centroids[triangle * 3 + 2];
      // 行 2（z 行）と点の内積。column-major なので row 2 は elements[2, 6, 10, 14]。
      depth[triangle] = elements[2] * x + elements[6] * y + elements[10] * z + elements[14];
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
    let dragging: 'rotate' | 'pan' | null = null;
    let lastX = 0;
    let lastY = 0;
    element.style.touchAction = 'none';
    element.addEventListener('pointerdown', (event) => {
      dragging = event.button === 0 ? 'rotate' : 'pan';
      lastX = event.clientX;
      lastY = event.clientY;
      element.setPointerCapture(event.pointerId);
      this.container.classList.add('dragging');
    });
    element.addEventListener('pointermove', (event) => {
      if (dragging === null) return;
      const deltaX = event.clientX - lastX;
      const deltaY = event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;
      if (dragging === 'rotate') {
        this.yaw += deltaX * RADIANS_PER_PIXEL;
        this.pitch = Math.min(
          PITCH_LIMIT,
          Math.max(-PITCH_LIMIT, this.pitch + deltaY * RADIANS_PER_PIXEL),
        );
      } else {
        // 画面内の平行移動。拡大したまま端を見るのに要る（拡大は中心のまま効くので、平行移動が
        // 無いと目・口の端が枠外に出て検査できない）。
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
      gpu.object.geometry.dispose();
      gpu.texture?.dispose();
      gpu.material.dispose();
    }
    this.meshes.length = 0;
    this.debugScene = null;
  }
}
