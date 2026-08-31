// 3D ビュー（書き出し結果の確認用）。
//
// デスクトップ側の「3Dビューを開く」に当たる。**開いた時点でフィット結果からシーンを作り**、回転・
// 拡大して公式UVの貼られ方を確認できる。閉じている間はメッシュ選別も GPU 転送も行わない
// （`DebugSceneSource` が参照だけを束ねている）。
//
// **層を切れるようにする。** 検査は「誰がこの画素を持っているか」を切り分ける作業なので、肌・眼球・
// 口腔内・髪を個別に消せることが要る。層の定義は `domain/debugScene` が持つ。
//
// 不透明を先に、半透明を後に描く（`drawPasses`）。半透明を先に描くと、その後ろに隠れる不透明が
// 上書きしてしまう。

import * as THREE from 'three';
import { AlphaImage, RgbImage } from '../domain/contract';
import { DebugMesh, DebugScene, drawPasses, sceneBounds } from '../domain/debugScene';

/** テクスチャの色空間は sRGB（`domain/contract.COLOR_SPACE` の申告と同じ）。 */
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
  // アトラスの行 0 は v = 1 側（`domain/contract` の座標規約）。three.js の既定は下から上なので
  // 反転させる。**ここを間違えると顔が上下逆に貼られる。**
  texture.flipY = true;
  texture.generateMipmaps = true;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.needsUpdate = true;
  return texture;
}

function meshToObject(mesh: DebugMesh): THREE.Mesh {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(mesh.positions, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(mesh.uvs, 2));
  geometry.setIndex(new THREE.BufferAttribute(mesh.triangles, 1));
  geometry.computeVertexNormals();

  const transparent = mesh.alpha !== null;
  const material = new THREE.MeshStandardMaterial({
    color:
      mesh.texture === null
        ? new THREE.Color(
            mesh.baseColor[0] / 255,
            mesh.baseColor[1] / 255,
            mesh.baseColor[2] / 255,
          ).convertSRGBToLinear()
        : 0xffffff,
    map: mesh.texture === null ? null : textureFrom(mesh.texture, mesh.alpha),
    roughness: 0.95,
    metalness: 0,
    transparent,
    // 半透明は深度書き込みを止めて重ねる（不透明が先に深度を埋めている）。
    depthWrite: !transparent,
    alphaTest: transparent ? 0.02 : 0,
    side: THREE.DoubleSide,
  });
  const object = new THREE.Mesh(geometry, material);
  object.name = mesh.name;
  object.userData.layer = mesh.layer;
  return object;
}

export class Viewer {
  private readonly scene = new THREE.Scene();
  private readonly camera: THREE.PerspectiveCamera;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly container: HTMLElement;
  private group: THREE.Group | null = null;
  private debugScene: DebugScene | null = null;
  private hidden = new Set<string>();
  private radius = 0.2;
  private center = new THREE.Vector3();

  yawDeg = 0;
  pitchDeg = 0;
  zoom = 1;

  constructor(container: HTMLElement) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera(30, 1, 0.01, 50);
    this.scene.background = new THREE.Color(0x1b1d21);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const key = new THREE.DirectionalLight(0xffffff, 0.9);
    key.position.set(0.6, 0.8, 1.2);
    this.scene.add(key);
    this.resize();
    // window の resize だけだとコンテナ単独のレイアウト変化を取りこぼす。
    new ResizeObserver(() => this.resize()).observe(container);
  }

  /** シーンを差し替える（層の表示状態は保つ）。 */
  setScene(scene: DebugScene): void {
    this.dispose();
    this.debugScene = scene;
    const bounds = sceneBounds(scene);
    this.center.set(bounds.center[0], bounds.center[1], bounds.center[2]);
    this.radius = bounds.radius;
    this.rebuild();
  }

  /** 層の表示を切る。 */
  setHiddenLayers(hidden: Iterable<string>): void {
    this.hidden = new Set(hidden);
    this.rebuild();
  }

  private rebuild(): void {
    if (this.debugScene === null) return;
    if (this.group !== null) {
      this.scene.remove(this.group);
      disposeGroup(this.group);
    }
    const group = new THREE.Group();
    group.rotation.order = 'YXZ'; // yaw→pitch の順で直感的に合成されるよう明示する。
    const passes = drawPasses(this.debugScene, this.hidden);
    for (const mesh of passes.opaque) group.add(meshToObject(mesh));
    for (const mesh of passes.transparent) group.add(meshToObject(mesh));
    group.position.set(-this.center.x, -this.center.y, -this.center.z);
    const holder = new THREE.Group();
    holder.rotation.order = 'YXZ';
    holder.add(group);
    this.group = holder;
    this.scene.add(holder);
    this.applyOrientation();
  }

  applyOrientation(): void {
    if (this.group === null) return;
    this.group.rotation.y = THREE.MathUtils.degToRad(this.yawDeg);
    this.group.rotation.x = THREE.MathUtils.degToRad(this.pitchDeg);
    // 境界球が視野にちょうど収まる距離を基準に、ホイールの倍率を掛ける。
    const distance = (this.radius / Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2))) / this.zoom;
    this.camera.position.set(0, 0, distance * 1.2);
    this.camera.near = Math.max(distance * 0.01, 1e-3);
    this.camera.far = distance * 10;
    this.camera.lookAt(0, 0, 0);
    this.camera.updateProjectionMatrix();
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

  render(): void {
    this.renderer.render(this.scene, this.camera);
  }

  dispose(): void {
    if (this.group === null) return;
    this.scene.remove(this.group);
    disposeGroup(this.group);
    this.group = null;
    this.debugScene = null;
  }
}

function disposeGroup(group: THREE.Object3D): void {
  group.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return;
    object.geometry.dispose();
    const material = object.material as THREE.MeshStandardMaterial;
    material.map?.dispose();
    material.dispose();
  });
}
