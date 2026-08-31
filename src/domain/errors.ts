// パイプラインの失敗語彙。
//
// 各段が「続けられない」と判断したときに投げる例外をここに集める。domain に置くのは
// 2つの理由から: 失敗の意味づけは計算の一部（どこまでが想定内の失敗か）であり、
// 上のレイヤー全て（application / infrastructure）から見えている必要があるため。
//
// 素の Error を投げない。呼び出し側が「写真を選び直させる失敗」と「環境が足りていない
// 失敗」を区別できないと、UI が出せるメッセージを決められない。

/**
 * このアプリが想定している失敗の基底。
 *
 * これを継承しない例外が出たらバグとして扱う（想定外の状態に入っている）。
 */
export class ExporterError extends Error {
  constructor(message: string) {
    super(message);
    this.name = new.target.name;
  }
}

/** 写真そのものが要求を満たさない。ユーザーに撮り直し・選び直しを促す種類の失敗。 */
export class InputImageError extends ExporterError {}

/**
 * 顔が1つも検出できなかった。
 *
 * フィットの入力が無いので後続の段はすべて成立しない。複数写っている場合は失敗にせず
 * 主役を1人選ぶ（選び方は `domain/faceSubject`）。
 */
export class FaceNotDetectedError extends InputImageError {}

/**
 * 顔の肌が1画素も取れず、基準の肌色を測れなかった。
 *
 * この色は口腔壁の塗りと、写真がどこからも届かないテクセルの下地になる。代わりの色を
 * 発明して通すと**写真と無関係な色が skin_albedo に焼き込まれる**。顔が検出できて
 * フィットも通った写真でここが空になるのはセグメンタ側の破綻なので、写真を選び直す
 * 種類の失敗として扱う。
 */
export class SkinColorUnavailableError extends InputImageError {}

/** 実行環境が要求を満たさない。写真を変えても直らない種類の失敗。 */
export class RuntimeEnvironmentError extends ExporterError {}

/**
 * 要求する GPU 実行環境が使えない。
 *
 * デスクトップ側は CUDA が無ければ起動時に落とす（CPU へ落ちれば動くが実測 150 倍
 * 遅い）。ブラウザでは WebGPU が無ければ WASM + int8 へ落ちる — **こちらは黙って
 * 落とさず、どちらで動いているかを画面に出す**（利用者が実行環境を選べないため）。
 */
export class GpuUnavailableError extends RuntimeEnvironmentError {}

/**
 * 推論モデルかアセットを取得できなかった。
 *
 * ブラウザは配布元 URL から実行時に読む（デスクトップ側の「ファイルが置かれていない」
 * に当たるのが、こちらでは「fetch が失敗した」）。
 */
export class ModelFileNotFoundError extends RuntimeEnvironmentError {}
