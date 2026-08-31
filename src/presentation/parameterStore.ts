// 選んだ書き出しパラメータの永続化。
//
// デスクトップ側は OS 標準のアプリ設定領域（`QSettings`）へ JSON で書く。ブラウザの同じ役割は
// `localStorage`。**保存する形（キー・JSON・壊れた値の扱い）はあちらと同じ**にしてある。
//
// 古い版や手編集で壊れた値は使わず、`application` 側の既定値へ戻す（`load` が null を返す）。
// 既定を presentation に持たないためで、**ここは「読めたか」しか判断しない**。

import { ExportSettings, validateExportSettings } from '../application/settings';

export const PARAMETERS_KEY = 'export_parameters/v1';

/** presentation が依存する、パラメータ保存先の最小契約。 */
export interface ParameterStore {
  load(): ExportSettings | null;
  save(parameters: ExportSettings): void;
}

/** `localStorage` を使うパラメータ保存先。 */
export class LocalStorageParameterStore implements ParameterStore {
  constructor(private readonly storage: Storage = localStorage) {}

  load(): ExportSettings | null {
    let raw: string | null;
    try {
      raw = this.storage.getItem(PARAMETERS_KEY);
    } catch {
      // プライベートウィンドウやサイトデータ拒否では読み書きそのものが例外になる。
      return null;
    }
    if (raw === null) return null;
    try {
      const values = JSON.parse(raw) as unknown;
      if (typeof values !== 'object' || values === null || Array.isArray(values)) return null;
      // 検査を通らない値は使わない（範囲外・選べない一辺・lift ≥ rolloff）。
      return validateExportSettings(values as ExportSettings);
    } catch {
      return null;
    }
  }

  save(parameters: ExportSettings): void {
    // 書く前に検査する。**壊れた値を保存しない**（次回起動で既定へ戻るだけになり、保存した
    // つもりの値が黙って消える）。
    validateExportSettings(parameters);
    try {
      this.storage.setItem(PARAMETERS_KEY, JSON.stringify(parameters, Object.keys(parameters).sort()));
    } catch (error) {
      throw new Error(`パラメーターを書き込めませんでした: ${String(error)}`);
    }
  }
}
