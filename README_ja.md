# ComfyUI-MiniMax-H3-Long-Video

[English](README.md) | [日本語](README_ja.md)

現行のComfyUI向けに開発中の、MiniMax H3 Reference to Video長尺生成ノードです。

## ノード

`MiniMax H3 Long Reference Sampler` は、ComfyUI標準の `MiniMax H3 Reference to Video` の参照入力と、カスタムサンプラー用の入力を組み合わせたノードです。24 fpsの長いタイムラインをモデルが処理可能なAV LATENTセグメントに分割し、直前の動画・音声LATENTから22フレームまたは39フレームを次のセグメントの0フレーム目に継続ガイドとして与えます。

各セグメントはSSDへ保存され、すべてのデコード済みフレームをRAMへ保持せず、選択されたチェックポイントから1本のMP4を作成します。継続セグメントの先頭にはガイド部分も生成されますが、この部分は完成した動画から取り除かれます。

プロンプトは `integrated_multimodal_description:` ブロック、または次のような通常のShotタイムラインを使用できます。

```text
[Shot 1] ...
[Shot 2] At 00:05.000, ...
[Shot 3] At 00:09.500, ...
```

> **意図した長尺進行には Shot マーカーが必要です。** すべての Shot に時刻を指定するタイムラインが最も正確です。すべてのShotで時刻が省略されている場合は、`length` と `max_raw_frames` から計算したセグメント数へ Shot を均等配分し、各セグメント内の時刻を自動で割り当てます。時刻あり・なしのShotは混在可能です。明示時刻は固定したまま、無時刻Shotを前後の明示時刻の間、または最後の明示時刻から動画終了までへ等間隔に配置します。Shot マーカー自体がない場合、ノードは動作内容を意味単位で分割できず、同じ全文を各セグメントへ渡します。その結果、各セグメントで同じ動作が最初から始まる可能性があります。

各Shotは、そのグローバル開始時刻を含むセグメントに一度だけ割り当てられます。前のセグメントですでに開始したShotの文章を、次のセグメントへ重複して渡すことはありません。各Shotの時刻はセグメント内のローカル時刻へ変換されます。最初のShotより前にある共通説明と、最後のShotより後にある認識可能なカメラ編集指示は、すべてのセグメントに共通する指示として扱われます。

新しいShotが1つもないセグメントには、以前の動作を繰り返さず、直前のAVコンテキストから続きを生成するための中立的な指示だけが渡されます。

すべてのセグメントで同じ `noise_seed` を使用します。セグメントごとのローカルタイムラインプロンプトと、直前のAV LATENTコンテキストによって生成内容を変化させます。

`max_raw_frames` は、VRAM使用量に影響する1セグメントあたりの総生成フレーム数です。完成動画から取り除く継続ガイド部分も、この値に含まれます。値はMiniMax H3の `17k+5` フレームグリッドに合わせる必要があります。24 fpsでの目安は次のとおりです。

- 73: 約3.0秒
- 90: 3.75秒
- 107: 約4.5秒
- 124: 約5.2秒（デフォルト）

各サンプリングで実際に使用したプロンプトは、LATENTチェックポイントと一緒に `prompts/segment_NNNN.txt` として保存されます。`manifest.json` には、各セグメントの完成動画上のタイムライン範囲とプロンプト範囲が記録されます。

任意入力の `initial_latent` は継続コンテキストとしてのみ使用します。その末尾が、このノードで最初に生成するセグメントの除去可能な先頭部分をガイドします。`initial_latent` 自体のフレームは出力動画に含まれず、このノードのプロンプトタイムラインは再び0秒から始まります。

## チェックポイントと途中からの再生成

`cache_name` は常にComfyUIのoutputフォルダーからの相対パスとして扱われ、指定した場所に関連ファイルをまとめて保存します。末尾の `/` は省略可能です。例えば `h3_long_video/%seed.seed%/` を指定すると、次のように保存されます。

- `output/h3_long_video/<seed>/master.mp4`
- `output/h3_long_video/<seed>/latents/segment_XXXX.safetensors`
- `output/h3_long_video/<seed>/prompts/segment_XXXX.txt`
- `output/h3_long_video/<seed>/manifest.json`

`%date:yyyy-MM-dd%` や `%Node name.widget_name%` 形式のパターンを利用できます。`Node name` は参照先ノードのタイトルまたは型名と一意に一致する必要があります。例えばseedノードのタイトルを `seed` に設定すると、`h3_long_video/%seed.seed%/` の値をそのノードの `seed` 入力から取得できます。Primitiveノードを参照する場合は、`h3_long_video/%noise_seed.value%/` のように、そのノードのタイトルと `value` を指定できます。

`resume` が無効で同名フォルダーがすでに存在する場合は上書きせず、フォルダー名の末尾へ `_2`、`_3` のように番号を追加します。`resume` が有効な場合は、展開後の正確なフォルダーを開き、その中のチェックポイントを再利用します。

互換性のある既存チェックポイントを利用するには `resume` を有効にします。`reroll_from_segment` を `-1` のままにすると、存在しない、または互換性のない最初のセグメントから生成を再開します。`N` を指定すると、セグメント `N` より前を維持し、`N` 以降をすべて再生成します。

後続セグメントは直前のLATENT末尾を引き継ぐため、前半のプロンプトを変更した場合は、その変更を含むセグメント以前から再生成してください。

## 保存済み長尺動画のLATENTアップスケール

`MiniMax H3 Long Latent Upscale & Assemble` は、完成済みのLong H3 bundleを読み込み、[Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)を使ってチェックポイントを1本ずつ処理します。このノードを使用する前に、リンク先のカスタムノードをインストールし、対応するモデルを `ComfyUI/models/latent_upscale_models/` に配置してください。

`source_path` には、ComfyUIのoutputフォルダーからの相対bundleパス、その `manifest.json`、または `master.mp4` を指定できます。ComfyUIのoutputフォルダー内であれば絶対パスも使用できます。SSDから元チェックポイントを1本だけロードし、24チャンネルの映像LATENTだけをアップスケールして音声LATENTを保持したまま、別のoutput bundleへ保存します。保存後に次のセグメントを読み込むため、全セグメントを同時にRAMへ保持しません。

継続ガイドを含む生のセグメント全体をアップスケールしてから、完成MP4の作成時に元manifestの `context_frames` と末尾の余剰フレームを除去します。そのため、アップスケール版masterの完成タイムラインは元masterと一致します。`resume` で互換性のあるアップスケール済みチェックポイントを再利用でき、`reroll_from_segment` で指定したセグメント以降を再処理できます。

希望するピクセル解像度を `target_width` と `target_height` に指定します。LATENTグリッドの `align` はデフォルトの2なら、32ピクセル単位の一般的なLong H3解像度を維持できます。大きな値を指定すると、実際の出力解像度が切り上げられる場合があります。`last_latent` は最後のアップスケール済み生AVセグメントで、`video` と `master_path` は結合済みの結果です。

出力例：

```text
output/h3_long_upscaled/
├── master.mp4
├── manifest.json
└── latents/
    ├── segment_0000.safetensors
    └── segment_0001.safetensors
```

## 現在の制限

- MiniMax H3のAV LATENT専用、バッチサイズ1
- 出力は24 fps固定
- H.264/AACのMP4出力
- 既存のH3 multishot経路と同様のBasicGuiderサンプリング
- `initial_latent` を接続する場合、`width` と `height` をそのLATENTの解像度に合わせる必要があります
- 長尺LATENTアップスケールには、H3 latent upscalerカスタムノードとモデルウェイトの別途インストールが必要です
