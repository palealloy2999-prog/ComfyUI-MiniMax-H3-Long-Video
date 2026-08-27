# ComfyUI-MiniMax-H3-Long-Video

[English](README.md) | [日本語](README_ja.md)

現行のComfyUI向けに開発中の、MiniMax H3 Reference to Video長尺生成ノードです。

## ノード

`MiniMax H3 Long Reference Sampler` は、ComfyUI標準の `MiniMax H3 Reference to Video` の参照入力と、カスタムサンプラー用の入力を組み合わせたノードです。24 fpsの長いタイムラインをモデルが処理可能なAV LATENTセグメントに分割し、直前の動画・音声LATENTから22フレームまたは39フレームを次のセグメントの0フレーム目に継続ガイドとして与えます。

各セグメントはSSDへ保存され、すべてのデコード済みフレームをRAMへ保持せず、選択されたチェックポイントから1本のMP4を作成します。継続セグメントの先頭にはガイド部分も生成されますが、この部分は完成した動画から取り除かれます。

プロンプトは公式Base形式の `integrated_multimodal_description:`、公式Ref2VA形式の `detailed_description:`、または次のような通常のShotタイムラインを使用できます。

```text
[Shot 1] ...
[Shot 2] At 00:05.000, ...
[Shot 3] At 00:09.500, ...
```

> **意図した長尺進行には Shot マーカーが必要です。** すべての Shot に時刻を指定するタイムラインが最も正確です。すべてのShotで時刻が省略されている場合は、`length` と `max_raw_frames` から計算したセグメント数へ Shot を均等配分し、各セグメント内の時刻を自動で割り当てます。時刻あり・なしのShotは混在可能です。明示時刻は固定したまま、無時刻Shotを前後の明示時刻の間、または最後の明示時刻から動画終了までへ等間隔に配置します。Shot マーカー自体がない場合、ノードは動作内容を意味単位で分割できず、同じ全文を各セグメントへ渡します。その結果、各セグメントで同じ動作が最初から始まる可能性があります。

各Shotは、そのグローバル開始時刻を含むセグメントで開始します。セグメント境界でまだ進行中のShotは、次のセグメントへ「冒頭から再演せず現在状態から続ける」指示付きで渡されます。segment 0の時刻は変更しません。segment 1以降は、そのセグメントのマスター開始秒を単純に引いてローカル時刻へ変換します。継続AVガイドはローカル時刻の外側にある先行コンテキストとして扱います。

すべてのセグメントへ適用する指示は、最後のShotより後に、単独行の `[Global Instructions]` を置いて記述してください。これは長尺マスター専用の内部区切りであり、配下の文章は全セグメントへ渡されますが、マーカー自体はH3へ送る前に除去されます。`Character-consistency requirement:` などのラベル自体には特別な意味はありません。マーカーがなければ、最後のShotより後の文章もそのShotの本文として扱われます。マーカーは1回だけ使用し、Shotの動作本文内には置かないでください。

```text
[Shot 1] 走り始める。
[Shot 2] At 00:05.000, 障害物を飛び越える。

[Global Instructions]
すべてのセグメントで同じキャラクター、衣装、連続した音楽を維持する。
```

新しいShotが1つもないセグメントには、直前のAVコンテキストと、その境界で進行中のShot本文を、冒頭から再演しない継続指示とともに渡します。

すべてのセグメントで同じ `noise_seed` を使用します。セグメントごとのローカルタイムラインプロンプトと、直前のAV LATENTコンテキストによって生成内容を変化させます。

`max_raw_frames` は、意図したセグメント秒数 `a` から `n=max(5, round(a*24)); n+(5-n%17)%17` で作られたH3グリッド値です。ノードは一般的な整数秒へ逆算し、73を3秒、107を4秒、124を5秒、362を15秒のマスター時間窓として扱います。同じ逆算を `length` にも適用するため、720とそのH3グリッド形式736はいずれも30秒です。362との組み合わせでは、0–15秒と15–30秒の2区間だけになります。継続ガイドとH3用paddingは内部LATENTにだけ追加され、プロンプト窓を短縮・移動しません。

- 73: 約3.0秒
- 90: 3.75秒
- 107: 約4.5秒
- 124: 約5.2秒（デフォルト）

各サンプリングで実際に使用したプロンプトは、LATENTチェックポイントと一緒に `prompts/segment_NNNN.txt` として保存されます。`manifest.json` には、各セグメントの完成動画上のタイムライン範囲とプロンプト範囲が記録されます。

サンプリング前に分割結果を確認・修正する場合は、`MiniMax H3 Long Prompt Planner` を追加し、その `prompt_plan` 出力を `MiniMax H3 Long Reference Sampler.prompt_plan` へ接続します。両ノードの `length`、`max_raw_frames`、`context_frames` は同じ値にし、Samplerへ `initial_latent` を接続する場合だけPlannerの `has_initial_latent` も有効にしてください。Plannerの `preview` は、各segmentの正確なローカルプロンプトを順番に格納したSTRINGリストです。完全に置き換えたい区間だけ `segment_prompt_N` を追加して修正できます。`prompt_plan` 接続時はその内容がSampler内部の分割より優先されます。既存ワークフロー互換のためSamplerの `prompt` 入力自体は残しています。

公式 [`h3-prompt-writing`](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing) を長尺マスター向けに補完するAgent Skillを [`skills/minimax-h3-long-video-prompt-writing`](skills/minimax-h3-long-video-prompt-writing) に同梱しています。公式のフィールド順、Ref2VAの `<Subject N>` 参照ラベル、`(Sx)` 話者IDを維持しながら、セグメント境界、`[Global Instructions]`、時間付きイベントの配置を定義します。時間に依存する映像・台詞・歌詞・効果音・音楽変化は必ず対応するShot内へ置き、全セグメントへ複製される `overall_soundscape` と `non_diegetic_music` にはマスター絶対時刻を書かないでください。

継続セグメントでは、全尺用のRef2VA `summary` を再利用せず、その区間の範囲と「再開しない」ことだけを示すローカル要約へ置き換えます。時刻を含まない音響・音楽条件は継続指示とともに保持しますが、共有音響欄に絶対時刻がある場合は、各区間でイントロ、ドロップ、フィナーレを再演しない継続指示へ置き換えます。プロンプト全体を囲むMarkdownの `text` コードフェンスはエンコード前に自動除去します。曖昧な `S1 = ...` 形式は生成前にエラーにし、映像参照には `<Subject N>`、話者だけに `(Sx)` を要求します。

```bash
npx skills add . --skill minimax-h3-long-video-prompt-writing
```

任意入力の `initial_latent` は継続コンテキストとしてのみ使用します。その末尾が、このノードで最初に生成するセグメントの除去可能な先頭部分をガイドします。`initial_latent` 自体のフレームは出力動画に含まれず、このノードのプロンプトタイムラインは再び0秒から始まります。

## チェックポイントと途中からの再生成

`cache_name` は常にComfyUIのoutputフォルダーからの相対パスとして扱われ、指定した場所に関連ファイルをまとめて保存します。末尾の `/` は省略可能です。例えば `h3_long_video/%seed.seed%/` を指定すると、次のように保存されます。

- `output/h3_long_video/<seed>/master.mp4`
- `output/h3_long_video/<seed>/latents/segment_XXXX.safetensors`
- `output/h3_long_video/<seed>/prompts/segment_XXXX.txt`
- `output/h3_long_video/<seed>/manifest.json`

`%date:yyyy-MM-dd%` や `%Node name.widget_name%` 形式のパターンを利用できます。`Node name` は参照先ノードのタイトルまたは型名と一意に一致する必要があります。例えばseedノードのタイトルを `seed` に設定すると、`h3_long_video/%seed.seed%/` の値をそのノードの `seed` 入力から取得できます。Primitiveノードを参照する場合は、`h3_long_video/%noise_seed.value%/` のように、そのノードのタイトルと `value` を指定できます。

`resume` が無効で同名フォルダーがすでに存在する場合は上書きせず、フォルダー名の末尾へ `_2`、`_3` のように番号を追加します。`resume` が有効な場合は、展開後の正確なフォルダーを開き、その中のチェックポイントを再利用します。

互換性のある既存チェックポイントを利用するには `resume` を有効にします。`reroll_from_segment` を `-1` のままにすると、存在しない、または互換性のない最初のセグメントから生成を再開します。`N` を指定すると、セグメント `N` より前を維持し、`N` 以降をすべて再生成します。互換性判定にはローカルプロンプト、seed、フレーム構成、model・CLIP・sampler・sigmasの上流設定、参照入力、`initial_latent`、および直前セグメントからの生成系統を使用します。古いmanifest schemaのチェックポイントは再利用されません。

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

### MMH3 Ultimate Upscaleで再サンプリングする

拡散モデルでLATENTを拡大・再サンプリングする場合は、追加した4つのループ補助ノードをEasyUseの `For Loop Start` / `For Loop End` および `MMH3 Ultimate Upscale` と組み合わせます。編集用サンプルは [`sample/minimax_h3_r2v-longtime_upscale.json`](sample/minimax_h3_r2v-longtime_upscale.json) です。

`MiniMax H3 Long Reference Sampler` の `master_path` は、`MiniMax H3 Long Upscale Prepare` の `master_path` に直接接続できます。手入力の場合はbundleフォルダーまたはその `manifest.json` も指定できます。

最終bundleの保存先は、元bundle直下の `upscale/` を基準にします。例えば `h3_long_video/123/master.mp4` からは `h3_long_video/123/upscale/master.mp4` を作ります。すでに同名フォルダーがある場合は上書きせず、`upscale_2/`、`upscale_3/` のように増やします。Prepareが最初にこの永続bundleを確保し、Segment Saveは処理済みチェックポイントとプロンプトをそこへ直接保存します。途中で処理に失敗しても、完了済みセグメントと `processing` 状態のmanifestはoutputフォルダー内に残ります。Assembleはファイル移動を行わず、保存済みセグメントを検証してからMP4を作成します。

中断したアップスケールを続行する場合は、同じUltimate Upscale設定を維持したまま、Prepareのadvanced入力 `resume` を有効にします。そのsourceで最新の未完了bundleを開き、元チェックポイントと保存済み出力を検証したうえで、未完了セグメントだけをループへ渡します。全セグメント保存後のMP4デコードだけが失敗していた場合は、Assembleへ新しいprogressを渡して再デコードできるよう、最終セグメントのみ再処理します。

配線は `Prepare -> For Loop Start -> Segment Load -> MiniMax H3 Reference to Video -> MMH3 Ultimate Upscale -> Segment Save -> For Loop End -> Assemble` です。`segment_count` をループ回数へ、EasyUseの `index` を `segment_index` へ接続します。Segment Loadは、元LATENTに加えて、そのセグメントのローカルプロンプト、seed、元のwidth・height、生フレーム数を出力します。Segment Saveは処理済みLATENTをその場でSSDへ保存し、Loop Endには小さな進捗情報だけを渡します。全ループ完了後にAssembleが各チェックポイントを1本ずつデコードし、記録済みの継続用重複部分を除去して1本のMP4にします。

元の生成で使った参照画像・参照動画・音声は、`MiniMax H3 Reference to Video` にもう一度接続してください。`prompt` と `length` はSegment Loadから接続し、widthとheightはMMH3 Ultimate Upscale側の目標解像度と一致させます。このReference to Videoノードの空LATENT出力は使わず、MMH3 Ultimate UpscaleのLATENT入力にはSegment Loadの元LATENTを接続します。

このワークフローには、別途 [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use)、[Comfyui-MMH3-UltimateUpscale](https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale)、および必要なモデルウェイトが必要です。実行前にサンプル内のモデル名と参照画像を環境に合わせて変更してください。

## 現在の制限

- MiniMax H3のAV LATENT専用、バッチサイズ1
- 出力は24 fps固定
- H.264/AACのMP4出力
- `initial_latent` を接続する場合、`width` と `height` をそのLATENTの解像度に合わせる必要があります
- 長尺LATENTアップスケールには、H3 latent upscalerカスタムノードとモデルウェイトの別途インストールが必要です。**※実験段階。未テストです。**

## ライセンス

[GNU General Public License v3.0](LICENSE)

本プロジェクトには、GPL-3.0で提供されるComfyUI標準のMiniMax H3実装から、2026年に移植・変更した部分が含まれます。
