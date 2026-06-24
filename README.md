---
title: 譯彩紛呈：重譯文本分析系統 TransPrism II
emoji: "\U0001F308"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
pinned: false
license: mit
---

# 譯彩紛呈：重譯文本分析系統

**TransPrism: An Analytical Framework for Retranslation II｜分析引擎 Spectra Engine**

針對重譯文本（retranslation）語言特徵的自動化量化分析平台，採雙 Agent 架構。

- **主 Agent（LLM1）**：研究情境引導 & 研究報告撰寫
- **工作程序 Agent（LLM2）**：上傳引導 & 格式驗證

系統對上傳的中文譯本計算 **75 項語言指標**（基礎統計、詞彙難易度、詞彙多樣性、語意複雜度、語法複雜度、篇章連貫、詞性標記），並進行敘述統計、差異性檢定、機器學習分群與分類，最後由 LLM 產出研究報告。

## 四種研究情境

1. 人工翻譯 vs 機器翻譯
2. 全譯 vs 編譯／節譯／改寫
3. 同期不同譯者譯本比較
4. 不同時期重譯本的歷時比較

## 操作流程

情境選擇 → 上傳引導 → 文本上傳 → 格式驗證 → 語料 Metadata 說明 → 指標計算 → 統計分析 → 報告撰寫 → 下載成果

> 文本須先經由 [ckip_segmentation_sumin](https://github.com/suhsiung/ckip_segmentation_sumin) 完成斷詞，檔名以 `A_xx.txt`、`B_xx.txt`（…）區分群組。

## 環境變數 (Secrets)

| 名稱 | 說明 |
|------|------|
| `LLM1_API_KEY` | 主 Agent OpenRouter API Key |
| `LLM2_API_KEY` | 工作程序 Agent OpenRouter API Key |
