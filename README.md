# 台灣實質薪資停滯的分解

本儲存庫提供一套可重現的描述性會計分析，使用行政院主計總處官方資料，將 2000–2024 年工業及服務業受僱員工的實質平均薪資變動分為：

1. 產業內薪資變動（within）；
2. 受僱人數占比變動（shift）；
3. 占比與薪資同時變動的交乘項（interaction）。

研究主規格採年資料、行業大類、實質經常性月薪與 Laspeyres 分解；Paasche、Törnqvist、2008–2024 與 2016–2024 期間、時薪、總薪資、中類粒度及固定占比反事實均作為敏感度分析。研究設計先行登錄於 [`PREANALYSIS.md`](PREANALYSIS.md)，最初登錄提交為 `71f947c`。

## 主要結果

- 2000–2024 年共同樣本的實質經常性月薪增加 4.61%，實質經常性時薪增加 15.56%；相差 10.95 個對數百分點，顯示工時變動對月薪停滯的量測很重要。
- Laspeyres 主分解中，實質經常性月薪增加新臺幣 2,107 元：產業內分量為 +2,525 元、結構移轉為 −12 元、交乘項為 −407 元。
- 固定 2000 年受僱人數占比時，2024 年反事實薪資比實際值高 418 元（0.89%）。
- 中類分解會顯著改變分量配置，而且 shift 的方向並未跨第 10 次與第 11 次行業分類期間維持一致。因此，資料不支持「跨產業結構移轉是長期實質薪資停滯主因」的穩健敘述。
- 循環移動區塊 bootstrap 對 shift 分量的雙尾檢定為 `p = 0.958`。這是描述性不確定性衡量，不是因果推論。

完整論述、方法、限制與附錄見 [`paper/wage_stagnation_decomposition_tw.pdf`](paper/wage_stagnation_decomposition_tw.pdf)。

## 資料與可稽核性

所有研究資料均來自行政院主計總處：

- 薪資及生產力統計年報與開放資料：2000–2024 年行業大類，及 2016–2024 年行業中類；
- 消費者物價指數：年平均 CPI，統一換算為 2024 年價格；
- 行業統計分類第 7 至第 11 次修訂及官方對照表。

共 424 個來源檔案的原始網址、取得日、期間、分類版本、檔案大小與 SHA-256 均記錄於 [`data/source_manifest.csv`](data/source_manifest.csv)。排除規則逐筆記錄於 [`data/exclusion_log.csv`](data/exclusion_log.csv)。原始檔案保留在 `data/raw/`，重現不依賴即時網路下載。

外部一致性閘門同時比較最新大類、同版大類與中類加總，共 15 項檢查；最大相對誤差為 0.052%，低於預先設定的 0.5%。詳細結果見 [`results/tables/table_02_external_validation.csv`](results/tables/table_02_external_validation.csv)。

2000–2008 年教育業在官方大類表中標示為無資料，未進行插補；涉及 2000 或 2008 端點的長期分解使用兩端皆有觀測的 16 個共同大類。2016 年起的分析使用 17 個大類。這項涵蓋差異不可解讀為經濟結構移轉。

## 一鍵重現

需要 Python 3.11 以上，以及 XeLaTeX 或 [Tectonic](https://tectonic-typesetting.github.io/)；`pikepdf` 用於修復 PDF 的 Unicode 對映。建議在虛擬環境中執行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-repro.txt
$env:PYTHONPATH = "src"
# 只有 Tectonic 不在 PATH 時才需要下一行
$env:TECTONIC = "C:\path\to\tectonic.exe"
python scripts/reproduce_all.py
python scripts/verify_external.py
python scripts/verify_paper_numbers.py
python -m pytest
```

`scripts/reproduce_all.py` 是唯一的完整重現入口：它會重新解析官方原始檔、重建分析面板、執行一致性檢查、輸出表圖、編譯論文，並產生最終 PDF。若任何必要來源、欄位、對照或外部一致性閘門不符合預分析規則，程式會以錯誤結束。

固定版本套件列於 [`requirements.txt`](requirements.txt)，完整安裝入口為 [`requirements-repro.txt`](requirements-repro.txt)。

## 儲存庫結構

```text
config/        研究規格與行業對照
data/raw/      主計總處官方原始檔
data/interim/  程式建立的標準化分析面板
paper/         LaTeX 原稿、樣式、圖與最終 PDF
results/       可稽核表格、圖與 SHA-256 清單
scripts/       完整重現及驗證入口
src/           資料整理、分解、反事實與推論程式
tests/         單元測試
```

## 研究邊界

本研究是 shift-share 會計恆等式，不識別產業結構對薪資的因果效果。平均薪資同時反映勞工組成、工時、獎金與景氣；行業分類修訂、共同樣本限制以及短期中類期間都會限制外推。月資料與總薪資僅作敏感度分析，不作主結論。

## 引用與授權

建議引用資訊見 [`CITATION.cff`](CITATION.cff)。程式與研究文字採 MIT License；官方資料的權利與使用條款仍依行政院主計總處規定。AI 協作範圍揭露於 [`AI_USE.md`](AI_USE.md)。
