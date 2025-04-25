# Twitter Sentiment Classification

## Evironment
- python: 3.13.2
- pandas ~= 2.2.3
- nltk ~= 3.9.1
- scikit-learn ~= 1.6.1
- matplotlib ~= 3.10.1
- seaborn ~= 0.13.2

## Data Preprocessing

### 透過 pandas 讀取csv
因為在此資料集中，有推文可能本身包含逗號（`,`），會被CSV誤判為欄位分隔。所以需要在讀取的時候，需要加入一些參數：
```python
df = pd.read_csv(
    "dataset.csv",
    encoding="latin1",         # 避免亂碼（原始檔可能不是 utf-8）
    quoting=1,                 # 引號處理：1 代表 QUOTE_ALL
    quotechar='"',             # 指定用 " 包住的欄位視為一個欄位
    error_bad_lines=False,     # 忽略格式錯誤的行（舊版 pandas）
    on_bad_lines='skip'        # pandas >= 1.3.0 使用這個取代上面那個
)
```

### Sentiment Text 處理
| 處理項目                | 原因 or 內容                                |
|:----------------------- |:------------------------------------------- |
| 小寫化                  | 避免大小寫視為不同詞                        |
| 移除網址                | 移除 `http://` 或 `https://`                |
| 移除@username與hashtags | 通常不影響情感，屬於雜訊                    |
| 移除標點符號與特殊字元  | 移除如`.,!?@#%&*()`等符號，只保留文字與空白 |
| 移除停用字（stopwords） | 無實質意義                                  |
| 詞幹還原（Stemming）    | 使用`nltk`的`PorterStemmer`                 |
| 空白與長空格            | 清除多餘空白、統一空格                      |


## 資料切割
不直接使用`DataFrame`的`head()`或`iloc[:70%]`去切割資料，因為有可能會導致訓練集或測試集中類別分佈不均衡，稱作 **class imbalance** 或 **sampling bias**。

### 問題在哪？
推文資料原本可能是先正面排好、再負面排好（尤其是這份資料是早期整理的），如果用`.head()`前70%，有可能：
- 訓練集幾乎都是正面
- 測試集幾乎都是負面

這樣分類器學到的不是「情感模式」，而是「排好順序的假象」。

### 解法：使用 stratified shuffle split（分層隨機切分）
💡概念：
- 在切割資料時，**維持各類別（ex:正面/負面）在訓練集與測試集中的比例與原始資料向近**
- 例如原始資料是 60%正面、40%負面，那訓練與測試集都會維持這個比例（基本上接近）

🛠 工具：
- Python 中用 `scikit-learn` 的 `train_test_split()` 加上 `stratify=` 參數即可辦到


## Training Model

### Baseline - CountVectorizer
🔧 功能：
- 將文字資料轉換成「每個詞出現幾次」的矩陣（詞袋模型）
- 例如：`["i love cats", "i hate dogs"]` ➜ 詞彙：{i, love, cats, hate, dogs} ➜ 對應向量表示

🔄 用法：
1. 初始化 `CountVectorizer`
2. `fit_transform()` 對訓練集做轉換
3. `transform()` 對測試集做轉換（不能再fit，避免 data leakage）

📌 適合：
- 快速建立 baseline
- 模型簡單、直觀
- 小資料效果不錯，但對常出現但不重要的字沒有壓制效果

### Baseline - TfidfVectorizer
🔧 功能：
- 和 `CountVectorizer` 類似，但它加入了「詞的資訊價值」
- 使用「詞頻-反文檔頻率(TF-IDF)」來加權
    - 常出現在整體語料的字（如 "the"、"is"）會被降低權重
    - 特定文中才出現的字（如 "awesome"、"terrible"）會被強化

🔄 用法：
- 和 `CountVectorizer` 幾乎一樣，只是換成 `TfidfVectorizer`

📌 適合：
- 想更精細地表示文字
- 當資料量大或詞彙很多時，TF-IDF 通常能給出更好的分類結果

### Classifier - MultinomialNB
🔧 功能：
- 適用於文字分類中詞頻向量的 Naïve Bayes 模型
- 根據特徵（詞）出現次數的機率分佈來預測類別
- 搭配上面兩個 vectorizer 使用效果最佳

🔄 用法：
1. 建立 `MultinomialNB()` 實例
2. 用 `.fit(x_train, y_train)` 訓練
3. 用 `predict(x_test)` 預測
4. 用 `classification_report()` 計算 precision / recall /F1

📌 適合：
- 高維度稀疏特徵（如詞袋模型）
- 建立 baseline 非常快速

