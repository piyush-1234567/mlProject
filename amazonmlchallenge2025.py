# ==============================
# ML Challenge 2025: Smart Product Pricing (Optimized + Callbacks)
# ==============================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack, csr_matrix

# -----------------------------
# 1️⃣ Load Data
# -----------------------------
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# -----------------------------
# 2️⃣ Preprocessing Target
# -----------------------------
# Clip extreme prices
train['price'] = train['price'].clip(lower=0.01, upper=train['price'].quantile(0.99))
y_train = np.log1p(train['price'])  # log(1 + price)

# -----------------------------
# 3️⃣ Text Data Preprocessing
# -----------------------------
train_corpus = train['catalog_content'].fillna('')
test_corpus  = test['catalog_content'].fillna('')

# -----------------------------
# 4️⃣ TF-IDF Features
# -----------------------------
tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1,2), stop_words='english')
X_train_tfidf = tfidf.fit_transform(train_corpus)
X_test_tfidf  = tfidf.transform(test_corpus)

# -----------------------------
# 5️⃣ Sentence Embeddings
# -----------------------------
model_st = SentenceTransformer('all-MiniLM-L6-v2')
train_emb = model_st.encode(train_corpus.tolist(), batch_size=64, show_progress_bar=True)
test_emb  = model_st.encode(test_corpus.tolist(), batch_size=64, show_progress_bar=True)

# Sparse matrices
train_emb_sparse = csr_matrix(train_emb)
test_emb_sparse  = csr_matrix(test_emb)

# -----------------------------
# 6️⃣ Simple Text Features
# -----------------------------
def create_text_features(corpus):
    return pd.DataFrame({
        'text_len': corpus.str.len(),
        'word_count': corpus.str.split().str.len(),
        'digit_count': corpus.str.count(r'\d'),
        'upper_count': corpus.str.count(r'[A-Z]'),
        'punct_count': corpus.str.count(r'[^\w\s]')
    })

train_feats = create_text_features(train_corpus)
test_feats  = create_text_features(test_corpus)

train_feats_sparse = csr_matrix(train_feats.values)
test_feats_sparse  = csr_matrix(test_feats.values)

# -----------------------------
# 7️⃣ Combine All Features
# -----------------------------
X_train_full = hstack([X_train_tfidf, train_emb_sparse, train_feats_sparse]).tocsr()
X_test_full  = hstack([X_test_tfidf, test_emb_sparse, test_feats_sparse]).tocsr()

# -----------------------------
# 8️⃣ Remove NaNs in y
# -----------------------------
y_train_np = y_train.values
mask = ~np.isnan(y_train_np)
X_train_clean = X_train_full[mask, :]
y_train_clean = y_train_np[mask]

# -----------------------------
# 9️⃣ Train/Validation Split
# -----------------------------
X_train_part, X_val, y_train_part, y_val = train_test_split(
    X_train_clean, y_train_clean, test_size=0.2, random_state=42
)

# -----------------------------
# 🔟 Train LightGBM with Callbacks
# -----------------------------
model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=64,
    max_depth=12,
    colsample_bytree=0.7,
    subsample=0.8,
    random_state=42
)

model.fit(
    X_train_part, y_train_part,
    eval_set=[(X_val, y_val)],
    eval_metric='l2',
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(period=50)
    ]
)

# -----------------------------
# 1️⃣1️⃣ Validation Predictions & SMAPE
# -----------------------------
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

y_pred_val_log = model.predict(X_val)
y_pred_val = np.expm1(y_pred_val_log)
y_val_true = np.expm1(y_val)
val_smape = smape(y_val_true, y_pred_val)
print(f"📊 Validation SMAPE: {val_smape:.2f}%")

# -----------------------------
# 1️⃣2️⃣ Train Final Model on Full Train
# -----------------------------
model.fit(X_train_clean, y_train_clean)

# -----------------------------
# 1️⃣3️⃣ Predict on Test Set
# -----------------------------
y_pred_log = model.predict(X_test_full)
y_pred = np.expm1(y_pred_log)

submission = pd.DataFrame({
    "sample_id": test["sample_id"],
    "price": y_pred
})
submission.to_csv("test_out.csv", index=False)
print("✅ Submission file created successfully!")
