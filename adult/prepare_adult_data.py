import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from fairlearn.metrics import (
    MetricFrame, selection_rate, false_positive_rate,
    false_negative_rate, demographic_parity_difference,
    equalized_odds_difference
)
from fairlearn.postprocessing import ThresholdOptimizer

st.set_page_config(page_title="Bias Audit App", layout="wide")
st.title("📊 Bias Audit Report: Income Prediction Model")

# --- 1. Load and Clean Data ---
@st.cache_data
def load_data():
    folder_path = r'C:\Users\Capaciti\Downloads\adult'
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
    ]
    train_path = os.path.join(folder_path, 'adult.data')
    test_path = os.path.join(folder_path, 'adult.test')

    df_train = pd.read_csv(train_path, header=None, names=columns, na_values=" ?", skipinitialspace=True)
    df_test = pd.read_csv(test_path, header=0, names=columns, na_values=" ?", skipinitialspace=True, skiprows=1)

    def clean_income(income_series):
        return income_series.str.strip().str.replace('.', '', regex=False)

    df_train['income'] = clean_income(df_train['income'])
    df_test['income'] = clean_income(df_test['income'])

    df_train.dropna(inplace=True)
    df_test.dropna(inplace=True)

    df_all = pd.concat([df_train, df_test], ignore_index=True)
    return df_train, df_test, df_all

df_train, df_test, df_all = load_data()
st.success(f"✅ Training data loaded: {df_train.shape}")
st.success(f"✅ Test data loaded: {df_test.shape}")

# --- 2. Visualize Distribution ---
st.subheader("📊 Income Distribution by Gender and Race")
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
sns.countplot(data=df_all, x='income', hue='sex', ax=axs[0], palette='pastel')
axs[0].set_title('Income by Gender')
axs[0].tick_params(axis='x', rotation=45)

sns.countplot(data=df_all, x='income', hue='race', ax=axs[1], palette='muted')
axs[1].set_title('Income by Race')
axs[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
st.pyplot(fig)

# --- 3. Prepare Data ---
features = ["age", "workclass", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "hours_per_week"]
target = "income"
df_model = df_all[features + [target]].copy()

label_encoders = {}
for col in df_model.select_dtypes(include=['object']).columns:
    if col != target:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        label_encoders[col] = le

df_model[target] = df_model[target].map({'<=50K': 0, '>50K': 1})

X = df_model[features]
y = df_model[target]
train_size = len(df_train)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# --- 4. Train Model ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.subheader("✅ Baseline Model Performance")
st.write(f"Accuracy: **{accuracy_score(y_test, y_pred):.4f}**")
report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
st.dataframe(report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}))

# --- 5. Fairness Metrics ---
sensitive_feature = X_test['sex']
metric_frame = MetricFrame(metrics={
    'accuracy': accuracy_score,
    'selection_rate': selection_rate
}, y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_feature)

sex_le = label_encoders['sex']
df_gender_metrics = metric_frame.by_group.copy()
df_gender_metrics.index = sex_le.inverse_transform(df_gender_metrics.index.astype(int))

st.subheader("📉 Fairness Metrics by Gender (Baseline)")
st.dataframe(df_gender_metrics)

# --- 6. Detailed Metrics ---
metric_frame_full = MetricFrame(metrics={
    'accuracy': accuracy_score,
    'selection_rate': selection_rate,
    'false_positive_rate': false_positive_rate,
    'false_negative_rate': false_negative_rate
}, y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_feature)

df_gender_metrics_full = metric_frame_full.by_group.copy()
df_gender_metrics_full.index = sex_le.inverse_transform(df_gender_metrics_full.index.astype(int))

st.subheader("📋 Detailed Fairness Metrics by Gender")
st.dataframe(df_gender_metrics_full)

dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_feature)
eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_feature)
st.write(f"📊 Demographic Parity Difference: **{dp_diff:.4f}**")
st.write(f"📊 Equalized Odds Difference: **{eo_diff:.4f}**")

# --- 7. Bias Mitigation ---
postprocessed = ThresholdOptimizer(
    estimator=model,
    constraints="equalized_odds",
    predict_method='predict_proba',
    prefit=True
)
postprocessed.fit(X_train, y_train, sensitive_features=X_train['sex'])
y_pred_post = postprocessed.predict(X_test, sensitive_features=X_test['sex'])

metric_frame_post = MetricFrame(metrics={
    'accuracy': accuracy_score,
    'selection_rate': selection_rate
}, y_true=y_test, y_pred=y_pred_post, sensitive_features=sensitive_feature)

df_gender_metrics_post = metric_frame_post.by_group.copy()
df_gender_metrics_post.index = sex_le.inverse_transform(df_gender_metrics_post.index.astype(int))

st.subheader("🔧 Post-Mitigation Fairness Metrics")
st.dataframe(df_gender_metrics_post)

# --- 8. Accuracy Comparison Chart ---
st.subheader("📊 Accuracy by Gender Before & After Mitigation")
labels = sex_le.inverse_transform([0, 1])
acc_before = metric_frame.by_group['accuracy'].values
acc_after = metric_frame_post.by_group['accuracy'].values

x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width / 2, acc_before, width, label='Before Mitigation', color='skyblue')
rects2 = ax.bar(x + width / 2, acc_after, width, label='After Mitigation', color='orange')

ax.set_ylabel('Accuracy')
ax.set_title('Accuracy by Gender Before & After Mitigation')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

if hasattr(ax, 'bar_label'):
    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')

plt.tight_layout()
st.pyplot(fig)

# --- 9. Fairness by Other Attributes ---
st.subheader("📋 Fairness by Other Sensitive Attributes")
for attr in ['sex', 'race', 'marital_status']:
    st.write(f"#### Fairness Metrics by `{attr}`")
    sensitive = X_test[attr]
    mframe = MetricFrame(metrics={
        'accuracy': accuracy_score,
        'selection_rate': selection_rate,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate
    }, y_true=y_test, y_pred=y_pred, sensitive_features=sensitive)

    if attr in label_encoders:
        le = label_encoders[attr]
        df_attr = mframe.by_group.copy()
        df_attr.index = le.inverse_transform(df_attr.index.astype(int))
    else:
        df_attr = mframe.by_group

    st.dataframe(df_attr)
