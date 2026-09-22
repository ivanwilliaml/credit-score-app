import re
import sys
from pathlib import Path
from collections import Counter
from typing import Tuple
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline as SkPipeline
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

try:                                  
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SEED = 42

# Kunci tracking store ke folder mlruns/ klasik (MLflow 3.x defaultnya pindah ke sqlite mlflow.db
# kalau tidak di-set eksplisit) -> supaya `mlflow ui` tanpa flag apapun selalu ketemu datanya
mlflow.set_tracking_uri("file:./mlruns")

# ----------------- Konstanta preprocessing (mengikuti notebook final v1) -----------------
ORDINAL_MAPS = {
    "Month": {m: i for i, m in enumerate(
        ["January", "February", "March", "April", "May", "June", "July", "August"])},
    "Credit_Mix": {"Bad": 0, "Standard": 1, "Good": 2, "Unknown": 1},
    "Spending_Level": {"Low": 0, "High": 1, "Unknown": -1},
    "Payment_Size": {"Small": 0, "Medium": 1, "Large": 2, "Unknown": -1},
    "Payment_of_Min_Amount": {"No": 0, "Yes": 1, "Unknown": 0},
}
IQR_COLS = ["Num_Bank_Accounts", "Num_Credit_Card", "Num_of_Loan", "Num_of_Delayed_Payment",
            "Num_Credit_Inquiries", "Delay_from_due_date", "Interest_Rate"]
P99_COLS = ["Annual_Income", "Monthly_Inhand_Salary", "Outstanding_Debt",
            "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance"]
ENGINEERED = ["Debt_to_Income", "EMI_to_Salary", "Investment_Rate", "Balance_to_Salary",
              "Credit_History_Years", "Loan_per_Account", "Delay_per_Loan", "Debt_per_Card",
              "Total_Credit_Products", "Delay_to_History"]
HIGH_CARD = ["Occupation", "Primary_Loan"]


# Strategi oversampling SMOTE-Tomek: hanya kelas minoritas dinaikkan setara kelas terbesar kedua (bukan full balance)
def smt_strategy(y):
    c = Counter(y)
    ordered = sorted(c, key=lambda k: c[k], reverse=True)   # [majority, middle, minority]
    minority = ordered[-1]
    return {minority: max(c[ordered[1]], c[minority])}


# 1. Preprocessing : custom sklearn transformer (stateful, fit di train, transform test)
class CreditScorePreprocessor(BaseEstimator, TransformerMixin):
    """Membungkus seluruh preprocessing notebook (clean → impute → cap → decompose →
    feature engineering → ordinal + one-hot encoding) menjadi satu transformer deployable."""

    # Bersihkan tipe data & noise mentah (simbol pada angka, umur invalid, kategori placeholder, dll)
    def _core_clean(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data = data.drop(columns=["Unnamed: 0", "ID", "Customer_ID", "Name", "SSN"], errors="ignore")
        for col in ["Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
                    "Changed_Credit_Limit", "Outstanding_Debt", "Amount_invested_monthly",
                    "Monthly_Balance"]:
            data[col] = pd.to_numeric(
                data[col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True).replace("", np.nan),
                errors="coerce")
        data["Age"] = data["Age"].where(data["Age"].between(18, 100), np.nan)
        data["Monthly_Balance"] = data["Monthly_Balance"].where(data["Monthly_Balance"] >= 0, np.nan)
        for col in ["Num_of_Loan", "Num_Bank_Accounts", "Num_of_Delayed_Payment",
                    "Delay_from_due_date", "Num_Credit_Card", "Num_Credit_Inquiries"]:
            data[col] = pd.to_numeric(data[col], errors="coerce").clip(lower=0)

        def to_months(x):
            if pd.isna(x):
                return np.nan
            if isinstance(x, (int, float)):
                return float(x)
            m = re.match(r"(\d+)\s*Years?\s*and\s*(\d+)\s*Months?", str(x))
            return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan
        data["Credit_History_Age"] = data["Credit_History_Age"].apply(to_months)

        data["Occupation"] = data["Occupation"].replace("_______", "Unknown")
        data["Credit_Mix"] = data["Credit_Mix"].replace("_", "Unknown")
        data["Payment_of_Min_Amount"] = data["Payment_of_Min_Amount"].replace("NM", "Unknown")
        data["Payment_Behaviour"] = data["Payment_Behaviour"].replace("!@9#%8", np.nan)
        data["Type_of_Loan"] = data["Type_of_Loan"].fillna("Not Specified")
        return data

    # Pecah kolom majemuk (Payment_Behaviour, Type_of_Loan) jadi beberapa fitur turunan yang lebih informatif
    def _decompose(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data["Spending_Level"] = data["Payment_Behaviour"].str.extract(r"(Low|High)_spent")
        data["Payment_Size"] = data["Payment_Behaviour"].str.extract(r"(Small|Medium|Large)_value")
        data[["Spending_Level", "Payment_Size"]] = data[["Spending_Level", "Payment_Size"]].fillna("Unknown")
        data = data.drop(columns=["Payment_Behaviour"])
        loans = data["Type_of_Loan"].str.replace(" and ", ",", regex=False)
        data["Num_Loan_Types"] = loans.apply(
            lambda s: 0 if str(s).strip() == "Not Specified" else len([t for t in str(s).split(",") if t.strip()]))
        data["Primary_Loan"] = loans.str.split(",").str[0].str.strip()
        data = data.drop(columns=["Type_of_Loan"])
        return data

    # Feature engineering berbasis domain (rasio utang, cicilan, investasi, dll terhadap income/saldo)
    def _add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        monthly_income = data["Annual_Income"] / 12 + 1
        data["Debt_to_Income"] = data["Outstanding_Debt"] / monthly_income
        data["EMI_to_Salary"] = data["Total_EMI_per_month"] / (data["Monthly_Inhand_Salary"] + 1)
        data["Investment_Rate"] = data["Amount_invested_monthly"] / (data["Monthly_Inhand_Salary"] + 1)
        data["Balance_to_Salary"] = data["Monthly_Balance"] / (data["Monthly_Inhand_Salary"] + 1)
        data["Credit_History_Years"] = data["Credit_History_Age"] / 12
        data["Loan_per_Account"] = data["Num_of_Loan"] / (data["Num_Bank_Accounts"] + 1)
        data["Delay_per_Loan"] = data["Num_of_Delayed_Payment"] / (data["Num_of_Loan"] + 1)
        data["Debt_per_Card"] = data["Outstanding_Debt"] / (data["Num_Credit_Card"] + 1)
        data["Total_Credit_Products"] = data["Num_Bank_Accounts"] + data["Num_Credit_Card"] + data["Num_of_Loan"]
        data["Delay_to_History"] = data["Num_of_Delayed_Payment"] / (data["Credit_History_Age"] / 12 + 1)
        return data

    # Encode kolom ordinal (Month, Credit_Mix, dst) jadi angka sesuai urutan tingkatannya
    def _apply_ordinal(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        for col, mp in ORDINAL_MAPS.items():
            data[col] = data[col].map(mp).fillna(-1).astype(float)
        return data

    # --- API sklearn ---
    # fit: pelajari statistik (median, cap outlier, one-hot) HANYA dari data train, simpan sebagai state
    def fit(self, X: pd.DataFrame, y=None):
        df = self._core_clean(X)
        self.num_cols_ = df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = df.select_dtypes(include=["object"]).columns.tolist()
        self.medians_ = {c: df[c].median() for c in self.num_cols_}
        df[self.num_cols_] = df[self.num_cols_].fillna(self.medians_)
        df[self.cat_cols_] = df[self.cat_cols_].fillna("Unknown")

        self.caps_ = {}
        for col in IQR_COLS:
            q1, q3 = df[col].quantile([0.25, 0.75]); iqr = q3 - q1
            self.caps_[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        for col in P99_COLS:
            self.caps_[col] = (df[col].min(), df[col].quantile(0.99))
        for col, (lo, hi) in self.caps_.items():
            df[col] = df[col].clip(lo, hi)

        df = self._add_features(self._decompose(df))
        for col in ENGINEERED:
            hi = df[col].quantile(0.99)
            self.caps_[col] = (df[col].min(), hi)
            df[col] = df[col].clip(upper=hi)

        df = self._apply_ordinal(df)
        self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[HIGH_CARD])
        self.ohe_names_ = self.ohe_.get_feature_names_out(HIGH_CARD).tolist()
        encoded = df.drop(columns=HIGH_CARD)
        self.feature_names_ = encoded.columns.tolist() + self.ohe_names_
        return self

    # transform: terapkan statistik hasil fit ke data baru (train/test/input inference), tanpa hitung ulang
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = self._core_clean(X)
        df[self.num_cols_] = df[self.num_cols_].fillna(self.medians_)
        df[self.cat_cols_] = df[self.cat_cols_].fillna("Unknown")
        # Cap fitur dasar SEBELUM feature engineering (urutan identik dgn fit -> anti-inkonsistensi)
        for col in IQR_COLS + P99_COLS:
            lo, hi = self.caps_[col]
            df[col] = df[col].clip(lo, hi)
        df = self._add_features(self._decompose(df))
        for col in ENGINEERED:                       # cap fitur engineered (upper P99, seperti fit)
            df[col] = df[col].clip(upper=self.caps_[col][1])
        df = self._apply_ordinal(df)
        ohe_arr = self.ohe_.transform(df[HIGH_CARD])
        out = pd.concat([df.drop(columns=HIGH_CARD),
                         pd.DataFrame(ohe_arr, columns=self.ohe_names_, index=df.index)], axis=1)
        return out.reindex(columns=self.feature_names_, fill_value=0)


# Bungkus classifier yang butuh label angka (XGBoost) supaya predict()/predict_proba() tetap pakai
# label string ("Poor"/"Standard"/"Good"), konsisten dengan RandomForest & LightGBM yang terima string langsung
class _LabelDecodingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        self.base_estimator.fit(X, self.encoder_.transform(y))
        return self

    def predict(self, X):
        return self.encoder_.inverse_transform(self.base_estimator.predict(X))

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)


# 2. Training : melatih 1 model kandidat (dipanggil berkali-kali oleh pipeline.py, 1x per kandidat), catat ke MLflow
class CreditModelTrainer:
    def __init__(self, experiment_name: str = "Credit Score Classification",
                 artifact_path: str = "artifacts", test_size: float = 0.2, random_state: int = SEED):
        self.experiment_name = experiment_name
        self.artifact_dir = Path(artifact_path)
        self.test_size = test_size
        self.random_state = random_state
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_experiment(self.experiment_name)

    # Tahap 2-3 pipeline: split data, latih estimator kandidat (dikirim dari pipeline.py), catat ke MLflow
    def run(self, data_path: str | Path, model_name: str, estimator) -> Tuple[str, SkPipeline, pd.DataFrame, pd.Series]:
        print(f"--- Step 2-3: Preprocessing & Training ({model_name}) ---")
        df = pd.read_csv(Path(data_path))
        X = df.drop(columns=["Credit_Score"])
        y = df["Credit_Score"]
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)

        # XGBoost butuh label angka, bukan string -> bungkus dengan _LabelDecodingClassifier
        classifier_step = (_LabelDecodingClassifier(estimator)
                           if type(estimator).__name__ == "XGBClassifier" else estimator)

        # Pipeline training: preprocessing -> SMOTE-Tomek (imbalance) -> classifier kandidat
        pipe = ImbPipeline([
            ("preprocessing", CreditScorePreprocessor()),
            ("smote_tomek", SMOTETomek(smote=SMOTE(sampling_strategy=smt_strategy, random_state=self.random_state),random_state=self.random_state)),
            ("classifier", classifier_step),
        ])

        with mlflow.start_run(run_name=model_name) as run:
            mlflow.log_param("model", model_name)
            mlflow.log_param("imbalance", "SMOTE-Tomek (limited)")
            # Log tiap hyperparameter primitif milik estimator (skip yang bukan tipe dasar)
            mlflow.log_params({k: v for k, v in estimator.get_params().items()
                               if isinstance(v, (str, int, float, bool)) or v is None})

            pipe.fit(x_train, y_train)
            # Buang step SMOTE-Tomek untuk pipeline deployment (cuma dibutuhkan saat training, bukan saat prediksi)
            deploy_pipe = SkPipeline([
                ("preprocessing", pipe.named_steps["preprocessing"]),
                ("classifier", pipe.named_steps["classifier"]),
            ])
            mlflow.sklearn.log_model(deploy_pipe, name="model")
            run_id = run.info.run_id

        # Metrik lengkap (accuracy, macro-F1, dll) dihitung terpisah oleh ModelEvaluator, bukan di sini
        return run_id, deploy_pipe, x_test, y_test

    # Dipanggil sekali oleh pipeline.py setelah model terbaik terpilih (bukan tiap kandidat, biar tidak numpuk .pkl)
    def save_best(self, deploy_pipe: SkPipeline, model_name: str) -> Path:
        model_file = self.artifact_dir / "credit_score_pipeline.pkl"
        joblib.dump(deploy_pipe, model_file)
        print(f"✅ Model terbaik ({model_name}) tersimpan ke {model_file}")
        return model_file


if __name__ == "__main__":
    # Contoh pemakaian standalone di luar pipeline.py - training 1 model saja untuk pengetesan cepat
    trainer = CreditModelTrainer()
    ingested = Path(__file__).parent / "ingested" / "credit_score.csv"
    data = ingested if ingested.exists() else (Path(__file__).parent / "data_C.csv")
    clf = LGBMClassifier(n_estimators=600, num_leaves=127, learning_rate=0.05, random_state=SEED, n_jobs=-1, verbose=-1)
    run_id, deploy_pipe, x_test, y_test = trainer.run(data, "LightGBM", clf)
    trainer.save_best(deploy_pipe, "LightGBM")
