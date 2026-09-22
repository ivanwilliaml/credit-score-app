import sys
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from data_ingestion import DataIngestion
from train import CreditModelTrainer, SEED
from evaluation import ModelEvaluator

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# Kandidat model = top-3 hasil GridSearchCV di notebook (hyperparameter sama persis, biar angkanya konsisten)
def build_candidates(random_state: int = SEED):
    return {
        "RandomForest": RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=2,
                                                max_features="sqrt", random_state=random_state, n_jobs=-1),
        "LightGBM": LGBMClassifier(n_estimators=600, num_leaves=127, learning_rate=0.05,
                                    random_state=random_state, n_jobs=-1, verbose=-1),
        "XGBoost": XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.05, subsample=0.8,
                                  objective="multi:softprob", eval_metric="mlogloss",
                                  random_state=random_state, n_jobs=-1, verbosity=0),
    }


# Orkestrator: menjalankan 4 tahap (ingestion -> train+evaluate tiap kandidat -> pilih terbaik -> keputusan deploy)
class CreditScorePipeline:
    def __init__(self, raw_data_path: str | Path, macro_f1_threshold: float = 0.7):
        self.base_dir = Path(__file__).parent
        self.raw_data_path = Path(raw_data_path)
        self.ingested_dir = self.base_dir / "ingested"
        self.macro_f1_threshold = macro_f1_threshold

        # Core components instantiation
        self.ingestor = DataIngestion(self.raw_data_path, self.ingested_dir)
        self.trainer = CreditModelTrainer()
        self.evaluator = ModelEvaluator()

    def execute(self):
        print("🚀 Executing Credit Score Classification Pipeline...\n")

        # 1. Handle raw data ingestion safely
        ingested_file_path = self.ingestor.run()

        # 2-3. Latih & evaluasi tiap kandidat model, semuanya tercatat sebagai run terpisah di MLflow
        results = []
        for model_name, estimator in build_candidates().items():
            run_id, deploy_pipe, x_test, y_test = self.trainer.run(ingested_file_path, model_name, estimator)
            accuracy, macro_f1, recall_poor = self.evaluator.run(run_id, x_test, y_test)
            results.append({"model": model_name, "deploy_pipe": deploy_pipe, "accuracy": accuracy,
                            "macro_f1": macro_f1, "recall_poor": recall_poor})

        # 4. Bandingkan semua kandidat, pilih Macro-F1 tertinggi sebagai model terbaik
        results.sort(key=lambda r: r["macro_f1"], reverse=True)
        best = results[0]
        print("\n--- Perbandingan Model ---")
        for r in results:
            print(f"{r['model']:<15} Macro-F1={r['macro_f1']:.4f} | Accuracy={r['accuracy']:.4f} | Recall(Poor)={r['recall_poor']:.4f}")
        print(f"\n🏆 Model terbaik: {best['model']} (Macro-F1={best['macro_f1']:.4f})")

        # 5. Simpan model terbaik saja sebagai artefak deployment (kandidat lain dibuang dari memory)
        self.trainer.save_best(best["deploy_pipe"], best["model"])

        # 6. Final conditional release assessment
        print("\n--- Deployment Approval Decision ---")
        if best["macro_f1"] >= self.macro_f1_threshold:
            print(f"🎉 Success: Macro-F1 ({best['macro_f1']:.3f}) lolos QA. Disetujui untuk deployment!")
        else:
            print(f"❌ Rejected: Macro-F1 ({best['macro_f1']:.3f}) di bawah threshold ({self.macro_f1_threshold})")


if __name__ == "__main__":
    DATA_INPUT = Path(__file__).parent / "data_C.csv"
    pipeline = CreditScorePipeline(raw_data_path=DATA_INPUT, macro_f1_threshold=0.65)
    pipeline.execute()
