import mlflow
import mlflow.sklearn
import pandas as pd
from typing import Tuple
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report)


# Tahap 4 pipeline: load model dari MLflow run, hitung metrik di test set, dan log metrik ke run yang sama
class ModelEvaluator:
    def run(self, run_id: str, x_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float, float]:
        print("--- Step 4: Evaluation ---")

        # Fetch the trained model/pipeline bundle from MLflow using run_id
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        preds = model.predict(x_test)

        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro")
        macro_prec = precision_score(y_test, preds, average="macro")
        macro_rec = recall_score(y_test, preds, average="macro")
        recall_poor = recall_score(y_test, preds, labels=["Poor"], average="macro")

        print(classification_report(y_test, preds, digits=3))
        print(f"Accuracy={acc:.3f} | Macro-F1={macro_f1:.3f} | "
              f"Macro-Precision={macro_prec:.3f} | Macro-Recall={macro_rec:.3f} | Recall(Poor)={recall_poor:.3f}")

        # Catat metrik ke MLflow run yang sama
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("macro_f1", macro_f1)
            mlflow.log_metric("macro_precision", macro_prec)
            mlflow.log_metric("macro_recall", macro_rec)
            mlflow.log_metric("recall_poor", recall_poor)

        return acc, macro_f1, recall_poor
