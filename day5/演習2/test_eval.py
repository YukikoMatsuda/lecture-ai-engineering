import pandas as pd
import time
from sklearn.metrics import accuracy_score
from joblib import load

def test_model_accuracy_and_speed():
    # モデルの読み込み
    model = load("day5/演習2/models/titanic_model.pkl")

    # データ読み込み
    df_test = pd.read_csv("day5/演習2/data/test.csv")
    X_test = df_test.drop("Survived", axis=1)
    y_test = df_test["Survived"]

    # 推論と評価
    start = time.time()
    y_pred = model.predict(X_test)
    end = time.time()

    accuracy = accuracy_score(y_test, y_pred)
    inference_time = end - start

    # テスト条件（例：精度0.8以上、推論時間2秒未満）
    assert accuracy > 0.8, f"Accuracy too low: {accuracy}"
    assert inference_time < 2, f"Inference too slow: {inference_time:.4f}秒"
