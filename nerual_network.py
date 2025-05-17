# creditcard_nn_pipeline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score

# Keras imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam

# 1. Load data
df = pd.read_csv('creditcard.csv')  # Đổi tên file nếu cần

# 2. Tách đặc trưng và nhãn
X = df.drop(columns='Class')
y = df['Class']

# 3. Chia train/test (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4. Định nghĩa hàm tạo mô hình Neural Network
def create_nn_model(input_dim):
    model = Sequential()
    model.add(Dense(32, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

# 5. Hàm tạo pipeline với model neural network
def create_pipeline_nn(input_dim):
    nn_model = KerasClassifier(build_fn=lambda: create_nn_model(input_dim), epochs=20, batch_size=64, verbose=0)
    pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', nn_model)
    ])
    return pipeline

def main():
    input_dim = X_train.shape[1]
    pipeline = create_pipeline_nn(input_dim)

    # 6. Huấn luyện
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # 7. Dự đoán
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # 8. Đánh giá
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")

if __name__ == "__main__":
    main()
