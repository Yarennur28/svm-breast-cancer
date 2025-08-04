import csv
import os 
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split


def load_data(filepath):
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader) 
        data = []
        for row in reader:
            id_val = row[0]
            *features, label = row[1:]
            features = list(map(float, features))
            label = int(label)
            data.append((id_val, features, label))
    return data


def split_data(data, train_ratio=0.7, val_ratio=0.15):
    random.shuffle(data)
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    return train_data, val_data, test_data


def prepare_features_labels(data, num_features=None):
    """
    Veriden id, özellik vektörü ve etiket listelerini ayırır.

    :param data: Girdi veri kümesi (liste)
    :param num_features: İsteğe bağlı olarak kullanılacak özellik sayısı
    :return: ids, X (np.array), y (np.array)
    """
    ids = [id_val for id_val, _, _ in data]
    X = [features[:num_features] if num_features else features for _, features, _ in data]
    y = [label for _, _, label in data]
    #y = [1 if label == 1 else -1 for _, _, label in data] 
    return ids, np.array(X), np.array(y)


def evaluate_model(ids, y_true, y_pred, output_path=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Evaluation -> Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

    if output_path:
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'true_label', 'predicted_label'])
            for id_val, true, pred in zip(ids, y_true, y_pred):
                writer.writerow([id_val, true, pred])


def plot_confusion_matrix(y_true, y_pred):
    """
    Karmaşıklık matrisini (confusion matrix) çizer.

    :param y_true: Gerçek etiketler
    :param y_pred: Tahmin edilen etiketler
    :return:
    """
    cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['1 (Malignant)', '-1 (Benign)'],
                yticklabels=['1 (Malignant)', '-1 (Benign)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Test Set)')
    plt.show()


def plot_decision_boundary(model, X, y, title="Decision Boundary", save_path=None):
    """
    2B PCA uzayında karar sınırını ve veri noktalarını görselleştirir.

    :param model: Eğitilmiş SVM modeli
    :param X: 2B PCA özellikleri
    :param y: Etiketler
    :param title: Grafik başlığı
    :param save_path: (Opsiyonel) Görselin kaydedileceği dosya yolu
    :return: None
    """
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.6)

    class_1 = y == 1
    class_neg1 = y == -1
    plt.scatter(X[class_1, 0], X[class_1, 1],
                 c='red', label='Malignant (1)', edgecolors='k')
    plt.scatter(X[class_neg1, 0], X[class_neg1, 1],
                 c='blue', label='Benign (-1)', edgecolors='k')


    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('PCA Feature 1')
    plt.ylabel('PCA Feature 2')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Karar sınırı görseli kaydedildi: {save_path}")

    plt.show()


if __name__ == "__main__":
    data = load_data("processed_data.csv")
    train_data, val_data, test_data = split_data(data)

    print("Training SVM with scikit-learn (all 30 features)...")

    
    _, X_train, y_train = prepare_features_labels(train_data)
    _, X_val, y_val = prepare_features_labels(val_data)
    test_ids, X_test, y_test = prepare_features_labels(test_data)

    
    model = SVC(kernel='linear', C=10)
    model.fit(X_train, y_train)

    
    print("\nTrain Set:")
    train_preds = model.predict(X_train)
    evaluate_model([None] * len(y_train), y_train, train_preds)

    
    print("\nValidation Set:")
    val_preds = model.predict(X_val)
    evaluate_model([None] * len(y_val), y_val, val_preds)

    
    print("\nTest Set:")
    test_preds = model.predict(X_test)
    evaluate_model(test_ids, y_test, test_preds, output_path="test_results_sklearn.csv")
    plot_confusion_matrix(y_test, test_preds)

    
    print("\nApplying PCA for visualization...")
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])

    pca = PCA(n_components=2)
    X_all_pca = pca.fit_transform(X_all)


    model_pca = SVC(kernel='linear', C=10)
    model_pca.fit(X_all_pca, y_all)

    
    print("\nDecision Boundary (Train + Validation PCA 2D):")
    plot_decision_boundary(model_pca, X_all_pca, y_all, save_path="images/decision_boundary_pca.png")

