import csv
import random
import math

def load_data(filepath):
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # başlık satırını atla
        data = []
        for row in reader:
            id_val = row[0]
            *features, label = row[1:]
            features = list(map(float, features))
            label = int(label)
            #label = 1 if label == 1 else -1
            data.append((id_val, features, label))
    return data

def split_data(data, train_ratio=0.7, val_ratio=0.15):
    random.shuffle(data) # verileri karıştır
    total = len(data) 
    train_end = int(total * train_ratio) # train bittiği yer
    val_end = int(total * (train_ratio + val_ratio)) # train + val bittiği yer
    train_data = data[:train_end] # başlangıçtan train_end'e kadar eğitim verisi
    val_data = data[train_end:val_end] # train_end ile val_end arası doğrulama verileri
    test_data = data[val_end:] # geriye kalan yani val_end'den sonrası test verisi
    return train_data, val_data, test_data

def dot_product(x, w):
    return sum(xi * wi for xi, wi in zip(x, w)) # yan yana eşleşen x,w çiftlerini çarp sonra topla

def predict(x, w, b):
    result = dot_product(x, w) + b
    return 1 if result >= 0 else -1 # xw +b > 0 ise 1 yani M

def compute_loss(data, w, b, C):
    loss = 0.0
    for _, x, y in data: # hinge loss + regularization 
        margin = y * (dot_product(x, w) + b)
        loss += max(0, 1 - margin) # margin >= 1 ise ceza 0 gelir, margin<1 ise ceza loss'a ekle
    regularization = 0.5 * sum(wi**2 for wi in w) # ağırlık cezalandır
    return regularization + C * loss

def f1_score(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == 1) # pozitif sınıf doğru tahmin edilen örneklerin sayısı
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == -1 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == -1)
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)

def train_svm(train_data, epochs=100, lr=0.001, C=10): # lineer svm stokastik gd ile yazıldı 
    feature_len = len(train_data[0][1]) # ilk örneğin özellik sayısı alınır
    w = [0.0] * feature_len
    b = 0.0

    for epoch in range(1, epochs + 1):
        random.shuffle(train_data)  # her epoch başında veriyi karıştır
        for _, x, y in train_data:
            margin = y * (dot_product(x, w) + b)
            if margin >= 1: # doğru tahminse ağırlıkları biraz küçült sadece regularization
                w = [wi - lr * wi for wi in w]  
            else:
                w = [wi - lr * (wi - C * y * xi) for wi, xi in zip(w, x)]
                b += lr * C * y

        loss = compute_loss(train_data, w, b, C)
        predictions = [predict(x, w, b) for _, x, _ in train_data]
        labels = [y for _, _, y in train_data]
        f1 = f1_score(labels, predictions)
        print(f"Epoch {epoch}: Train Loss = {loss:.4f}, Train F1 = {f1:.4f}")

    norm_w = math.sqrt(sum(wi**2 for wi in w)) # öklidyen normu w vektörü için 
    print(f"Final ||w|| = {norm_w:.4f}")
    #print(f"Final b = {b:.4f}")
    return w, b


def evaluate(data, w, b, output_path=None):
    predictions = []
    labels = []
    results = []

    for id_val, x, y in data:
        pred = predict(x, w, b)
        predictions.append(pred)
        labels.append(y)
        results.append((id_val, y, pred))

    correct = sum(1 for y_true, y_pred in zip(labels, predictions) if y_true == y_pred) # doğru tahminlerin sayısı tutulur
    accuracy = correct / len(data)
    f1 = f1_score(labels, predictions)

    print(f"Evaluation -> Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

    if output_path:
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'true_label', 'predicted_label'])
            for row in results:
                writer.writerow(row)

if __name__ == "__main__":
    data = load_data("processed_data.csv")
    train_data, val_data, test_data = split_data(data)

    print("Training SVM...")
    w, b = train_svm(train_data)

    print("\nValidation Set:")
    evaluate(val_data, w, b)

    print("\nTest Set:")
    evaluate(test_data, w, b, output_path="test_results_scratch.csv")
