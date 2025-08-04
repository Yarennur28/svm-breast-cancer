import csv

def preprocess_data(input_path='data.csv', output_path='processed_data.csv'):
    data = []
    with open(input_path, 'r') as infile:
        reader = csv.reader(infile)
        header = next(reader)  
        for row in reader:
            if len(row) != len(header):
                continue  
            id_val = row[0]
            diagnosis = 1 if row[1] == 'M' else -1 # M,B etiketleme işlemi
            features = list(map(float, row[2:]))  
            data.append((id_val, features, diagnosis))


    feature_count = len(data[0][1])
    min_vals = [float('inf')] * feature_count # max ve min en başta sonsuz değerler olarak belirle
    max_vals = [float('-inf')] * feature_count

    for _, features, _ in data: # features listesini dolaş ve her sütun için min max güncelle
        for i in range(feature_count):
            min_vals[i] = min(min_vals[i], features[i])
            max_vals[i] = max(max_vals[i], features[i])

    
    for i in range(len(data)): # min-max normalizasyon
        normalized_features = []
        for j in range(feature_count):
            val = data[i][1][j]
            min_val = min_vals[j]
            max_val = max_vals[j]
            if max_val - min_val == 0:
                normalized = 0.0 
            else:
                normalized = (val - min_val) / (max_val - min_val)
            normalized_features.append(normalized)
        data[i] = (data[i][0], normalized_features, data[i][2])


    with open(output_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['id'] + header[2:] + ['label'])  
        for id_val, features, label in data:
            writer.writerow([id_val]+ features + [label])
