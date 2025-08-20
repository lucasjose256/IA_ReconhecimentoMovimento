import pandas as pd

# Ler arquivo CSV
df = pd.read_csv('/Users/lucasbarszcz/Desktop/TCC/Plank_detection.v1i.tensorflow/train/_annotations.csv', encoding='utf-8', delimiter=',')  # ajuste encoding e delimiter conforme necessário

print(df.head())
print(df["class"]=="Good-plank")

newDf = pd.DataFrame(df["class"])