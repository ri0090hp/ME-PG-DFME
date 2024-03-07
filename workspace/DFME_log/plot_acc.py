import pandas as pd
import matplotlib.pyplot as plt

file_path = 'MNIST/accuracy.csv'
data = pd.read_csv(file_path)

# epochとaccuracyを抽出
epochs = data['epoch']
accuracy = data['accuracy']

# plt.xscale("log")

# x軸の値を作成
x_values = [i for i in range(1, len(epochs) + 1)]

# 結果のプロット
plt.plot(x_values, accuracy, color='green', label="Fashionmnist", linestyle='--')

# グラフにタイトルを設定
plt.title('Model Results')

# x軸とy軸にラベルを設定
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0, 100)
# plt.gca().set_xticks([1, 10, 100]) # Comment this out
# plt.gca().set_xticklabels([10**3, 10**4, 10**5]) # Comment this out

plt.legend()
plt.show()
