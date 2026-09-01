import os
import matplotlib.pyplot as plt

# Nilai evaluasi berdasarkan input terbaik MobileNetV2
acc = 97.22
precision = 97.50
recall = 97.22
f1 = 97.35

print(f"Accuracy: {acc:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1-Score: {f1:.2f}%")

# Plot bar chart metrik
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [acc, precision, recall, f1]
colors = ['#4CAF50', '#2196F3', '#FFC107', '#9C27B0']

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(metrics, values, color=colors)

ax.set_ylim([0, 110])
ax.set_title('Model Evaluation Metrics (MobileNetV2)', fontsize=16, fontweight='bold')
ax.set_ylabel('Percentage (%)', fontsize=12)

# Tampilkan nilai di atas setiap bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.2f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300)
print("Berhasil menyimpan gambar ke 'model_evaluation.png'")
