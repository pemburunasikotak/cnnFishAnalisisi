import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Daftar teks untuk setiap kotak
steps = [
    "Pengumpulan Dataset",
    "Data Preprocessing",
    "Data Augmentation",
    "Training Model CNN",
    "Evaluasi Model CNN",
    "Konversi TensorFlow Lite",
    "Implementasi Smartphone",
    "Pengujian Sistem"
]

fig, ax = plt.subplots(figsize=(6, 12))

# Konfigurasi ukuran dan posisi kotak
box_width = 0.6
box_height = 0.06
spacing = 0.11

# Warna-warni untuk mempercantik
colors = ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5', '#2196F3', '#1E88E5', '#1976D2']

for i, step in enumerate(steps):
    # Posisi Y dihitung dari atas ke bawah
    y = 0.9 - i * spacing
    
    # Buat kotak (Rounded Rectangle)
    rect = patches.FancyBboxPatch(
        (0.5 - box_width/2, y), box_width, box_height,
        boxstyle="round,pad=0.03", 
        linewidth=2, 
        edgecolor='#0D47A1', 
        facecolor=colors[i]
    )
    ax.add_patch(rect)
    
    # Tambahkan Teks di tengah kotak
    ax.text(0.5, y + box_height/2, step, 
            ha='center', va='center', fontsize=12, fontweight='bold', color='black' if i < 5 else 'white')
    
    # Tambahkan Panah ke bawah (kecuali kotak terakhir)
    if i < len(steps) - 1:
        ax.annotate('', 
                    xy=(0.5, y - 0.02), xycoords='data',
                    xytext=(0.5, y), textcoords='data',
                    arrowprops=dict(arrowstyle="->", lw=2.5, color='#0D47A1'))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')  # Hilangkan axis

plt.title("Alur Sistem", fontsize=18, fontweight='bold', pad=20, color='#0D47A1')
plt.tight_layout()
plt.savefig('flowchart.png', dpi=300, bbox_inches='tight', transparent=False)
print("Berhasil menyimpan gambar flowchart ke 'flowchart.png'")
