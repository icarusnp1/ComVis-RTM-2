import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue

# Gambar grayscale simulasi
img = np.array([
    [9, 9, 9, 9, 9],
    [9, 5, 4, 5, 9],
    [9, 4, 2, 4, 9],
    [9, 5, 4, 5, 9],
    [9, 9, 9, 9, 9]
], dtype=np.uint8)

plt.imshow(img, cmap='gray')
plt.title("Peta Ketinggian (semakin gelap = lebih rendah)")
plt.show()

# --- Langkah 1: cari titik minimum (lembah) ---
min_val = img.min()
seed_points = np.argwhere(img == min_val)  # titik awal air
print("Seed points (lembah):", seed_points)

# --- Langkah 2: siapkan label untuk setiap pixel ---
labels = np.zeros_like(img, dtype=np.int32)
label_id = 1
for (y, x) in seed_points:
    labels[y, x] = label_id
    label_id += 1

# --- Langkah 3: siapkan priority queue (berdasarkan ketinggian pixel) ---
pq = PriorityQueue()
for (y, x) in seed_points:
    pq.put((img[y, x], (y, x)))

# --- Fungsi bantu untuk tetangga ---
def neighbors(y, x, shape):
    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        ny, nx = y+dy, x+dx
        if 0 <= ny < shape[0] and 0 <= nx < shape[1]:
            yield ny, nx

# --- Langkah 4: proses flooding ---
while not pq.empty():
    height, (y, x) = pq.get()
    for ny, nx in neighbors(y, x, img.shape):
        if labels[ny, nx] == 0:  # belum dilabel
            labels[ny, nx] = labels[y, x]  # isi dengan label lembah
            pq.put((img[ny, nx], (ny, nx)))
        elif labels[ny, nx] != labels[y, x]:
            labels[y, x] = -1  # bendungan

plt.imshow(labels, cmap='nipy_spectral')
plt.title("Hasil Segmentasi Manual Watershed")
plt.show()
