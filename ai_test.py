import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# === 1️⃣ BACA GAMBAR DAN PREPROCESSING ===
img = cv.imread('botol_berjajar.png')  # ganti nama file sesuai lokasi kamu
img = cv.resize(img, (800, 300))  # ubah ukuran agar seragam
blur = cv.GaussianBlur(img, (5, 5), 0)

# Konversi ke HSV untuk segmentasi warna
hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

# === 2️⃣ DEFINISIKAN RANGE WARNA (HSV) UNTUK MERAH, ORANYE, KUNING ===
# Catatan: nilai ini bisa kamu sesuaikan hasil real gambar kamu
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])
mask_red = cv.inRange(hsv, lower_red1, upper_red1) + cv.inRange(hsv, lower_red2, upper_red2)

lower_orange = np.array([10, 100, 100])
upper_orange = np.array([25, 255, 255])
mask_orange = cv.inRange(hsv, lower_orange, upper_orange)

lower_yellow = np.array([25, 100, 100])
upper_yellow = np.array([35, 255, 255])
mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)

# Gabungkan semua mask warna
mask_total = mask_red + mask_orange + mask_yellow

# === 3️⃣ MORPHOLOGICAL FILTERING UNTUK MEMBERSIHKAN MASK ===
kernel = np.ones((5, 5), np.uint8)
mask_clean = cv.morphologyEx(mask_total, cv.MORPH_CLOSE, kernel, iterations=2)
mask_clean = cv.morphologyEx(mask_clean, cv.MORPH_OPEN, kernel, iterations=1)

# === 4️⃣ TEMUKAN KONTOUR OBJEK BOTOL ===
contours, _ = cv.findContours(mask_clean, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

result = img.copy()
botol_data = []

for c in contours:
    area = cv.contourArea(c)
    if area > 10000:  # abaikan noise kecil
        x, y, w, h = cv.boundingRect(c)

        # Ambil ROI untuk analisis warna dominan
        roi = img[y:y+h, x:x+w]
        hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        mean_color = cv.mean(hsv_roi, mask=None)
        hue_value = mean_color[0]

        # Tentukan warna dominan berdasarkan HUE rata-rata
        if hue_value < 10 or hue_value > 160:
            warna = "Merah"
        elif 10 <= hue_value < 25:
            warna = "Oranye"
        else:
            warna = "Kuning"

        # Tentukan ukuran berdasarkan tinggi bounding box
        ukuran = "Besar" if h > 170 else "Kecil"

        # Gambar kotak dan label
        cv.drawContours(result, [c], -1, (0, 255, 0), 2)  # bentuk asli
        cv.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 1)  # kotak pembatas tipis

        cv.putText(result, f"{warna}, {ukuran}", (x, y-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        botol_data.append({
            "x": x, "y": y, "w": w, "h": h, "warna": warna, "ukuran": ukuran, "area": area
        })

# Urutkan botol dari kiri ke kanan berdasarkan posisi x
botol_data = sorted(botol_data, key=lambda d: d['x'])

# === 5️⃣ TAMPILKAN HASIL ===
plt.figure(figsize=(10,4))
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.title("Hasil Segmentasi Botol Berdasarkan Warna & Ukuran")
plt.axis("off")
plt.show()

# Cetak data botol
for i, b in enumerate(botol_data, 1):
    print(f"Botol {i}: Warna={b['warna']}, Ukuran={b['ukuran']}, Area={int(b['area'])}, h={b['h']}, w={b['w']} ")
