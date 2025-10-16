import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

# -----------------------------
# 1️⃣ Load dan Preprocessing
# -----------------------------
img = cv.imread('botol_berjajar_berdempetan_povatas1.png')
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Blur untuk mengurangi noise
blur = cv.GaussianBlur(gray, (5,5), 0)

# Threshold untuk pisahkan objek dan background
_, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# -----------------------------
# 2️⃣ Morphology Cleanup
# -----------------------------
kernel = np.ones((3,3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv.dilate(opening, kernel, iterations=3)

# -----------------------------
# 3️⃣ Distance Transform (untuk area inti objek)
# -----------------------------
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
_, sure_fg = cv.threshold(dist_transform, 0.8 * dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

# -----------------------------
# 4️⃣ Marker untuk Watershed
# -----------------------------
num_labels, markers = cv.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# -----------------------------
# 5️⃣ Jalankan Watershed
# -----------------------------
markers_ws = cv.watershed(img, markers)
segmented = img_rgb.copy()
segmented[markers_ws == -1] = [255, 0, 0]  # garis batas merah

# -----------------------------
# 6️⃣ Deteksi dan Label Warna per Botol
# -----------------------------
result = img_rgb.copy()
contours, _ = cv.findContours(np.uint8(markers_ws > 1), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

def detect_color(bgr_color):
    """Fungsi bantu untuk deteksi warna dominan sederhana."""
    hsv = cv.cvtColor(np.uint8([[bgr_color]]), cv.COLOR_BGR2HSV)[0][0]
    h = hsv[0]
    if h < 10 or h > 160:
        return "Merah"
    elif 25 <= h < 35:
        return "Kuning"
    elif 35 <= h < 85:
        return "Hijau"
    elif 85 <= h < 130:
        return "Biru"
    else:
        return "Anomali"

botol_data = []
for c in contours:
    area = cv.contourArea(c)
    if area > 1000:
        x, y, w, h = cv.boundingRect(c)
        roi = img[y:y+h, x:x+w]
        avg_color = cv.mean(roi)[:3]  # Rata-rata warna BGR
        warna = detect_color(avg_color)

        # Gambar bounding box
        cv.rectangle(result, (x, y), (x+w, y+h), (0,255,0), 2)
        cv.putText(result, warna, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (10,255,255), 2)

        botol_data.append({
            "warna": warna,
            "area": int(area),
            "x": x
        })

botol_data = sorted(botol_data, key=lambda d: d['x'])

# -----------------------------
# 7️⃣ Visualisasi Lengkap (Grid)
# -----------------------------
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
ax = axes.ravel()

ax[0].imshow(img_rgb)
ax[0].set_title('1. Gambar Asli')

ax[1].imshow(gray, cmap='gray')
ax[1].set_title('2. Grayscale')

ax[2].imshow(blur, cmap='gray')
ax[2].set_title('3. Gaussian Blur')

ax[3].imshow(thresh, cmap='gray')
ax[3].set_title('4. Threshold + Otsu')

ax[4].imshow(opening, cmap='gray')
ax[4].set_title('5. Morphology (Open)')

ax[5].imshow(sure_bg, cmap='gray')
ax[5].set_title('6. Sure Background')

ax[6].imshow(dist_transform, cmap='jet')
ax[6].set_title('7. Distance Transform')

ax[7].imshow(sure_fg, cmap='gray')
ax[7].set_title('8. Sure Foreground')

ax[8].imshow(unknown, cmap='gray')
ax[8].set_title('9. Unknown Region')

ax[9].imshow(markers, cmap='nipy_spectral')
ax[9].set_title('10. Markers (Label)')

ax[10].imshow(segmented)
ax[10].set_title('11. Hasil Watershed (Batas)')

ax[11].imshow(result)
ax[11].set_title('12. Bounding Box + Warna')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()

# -----------------------------
# 8️⃣ Cetak Data Botol
# -----------------------------
for i, b in enumerate(botol_data, 1):
    print(f"Botol {i}: Warna={b['warna']}, Area={b['area']}")
