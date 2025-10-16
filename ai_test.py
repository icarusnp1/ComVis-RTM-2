import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

# -----------------------------
# 1️⃣ Load dan Preprocessing
# -----------------------------
img = cv.imread('botol_berjajar_berdempetan8.png')
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Blur untuk mengurangi noise
blur = cv.GaussianBlur(gray, (5,5), 0)

# Threshold untuk pisahkan objek dan background
_, thresh = cv.threshold(gray, 190, 255, cv.THRESH_BINARY_INV)

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
_, sure_fg = cv.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)
# _, sure_fg = cv.threshold(dist_transform, 75, 255, 0)

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
# 6️⃣ Fungsi Deteksi Warna
# -----------------------------
def detect_color(bgr_color):
    hsv = cv.cvtColor(np.uint8([[bgr_color]]), cv.COLOR_BGR2HSV)[0][0]
    h = hsv[0]
    if h < 10 or h > 160:
        return "Merah", (255, 0, 0)
    elif 25 <= h < 35:
        return "Kuning", (255, 255, 0)
    elif 35 <= h < 85:
        return "Hijau", (0, 255, 0)
    elif 85 <= h < 130:
        return "Biru", (0, 0, 255)
    else:
        return "Anomali", (255, 255, 255)

# -----------------------------
# 7️⃣ Deteksi, Warna, dan Ukuran per Botol
# -----------------------------
result = img_rgb.copy()
botol_data = []

unique_labels = np.unique(markers_ws)

for label in unique_labels:
    if label <= 1:  # 0 dan 1 = background, -1 = garis batas
        continue

    mask = np.uint8(markers_ws == label)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        continue

    c = max(contours, key=cv.contourArea)
    area = cv.contourArea(c)
    if area < 1000:
        continue

    x, y, w, h = cv.boundingRect(c)
    roi = img[y:y+h, x:x+w]
    avg_color = cv.mean(roi)[:3]

    warna, box_color = detect_color(avg_color)
    ukuran = "Besar" if area > 7000 else "Kecil"

    cv.rectangle(result, (x, y), (x+w, y+h), box_color, 3)
    cv.putText(result, f"{warna} - {ukuran}", (x, y-10),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    botol_data.append({
        "warna": warna,
        "ukuran": ukuran,
        "area": int(area),
        "x": x
    })

botol_data = sorted(botol_data, key=lambda d: d['x'])

# -----------------------------
# 8️⃣ Visualisasi Lengkap (Grid)
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
ax[3].set_title('4. Threshold')

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
ax[11].set_title('12. Bounding Box + Warna + Ukuran')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()

# -----------------------------
# 9️⃣ Cetak Data Botol
# -----------------------------
for i, b in enumerate(botol_data, 1):
    print(f"Botol {i}: Warna={b['warna']}, Ukuran={b['ukuran']}, Area={b['area']}")
