import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def watershed():
    root = os.getcwd()
    img = cv.imread(os.path.join(root, 'botol_obat_bertumpuk.png'))
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 1. Threshold biner
    _, thresh = cv.threshold(gray, 60, 255, cv.THRESH_BINARY_INV)

    # 2. Noise removal (opening)
    kernel = np.ones((3,3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    # 3. Sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    # 4. Distance transform dan threshold untuk sure foreground
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    _, sure_fg = cv.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

    # 5. Unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # 6. Marker labeling
    _, markers = cv.connectedComponents(sure_fg)

    # Tambahkan +1 agar background bukan 0
    markers = markers + 1

    # Tandai unknown region sebagai 0
    markers[unknown == 255] = 0

    # 7. Watershed
    markers = cv.watershed(imgRGB, markers)
    imgRGB[markers == -1] = [255, 0, 0]  # boundary warna merah

    # 8. Tampilkan hasil
    plt.figure(figsize=(12,8))
    plt.subplot(231); plt.title("Asli"); plt.imshow(imgRGB)
    plt.subplot(232); plt.title("Gray"); plt.imshow(gray, cmap='gray')
    plt.subplot(233); plt.title("Thresh"); plt.imshow(thresh, cmap='gray')
    plt.subplot(234); plt.title("Sure BG"); plt.imshow(sure_bg, cmap='gray')
    plt.subplot(235); plt.title("Sure FG"); plt.imshow(sure_fg, cmap='gray')
    plt.subplot(236); plt.title("Hasil Watershed"); plt.imshow(imgRGB)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    watershed()
