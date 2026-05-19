# 🔏 Digital Watermarking — DCT Spread Spectrum

Repositori ini berisi implementasi **invisible digital watermarking** pada citra menggunakan metode **DCT Spread Spectrum**. Sistem menyisipkan watermark (logo/pesan) ke dalam domain frekuensi (DCT) dari sebuah gambar sehingga perubahan visualnya tidak kasat mata, namun watermark tetap dapat diekstrak kembali meskipun gambar telah dikompresi.

---

## 🖼️ Perbandingan Visual (Showcase)

Berikut adalah hasil penyisipan watermark menggunakan script ini:

| Gambar Host (Asli) | Watermark (Logo) | Hasil Watermarked |
| :---: | :---: | :---: |
| ![Host](images/Test_Project.jpg) | ![Watermark](images/watermark.png) | ![Result](images/Test_Project_watermarked.jpg) |

> **Catatan:** Perhatikan bahwa secara visual, hampir tidak ada perbedaan antara Gambar Host dan Hasil Watermarked (bersifat *invisible*), namun di dalam frekuensi DCT gambar tersebut telah tersimpan informasi dari Logo Watermark.

---

## 📁 Struktur Folder

```text
WATERMARKING/
├── docs/
│   └── Tugas Watermarking.pdf           ← Referensi teori & tugas
├── images/                              ← Folder input & output gambar
│   ├── Test_Project.jpg                 ← Gambar host (Asli)
│   ├── watermark.png                    ← Gambar watermark (Logo)
│   ├── Test_Project_watermarked.jpg     ← Hasil gambar yang telah disisipi watermark
│   ├── Test_Project_watermarked_qf*.jpg ← Hasil uji kompresi JPEG (berbagai QF)
│   └── watermark_evaluation.png         ← Grafik hasil evaluasi (BER, NC, PSNR)
├── src/
│   └── watermark_dct.py                 ← Script utama pemrosesan
├── .gitattributes
└── README.md
```

---

## ⚙️ Instalasi & Cara Menjalankan

Pastikan Anda memiliki **Python 3** terinstall, lalu install pustaka yang dibutuhkan:

```bash
pip install numpy opencv-python matplotlib pillow
```

Jalankan program dari direktori utama project. Program otomatis akan mencari gambar di folder `images/` dan menyisipkan `watermark.png`.

```bash
python src/watermark_dct.py image
```

---

## 📖 Alur Sistem (Step-by-Step)

Berikut adalah penjelasan langkah demi langkah bagaimana program `src/watermark_dct.py` bekerja memproses gambar `images/Test_Project.jpg` dan menyisipkan `images/watermark.png`.

### 1. Load Gambar Host
Sistem pertama kali akan membaca gambar host (`Test_Project.jpg`). Gambar diubah ukurannya menjadi 256×256 piksel agar ukuran seragam.
```python
def load_or_create_host(path=None, size=256):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return img
```

### 2. Membuat / Load Watermark
Sistem memuat gambar `watermark.png`, me-resize menjadi 32×32 piksel, lalu melakukan binarisasi (hitam=0, putih=1). Hasilnya diratakan (flatten) menjadi array 1D berisi **1024 bit**.
```python
def generate_watermark(n_bits=WATERMARK_BITS, mode="image", image_path=None):
    wm_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    dim = int(np.sqrt(n_bits)) # 32
    wm_resized = cv2.resize(wm_img, (dim, dim))
    _, wm_bin = cv2.threshold(wm_resized, 128, 1, cv2.THRESH_BINARY)
    return wm_bin.flatten().astype(np.float32)
```

### 3. Konversi Gambar RGB ke YCrCb
Sebelum watermark disisipkan, `Test_Project.jpg` diubah dari ruang warna RGB ke YCrCb. Sistem hanya mengambil **channel Y (Luminance)** karena channel ini mewakili informasi terang-gelap gambar yang lebih relevan terhadap struktur visual dan lebih kuat terhadap kompresi JPEG.
```python
img_ycbcr = cv2.cvtColor(host_img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
Y = img_ycbcr[:, :, 0]
```

### 4. Transformasi DCT
Channel Y diubah dari domain spasial (piksel) ke domain frekuensi menggunakan 2D *Discrete Cosine Transform* (DCT).
```python
Y_dct = cv2.dct(Y)
flat_dct = Y_dct.flatten()
```

### 5. Memilih Koefisien Mid-Frequency
Sistem tidak menyisipkan watermark pada frekuensi rendah (akan merusak gambar) atau frekuensi tinggi (akan hilang saat kompresi JPEG). Oleh karena itu, dipilih area pita frekuensi menengah (*mid-frequency*).
```python
n_total = len(flat_dct)
mid_start = n_total // 6
mid_end   = n_total // 2
n_coeffs  = (mid_end - mid_start) // n_bits
```

### 6. Membuat Pseudo-Noise (PN) Sequence
Untuk menyebar sinyal watermark (*spread spectrum*), sistem membuat PN sequence (berisi -1.0 dan 1.0) menggunakan kunci rahasia (`SEED = 42`).
```python
def get_pn_sequences(n_bits, n_coeffs, seed=SEED):
    rng = np.random.default_rng(seed)
    pn = rng.choice([-1.0, 1.0], size=(n_bits, n_coeffs))
    return pn
```

### 7. Penyisipan Watermark (Embedding)
Watermark disisipkan menggunakan metode *additive spread spectrum*. Jika bit bernilai 1, PN sequence ditambahkan. Jika 0, PN sequence dikurangkan. Kekuatan penyisipan diatur oleh variabel `ALPHA = 8.0`.
```python
for i, bit in enumerate(watermark_bits):
    polarity = 1.0 if bit >= 0.5 else -1.0
    idx_start = mid_start + i * n_coeffs
    idx_end   = idx_start + n_coeffs
    flat_dct[idx_start:idx_end] += ALPHA * polarity * pn_seqs[i]
```

### 8. Inverse DCT & Penggabungan Channel
Matriks frekuensi dikembalikan ke domain piksel dengan Inverse DCT. Nilainya dibatasi dalam rentang 0-255, lalu digabungkan kembali dengan channel Cb dan Cr asli, dan dikonversi ke RGB.
```python
Y_wm = cv2.idct(flat_dct.reshape(H, W))
Y_wm = np.clip(Y_wm, 0, 255)

img_wm = img_ycbcr.copy()
img_wm[:, :, 0] = Y_wm
watermarked = cv2.cvtColor(img_wm.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
```
*Hasil dari langkah ini disimpan sebagai `Test_Project_watermarked.jpg`.*

### 9. Ekstraksi Watermark
Proses ekstraksi adalah kebalikan dari embedding. Gambar yang diterima diubah ke DCT, kemudian dihitung korelasinya dengan PN sequence yang dibangkitkan dari `SEED` yang sama. Jika korelasi > 0, bit dianggap 1, sebaliknya bit dianggap 0.
```python
extracted = np.zeros(n_bits, dtype=np.float32)
for i in range(n_bits):
    idx_start = mid_start + i * n_coeffs
    idx_end   = idx_start + n_coeffs
    corr = np.dot(flat_dct[idx_start:idx_end], pn_seqs[i])
    extracted[i] = 1.0 if corr > 0 else 0.0
```

### 10. Evaluasi Ketahanan (Robustness) terhadap Kompresi JPEG
Sistem memvalidasi ketahanan watermark dengan melakukan kompresi JPEG menggunakan berbagai tingkat *Quality Factor* (QF). Semakin kecil QF, semakin kuat kompresinya.
```python
def jpeg_compress(img_rgb, quality):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', img_bgr, encode_params)
    decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
```

---

## 📈 Metrik & Hasil Evaluasi

Sistem menggunakan tiga metrik utama untuk evaluasi:
1. **BER (Bit Error Rate)**: Proporsi bit yang salah ekstrak. (*Ideal: 0.0, Batas sukses < 0.10*)
2. **NC (Normalized Correlation)**: Tingkat kemiripan pola watermark yang diekstrak dengan aslinya. (*Ideal: +1.0*)
3. **PSNR (Peak Signal-to-Noise Ratio)**: Kualitas visual gambar bersisipan dibanding aslinya. (*Ideal: > 30 dB*)

**Hasil Eksperimen pada `Test_Project.jpg` (`ALPHA = 8.0`, Mode: `image` 1024 bit)**:

| QF  | BER    | NC      | PSNR (dB) | Status Ekstraksi | Keterangan |
| --- | ------ | ------- | --------- | ---------------- | ---------- |
| 5   | 0.4941 | +0.0117 | 23.9      | ✗ Gagal          | Kompresi terlalu agresif menghancurkan pita mid-frequency |
| 30  | 0.2754 | +0.4492 | 29.6      | ✗ Gagal          | Kuantisasi JPEG masih merusak struktur watermark |
| 60  | 0.1396 | +0.7207 | 31.9      | ✗ Gagal          | Mendekati batas ekstraksi yang wajar |
| 70  | 0.0908 | +0.8184 | 32.9      | ✓ Berhasil       | Mulai titik ini, watermark dapat diekstrak secara visual |
| 80  | 0.0557 | +0.8887 | 34.4      | ✓ Berhasil       | Ekstraksi sangat baik |
| 100 | 0.0410 | +0.9180 | 45.2      | ✓ Berhasil       | PSNR tinggi, tanpa kompresi kehilangan (lossless behavior) |

> **Kesimpulan:** 
> PSNR gambar asli vs watermarked adalah **34.98 dB**, yang menandakan penyisipan watermark bersifat *invisible* dan aman untuk penglihatan.
> Namun, karena pesan yang dimasukkan cukup besar (1024 bit), ketahanan terhadap JPEG sedikit berkurang dibandingkan watermark ukuran kecil. Ekstraksi mulai berhasil dan stabil di **QF ≥ 70**.
