# 🔏 Digital Watermarking — DCT Spread Spectrum

Implementasi **invisible digital watermarking** pada citra wajah menggunakan metode **DCT Spread Spectrum**.  
Watermark disisipkan ke domain frekuensi (DCT) sehingga tidak terlihat mata namun dapat diekstrak kembali.

---

## 📁 Struktur Folder

```
WATERMARKING/
├── docs/
│   └── Tugas Watermarking.pdf
├── images/                          ← Input & Output gambar
│   ├── Test_Project.jpg             ← Foto asli (input)
│   ├── Test_Project_watermarked.jpg ← Hasil watermark (output)
│   └── watermark_evaluation.png     ← Grafik evaluasi (output)
├── src/
│   └── watermark_dct.py             ← Script utama
├── .gitattributes
└── README.md
```

> **Aturan:** Script **hanya memproses file `.jpg`/`.jpeg` di folder `images/`**.  
> Semua output (gambar watermarked + grafik evaluasi) otomatis tersimpan ke `images/`.

---

## ⚙️ Instalasi

Pastikan Python 3 sudah terinstall, lalu install dependencies:

```bash
pip install numpy opencv-python matplotlib pillow
```

---

## 🚀 Cara Menjalankan

Jalankan dari folder `src/`:

```bash
cd src

# Mode watermark biner (default)
python watermark_dct.py

# Mode watermark acak
python watermark_dct.py random
```

> Script akan otomatis mencari **semua file JPG** di folder `images/` dan memprosesnya satu per satu.  
> File yang sudah diberi suffix `_watermarked` akan dilewati agar tidak diproses ulang.

**Contoh output di terminal:**

```
============================================================
  DCT Spread Spectrum Watermarking
  Folder input/output : C:\...\WATERMARKING\images
  Mode watermark      : binary
============================================================

[✓] Ditemukan 1 file JPG:
    • Test_Project.jpg

[>] Memproses : Test_Project.jpg
    Output    : Test_Project_watermarked.jpg

[✓] Watermark embedded — PSNR host vs watermarked: 34.91 dB
[✓] Tersimpan : .../images/Test_Project_watermarked.jpg
[✓] Grafik evaluasi disimpan : .../images/watermark_evaluation.png

============================================================
  Selesai! Hasil tersimpan di folder:
  C:\...\WATERMARKING\images
============================================================
```

---

## 🔧 Konfigurasi

Edit bagian atas `src/watermark_dct.py` untuk mengubah parameter:

```python
ALPHA          = 8.0   # Kekuatan watermark
                       # ↑ Besar = lebih tahan JPEG, tapi gambar sedikit berubah
                       # ↓ Kecil = tidak terlihat, tapi mudah hancur oleh kompresi

WATERMARK_BITS = 64    # Jumlah bit pesan watermark

SEED           = 42    # Kunci rahasia — HARUS SAMA saat embedding & extraction
                       # Ganti nilai ini untuk membuat watermark unik

QF_VALUES = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                       # Nilai Quality Factor JPEG yang diuji saat evaluasi
```

---

## 📊 Output yang Dihasilkan

| File | Keterangan |
|------|-----------|
| `*_watermarked.jpg` | Foto hasil watermark (disimpan ke `images/`) |
| `watermark_evaluation.png` | Grafik BER, NC, PSNR vs Quality Factor JPEG |

---

## 📐 Metode: DCT Spread Spectrum

### Alur Embedding (Penyisipan)

```
Foto asli (RGB)
    │
    ▼
Konversi ke YCbCr → ambil channel Y (luminance)
    │
    ▼
DCT 2D global pada channel Y
    │
    ▼
Pilih koefisien mid-frekuensi [N/6 : N/2]
    │
    ▼
Generate PN sequence dari SEED (kunci rahasia)
    │
    ▼
Additive embedding:
    F_wm[k] = F_host[k] + α × b × pn[k]
    (b = +1 jika bit=1, b = -1 jika bit=0)
    │
    ▼
Inverse DCT → gabung dengan Cb, Cr asli
    │
    ▼
Simpan sebagai *_watermarked.jpg
```

### Alur Extraction (Ekstraksi)

```
Foto watermarked (bisa sudah dikompres JPEG)
    │
    ▼
YCbCr → channel Y → DCT 2D
    │
    ▼
Ambil koefisien dengan indeks yang SAMA
    │
    ▼
Korelasi dengan PN sequence (SEED sama):
    corr = Σ F_received[k] × pn[k]
    │
    ▼
Keputusan: bit = 1 jika corr > 0, bit = 0 jika corr ≤ 0
    │
    ▼
Watermark terekstrak (64 bit)
```

---

## 📈 Hasil Evaluasi

Hasil eksperimen pada `Test_Project.jpg` dengan `ALPHA = 8.0`:

| QF | BER | NC | PSNR | Status |
|----|-----|----|------|--------|
| 5  | 0.3438 | +0.3125 | 23.9 dB | ✗ Gagal |
| 10 | 0.1250 | +0.7500 | 26.2 dB | ✗ Gagal |
| 15 | 0.0312 | +0.9375 | 27.4 dB | ✗ Gagal |
| 20 | 0.0156 | +0.9688 | 28.3 dB | ✓ Berhasil |
| 30 | 0.0000 | +1.0000 | 29.6 dB | ✓ Berhasil |
| ≥30 | 0.0000 | +1.0000 | ≥29.6 dB | ✓ Berhasil |

> **Kesimpulan:** Watermark **tidak dapat diekstrak** pada QF ≤ 10.  
> Mulai **QF = 20**, watermark berhasil diekstrak. Pada **QF ≥ 30**, semua bit terekstrak sempurna.

### Penjelasan Metrik

| Metrik | Kepanjangan | Ideal | Keterangan |
|--------|------------|-------|-----------|
| **BER** | Bit Error Rate | `0.0` | Proporsi bit yang salah. `< 0.10` = berhasil |
| **NC** | Normalized Correlation | `+1.0` | Kemiripan watermark. `≥ 0.80` = sangat baik |
| **PSNR** | Peak Signal-to-Noise Ratio | `∞ dB` | Kualitas visual. `≥ 30 dB` = tidak terlihat mata |

---

## 🛠️ Troubleshooting

| Error | Solusi |
|-------|--------|
| `ModuleNotFoundError: No module named 'numpy'` | Jalankan `pip install numpy opencv-python matplotlib pillow` |
| `Tidak ada file JPG ditemukan` | Pastikan foto ada di folder `images/`, bukan di `src/` |
| Watermark tidak terdeteksi | Pastikan `SEED` sama antara saat embedding dan extraction |
| Gambar output terlalu buram | Turunkan nilai `ALPHA` (misal dari 8.0 ke 5.0) |
| Watermark hancur di QF rendah | Naikkan nilai `ALPHA` (misal dari 8.0 ke 15.0) |

---

## 📚 Dependencies

| Library | Versi | Fungsi |
|---------|-------|--------|
| `numpy` | ≥ 1.21 | Operasi array & matriks |
| `opencv-python` | ≥ 4.5 | Baca/tulis gambar, DCT, kompresi JPEG |
| `matplotlib` | ≥ 3.4 | Plot grafik evaluasi |
| `pillow` | ≥ 8.0 | Utilitas gambar tambahan |

---

*Metode: DCT Spread Spectrum Watermarking | Tools: Python 3, OpenCV, NumPy*
