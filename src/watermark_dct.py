"""
=============================================================================
  DCT-Domain Spread Spectrum Watermarking + JPEG Robustness Evaluation
=============================================================================
  Teknik: Spread Spectrum Watermarking di domain DCT (mirip JPEG internals)
  Evaluasi: BER (Bit Error Rate) & NC (Normalized Correlation) vs QF JPEG
=============================================================================
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import io
import os

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────
ALPHA        = 8.0        # Kekuatan embedding (semakin besar = lebih robust tapi visible)
WATERMARK_BITS = 64       # Jumlah bit watermark (panjang pesan)
SEED         = 42         # Seed untuk PN sequence (kunci rahasia)
QF_VALUES    = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Quality Factor JPEG

# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERATE / LOAD GAMBAR HOST
# ─────────────────────────────────────────────────────────────────────────────
def create_sample_face(size=256):
    """Buat gambar wajah sintetis jika tidak ada foto asli."""
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Background gradient (langit)
    for y in range(size):
        val = int(180 + 60 * y / size)
        img[y, :] = [val, val - 20, 255 - val // 2]

    # Badan (baju)
    cv2.ellipse(img, (size//2, size - 30), (60, 50), 0, 0, 360, (70, 100, 160), -1)

    # Leher
    cv2.rectangle(img, (size//2 - 15, size//2 + 50), (size//2 + 15, size//2 + 80), (220, 180, 140), -1)

    # Wajah (kulit)
    cv2.ellipse(img, (size//2, size//2 + 10), (65, 80), 0, 0, 360, (220, 180, 140), -1)

    # Rambut
    cv2.ellipse(img, (size//2, size//2 - 30), (68, 60), 0, 180, 360, (60, 40, 20), -1)
    cv2.ellipse(img, (size//2 - 65, size//2 + 10), (12, 30), 20, 0, 360, (60, 40, 20), -1)
    cv2.ellipse(img, (size//2 + 65, size//2 + 10), (12, 30), -20, 0, 360, (60, 40, 20), -1)

    # Alis
    cv2.ellipse(img, (size//2 - 28, size//2 - 22), (18, 5), 0, 0, 360, (60, 40, 20), -1)
    cv2.ellipse(img, (size//2 + 28, size//2 - 22), (18, 5), 0, 0, 360, (60, 40, 20), -1)

    # Mata
    cv2.ellipse(img, (size//2 - 28, size//2 - 8), (16, 10), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (size//2 + 28, size//2 - 8), (16, 10), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, (size//2 - 28, size//2 - 8), 7, (60, 40, 20), -1)
    cv2.circle(img, (size//2 + 28, size//2 - 8), 7, (60, 40, 20), -1)
    cv2.circle(img, (size//2 - 26, size//2 - 10), 2, (255, 255, 255), -1)
    cv2.circle(img, (size//2 + 30, size//2 - 10), 2, (255, 255, 255), -1)

    # Hidung
    pts = np.array([[size//2, size//2 + 10],
                    [size//2 - 12, size//2 + 30],
                    [size//2 + 12, size//2 + 30]], np.int32)
    cv2.polylines(img, [pts], False, (180, 130, 100), 2)
    cv2.ellipse(img, (size//2, size//2 + 30), (8, 5), 0, 0, 360, (180, 130, 100), 1)

    # Mulut
    cv2.ellipse(img, (size//2, size//2 + 48), (22, 8), 0, 0, 180, (180, 80, 80), -1)
    cv2.line(img, (size//2 - 22, size//2 + 48), (size//2 + 22, size//2 + 48), (120, 50, 50), 2)

    # Telinga
    cv2.ellipse(img, (size//2 - 65, size//2 + 10), (10, 18), 0, 0, 360, (210, 170, 130), -1)
    cv2.ellipse(img, (size//2 + 65, size//2 + 10), (10, 18), 0, 0, 360, (210, 170, 130), -1)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def load_or_create_host(path=None, size=256):
    """Load gambar dari path atau buat sintetis."""
    if path and os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (size, size))
            print(f"[✓] Gambar host dimuat dari: {path}")
            return img
    print("[!] Foto tidak ditemukan — menggunakan wajah sintetis.")
    return create_sample_face(size)


# ─────────────────────────────────────────────────────────────────────────────
# 2. GENERATE WATERMARK
# ─────────────────────────────────────────────────────────────────────────────
def generate_watermark(n_bits=WATERMARK_BITS, mode="binary"):
    """
    Buat watermark:
      - 'binary' : bit {0,1} secara deterministik
      - 'random' : bit acak sepenuhnya
    """
    rng = np.random.default_rng(SEED + 1)
    if mode == "binary":
        # Pola biner deterministik (bisa diganti teks/ID)
        bits = np.array([int(b) for b in format(0xDEADBEEFCAFEBABE, '064b')[:n_bits]])
    else:
        bits = rng.integers(0, 2, n_bits)
    return bits.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. EMBEDDING — DCT SPREAD SPECTRUM
# ─────────────────────────────────────────────────────────────────────────────
def get_pn_sequences(n_bits, n_coeffs, seed=SEED):
    """Hasilkan n_bits PN (pseudo-noise) sequences masing-masing panjang n_coeffs."""
    rng = np.random.default_rng(seed)
    pn = rng.choice([-1.0, 1.0], size=(n_bits, n_coeffs))
    return pn


def embed_watermark(host_img, watermark_bits, alpha=ALPHA):
    """
    Embed watermark ke channel Y (luminance) menggunakan DCT global.
    Setiap bit watermark dimodulasi ke n_coeffs koefisien DCT mid-frekuensi
    melalui additive spread spectrum.

    w_i(x) = 1  →  tambah  alpha * pn_i
    w_i(x) = 0  →  tambah -alpha * pn_i
    """
    # Konversi ke YCbCr
    img_ycbcr = cv2.cvtColor(host_img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    Y = img_ycbcr[:, :, 0]

    H, W = Y.shape
    # DCT 2D pada seluruh channel Y
    Y_dct = cv2.dct(Y)

    # Pilih koefisien mid-frekuensi (hindari DC dan high-freq)
    flat_dct = Y_dct.flatten()
    n_total  = len(flat_dct)
    n_bits   = len(watermark_bits)

    # Koefisien indeks pertengahan untuk robustness terhadap JPEG
    mid_start = n_total // 6
    mid_end   = n_total // 2
    n_coeffs  = (mid_end - mid_start) // n_bits

    pn_seqs = get_pn_sequences(n_bits, n_coeffs)

    # Embedding
    for i, bit in enumerate(watermark_bits):
        polarity = 1.0 if bit >= 0.5 else -1.0
        idx_start = mid_start + i * n_coeffs
        idx_end   = idx_start + n_coeffs
        flat_dct[idx_start:idx_end] += alpha * polarity * pn_seqs[i]

    # Inverse DCT
    Y_wm = cv2.idct(flat_dct.reshape(H, W))
    Y_wm = np.clip(Y_wm, 0, 255)

    img_wm = img_ycbcr.copy()
    img_wm[:, :, 0] = Y_wm
    watermarked = cv2.cvtColor(img_wm.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    return watermarked


# ─────────────────────────────────────────────────────────────────────────────
# 4. EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_watermark(watermarked_img, n_bits=WATERMARK_BITS):
    """
    Ekstrak watermark dari gambar (mungkin sudah dikompres).
    Korelasi koefisien DCT dengan PN sequence yang sama.
    """
    img_ycbcr = cv2.cvtColor(watermarked_img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    Y = img_ycbcr[:, :, 0]
    H, W = Y.shape

    Y_dct    = cv2.dct(Y)
    flat_dct = Y_dct.flatten()
    n_total  = len(flat_dct)

    mid_start = n_total // 6
    mid_end   = n_total // 2
    n_coeffs  = (mid_end - mid_start) // n_bits

    pn_seqs = get_pn_sequences(n_bits, n_coeffs)

    extracted = np.zeros(n_bits, dtype=np.float32)
    for i in range(n_bits):
        idx_start = mid_start + i * n_coeffs
        idx_end   = idx_start + n_coeffs
        corr = np.dot(flat_dct[idx_start:idx_end], pn_seqs[i])
        extracted[i] = 1.0 if corr > 0 else 0.0

    return extracted


# ─────────────────────────────────────────────────────────────────────────────
# 5. JPEG COMPRESSION
# ─────────────────────────────────────────────────────────────────────────────
def jpeg_compress(img_rgb, quality):
    """Kompres gambar dengan JPEG pada quality factor tertentu, lalu decode."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', img_bgr, encode_params)
    decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# 6. METRIK EVALUASI
# ─────────────────────────────────────────────────────────────────────────────
def ber(original_bits, extracted_bits):
    """Bit Error Rate: proporsi bit yang salah."""
    errors = np.sum(original_bits != extracted_bits)
    return errors / len(original_bits)


def nc(original_bits, extracted_bits):
    """Normalized Correlation antara watermark asli dan yang diekstrak."""
    orig = 2 * original_bits - 1   # {0,1} → {-1,+1}
    extr = 2 * extracted_bits - 1
    num  = np.dot(orig, extr)
    den  = np.sqrt(np.dot(orig, orig) * np.dot(extr, extr) + 1e-9)
    return num / den


def psnr(img1, img2):
    """Peak Signal-to-Noise Ratio antara dua gambar."""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse < 1e-10:
        return float('inf')
    return 10 * np.log10(255 ** 2 / mse)


# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALISASI UTAMA
# ─────────────────────────────────────────────────────────────────────────────
def visualize_watermark_pattern(watermark_bits):
    """Render bit watermark sebagai gambar 8×8."""
    n = int(np.sqrt(len(watermark_bits)))
    wm_img = (watermark_bits[:n*n].reshape(n, n) * 255).astype(np.uint8)
    return cv2.resize(wm_img, (128, 128), interpolation=cv2.INTER_NEAREST)


def run_full_evaluation(host_path=None, watermark_mode="binary"):
    print("=" * 60)
    print("  DCT Spread Spectrum Watermarking — Evaluasi Robustness JPEG")
    print("=" * 60)

    # ── Load gambar host ──
    host_img   = load_or_create_host(host_path, size=256)
    wm_bits    = generate_watermark(WATERMARK_BITS, mode=watermark_mode)
    wm_display = visualize_watermark_pattern(wm_bits)

    print(f"[✓] Watermark ({watermark_mode}): {WATERMARK_BITS} bit")
    print(f"    Bits: {wm_bits.astype(int)}")

    # ── Embedding ──
    wm_img = embed_watermark(host_img, wm_bits, alpha=ALPHA)
    psnr_embed = psnr(host_img, wm_img)
    print(f"\n[✓] Watermark embedded — PSNR host vs watermarked: {psnr_embed:.2f} dB")

    # ── Ekstraksi tanpa kompresi ──
    ex0     = extract_watermark(wm_img, WATERMARK_BITS)
    ber0    = ber(wm_bits, ex0)
    nc0     = nc(wm_bits, ex0)
    print(f"[✓] Tanpa kompresi — BER: {ber0:.4f}  NC: {nc0:.4f}")

    # ── Evaluasi per QF ──
    results = []
    compressed_samples = {}
    for qf in QF_VALUES:
        comp = jpeg_compress(wm_img, qf)
        ex   = extract_watermark(comp, WATERMARK_BITS)
        b    = ber(wm_bits, ex)
        n_   = nc(wm_bits, ex)
        p    = psnr(wm_img, comp)
        extractable = b < 0.1   # threshold: BER < 10% dianggap berhasil
        results.append({
            "qf": qf, "ber": b, "nc": n_, "psnr": p,
            "extractable": extractable, "extracted": ex
        })
        if qf in [5, 15, 30, 60, 90]:
            compressed_samples[qf] = comp
        status = "✓ OK" if extractable else "✗ GAGAL"
        print(f"  QF={qf:3d} | BER={b:.4f} | NC={n_:+.4f} | PSNR={p:.1f} dB | {status}")

    # ── Temukan threshold QF ──
    fail_qfs = [r["qf"] for r in results if not r["extractable"]]
    if fail_qfs:
        print(f"\n[!] Watermark TIDAK dapat diekstrak (BER ≥ 0.10) pada QF: {fail_qfs}")
        print(f"    → QF ≥ {min([r['qf'] for r in results if r['extractable']], default='N/A')} "
              f"watermark masih dapat diekstrak")
    else:
        print("\n[✓] Watermark dapat diekstrak di semua nilai QF yang diuji!")

    # ─────────────────────────────────────────────────
    # PLOT BESAR
    # ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 22), facecolor='#0f1117')
    plt.rcParams.update({
        'text.color': 'white', 'axes.labelcolor': 'white',
        'xtick.color': 'white', 'ytick.color': 'white',
        'axes.edgecolor': '#444', 'figure.facecolor': '#0f1117',
    })

    gs = gridspec.GridSpec(4, 5, figure=fig,
                           hspace=0.45, wspace=0.35,
                           top=0.96, bottom=0.04, left=0.06, right=0.97)

    ACCENT  = '#00d4ff'
    GREEN   = '#00ff88'
    RED     = '#ff4444'
    YELLOW  = '#ffd700'
    GRAY    = '#aaaaaa'

    # ── Baris 1: Gambar utama ──────────────────────────────────────────────
    ax_host = fig.add_subplot(gs[0, 0])
    ax_host.imshow(host_img)
    ax_host.set_title('Gambar Host', color=ACCENT, fontsize=10, fontweight='bold')
    ax_host.axis('off')

    ax_wm = fig.add_subplot(gs[0, 1])
    ax_wm.imshow(wm_display, cmap='gray', vmin=0, vmax=255)
    ax_wm.set_title(f'Watermark ({watermark_mode})\n{WATERMARK_BITS} bit', color=YELLOW, fontsize=10, fontweight='bold')
    ax_wm.axis('off')

    ax_wimg = fig.add_subplot(gs[0, 2])
    ax_wimg.imshow(wm_img)
    ax_wimg.set_title(f'Setelah Watermark\nPSNR={psnr_embed:.1f} dB', color=GREEN, fontsize=10, fontweight='bold')
    ax_wimg.axis('off')

    # Diff (diperkuat ×10)
    diff = np.clip((wm_img.astype(int) - host_img.astype(int) + 128), 0, 255).astype(np.uint8)
    ax_diff = fig.add_subplot(gs[0, 3])
    ax_diff.imshow(diff)
    ax_diff.set_title('Perbedaan (×amplified)\nHost vs Watermarked', color=GRAY, fontsize=10)
    ax_diff.axis('off')

    # Histogram perbedaan
    ax_hist = fig.add_subplot(gs[0, 4])
    diff_flat = (wm_img.astype(int) - host_img.astype(int)).flatten()
    ax_hist.hist(diff_flat, bins=50, color=ACCENT, alpha=0.8, edgecolor='none')
    ax_hist.set_title('Distribusi Noise\nWatermark', color=ACCENT, fontsize=10)
    ax_hist.set_xlabel('Δ Pixel', color=GRAY, fontsize=8)
    ax_hist.set_ylabel('Frekuensi', color=GRAY, fontsize=8)
    ax_hist.set_facecolor('#1a1d27')
    ax_hist.tick_params(labelsize=7)

    # ── Baris 2: Gambar hasil kompresi ─────────────────────────────────────
    sample_qfs = [5, 15, 30, 60, 90]
    for idx, sqf in enumerate(sample_qfs):
        ax = fig.add_subplot(gs[1, idx])
        ax.imshow(compressed_samples[sqf])
        r = next(x for x in results if x["qf"] == sqf)
        status_color = GREEN if r["extractable"] else RED
        status_sym   = "✓" if r["extractable"] else "✗"
        ax.set_title(f'QF={sqf}  {status_sym}\nBER={r["ber"]:.3f}  NC={r["nc"]:+.3f}',
                     color=status_color, fontsize=9, fontweight='bold')
        ax.axis('off')

    # ── Baris 3: Grafik BER & NC vs QF ────────────────────────────────────
    qf_list  = [r["qf"]  for r in results]
    ber_list = [r["ber"] for r in results]
    nc_list  = [r["nc"]  for r in results]
    psnr_list= [r["psnr"]for r in results]

    # BER vs QF
    ax_ber = fig.add_subplot(gs[2, :3])
    ax_ber.set_facecolor('#1a1d27')
    colors_ber = [RED if b >= 0.1 else GREEN for b in ber_list]
    bars = ax_ber.bar(qf_list, ber_list, color=colors_ber, alpha=0.85, width=6, zorder=3)
    ax_ber.plot(qf_list, ber_list, 'o-', color=ACCENT, linewidth=2, markersize=6, zorder=4)
    ax_ber.axhline(0.1, color=RED, linestyle='--', linewidth=1.5, label='Threshold BER=0.10', zorder=5)
    ax_ber.axhline(0.0, color=GREEN, linestyle=':', linewidth=1, alpha=0.5)
    # Anotasi
    for qf, b in zip(qf_list, ber_list):
        ax_ber.text(qf, b + 0.003, f'{b:.3f}', ha='center', va='bottom',
                    fontsize=7, color='white')
    ax_ber.set_title('Bit Error Rate (BER) vs Quality Factor JPEG\n'
                     '[ Hijau = dapat diekstrak | Merah = gagal ]',
                     color='white', fontsize=11, fontweight='bold')
    ax_ber.set_xlabel('Quality Factor (QF)', color=GRAY)
    ax_ber.set_ylabel('BER (↓ lebih baik)', color=GRAY)
    ax_ber.set_xticks(qf_list)
    ax_ber.set_ylim(-0.02, max(ber_list) + 0.08)
    ax_ber.legend(facecolor='#2a2d3a', labelcolor='white', fontsize=9)
    ax_ber.grid(True, alpha=0.15, axis='y')
    ax_ber.tick_params(labelsize=8)

    # NC vs QF
    ax_nc = fig.add_subplot(gs[2, 3:])
    ax_nc.set_facecolor('#1a1d27')
    colors_nc = [GREEN if n >= 0.8 else (YELLOW if n >= 0.5 else RED) for n in nc_list]
    ax_nc.bar(qf_list, nc_list, color=colors_nc, alpha=0.85, width=6, zorder=3)
    ax_nc.plot(qf_list, nc_list, 's-', color=YELLOW, linewidth=2, markersize=6, zorder=4)
    ax_nc.axhline(0.8, color=GREEN, linestyle='--', linewidth=1.5, label='NC=0.80 (threshold OK)', zorder=5)
    ax_nc.axhline(0.5, color=YELLOW, linestyle=':', linewidth=1.5, label='NC=0.50 (batas bawah)', zorder=5)
    for qf, n_ in zip(qf_list, nc_list):
        ax_nc.text(qf, n_ + 0.01, f'{n_:.3f}', ha='center', va='bottom',
                   fontsize=7, color='white')
    ax_nc.set_title('Normalized Correlation (NC) vs QF\n'
                    '[ Hijau ≥0.8 | Kuning ≥0.5 | Merah <0.5 ]',
                    color='white', fontsize=11, fontweight='bold')
    ax_nc.set_xlabel('Quality Factor (QF)', color=GRAY)
    ax_nc.set_ylabel('NC (↑ lebih baik)', color=GRAY)
    ax_nc.set_xticks(qf_list)
    ax_nc.set_ylim(-0.1, 1.15)
    ax_nc.legend(facecolor='#2a2d3a', labelcolor='white', fontsize=8)
    ax_nc.grid(True, alpha=0.15, axis='y')
    ax_nc.tick_params(labelsize=8)

    # ── Baris 4: PSNR vs QF + Bit pattern comparison ──────────────────────
    ax_psnr = fig.add_subplot(gs[3, :3])
    ax_psnr.set_facecolor('#1a1d27')
    ax_psnr.plot(qf_list, psnr_list, 'D-', color='#ff9944', linewidth=2,
                 markersize=7, markerfacecolor='white', zorder=4)
    ax_psnr.fill_between(qf_list, psnr_list, alpha=0.2, color='#ff9944')
    ax_psnr.axhline(30, color=GREEN, linestyle='--', linewidth=1.5,
                    label='PSNR=30 dB (kualitas cukup)')
    for qf, p in zip(qf_list, psnr_list):
        ax_psnr.text(qf, p + 0.5, f'{p:.1f}', ha='center', fontsize=7, color='white')
    ax_psnr.set_title('PSNR (Kualitas Visual) Setelah Kompresi JPEG',
                      color='white', fontsize=11, fontweight='bold')
    ax_psnr.set_xlabel('Quality Factor (QF)', color=GRAY)
    ax_psnr.set_ylabel('PSNR (dB) — ↑ lebih baik', color=GRAY)
    ax_psnr.set_xticks(qf_list)
    ax_psnr.legend(facecolor='#2a2d3a', labelcolor='white', fontsize=9)
    ax_psnr.grid(True, alpha=0.15, axis='y')
    ax_psnr.tick_params(labelsize=8)

    # Bit pattern: asli vs 3 QF
    ax_bits = fig.add_subplot(gs[3, 3:])
    ax_bits.set_facecolor('#1a1d27')
    show_bits = min(32, WATERMARK_BITS)
    y_labels = ['Original']
    bit_matrix = [wm_bits[:show_bits]]

    highlight_qfs = [r["qf"] for r in results if r["qf"] in [5, 15, 30, 60, 90]][:4]
    for hqf in highlight_qfs:
        r = next(x for x in results if x["qf"] == hqf)
        bit_matrix.append(r["extracted"][:show_bits])
        y_labels.append(f'QF={hqf}')

    bm = np.array(bit_matrix)
    ax_bits.imshow(bm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1, interpolation='nearest')
    ax_bits.set_title(f'Perbandingan Bit Watermark (first {show_bits} bit)',
                      color='white', fontsize=10, fontweight='bold')
    ax_bits.set_yticks(range(len(y_labels)))
    ax_bits.set_yticklabels(y_labels, fontsize=8)
    ax_bits.set_xlabel('Indeks Bit', color=GRAY, fontsize=8)
    ax_bits.tick_params(axis='x', labelsize=7)

    # ── Judul utama ────────────────────────────────────────────────────────
    fig.text(0.5, 0.985,
             f'DCT Spread Spectrum Watermarking — Robustness vs JPEG Compression  '
             f'(α={ALPHA}, {WATERMARK_BITS} bits, mode={watermark_mode})',
             ha='center', va='top', fontsize=13, fontweight='bold', color='white')

    # ── Tabel ringkasan ────────────────────────────────────────────────────
    summary_text = (
        "RINGKASAN\n"
        f"{'QF':>5} │ {'BER':>7} │ {'NC':>7} │ {'PSNR':>8} │ Status\n"
        + "─" * 47 + "\n"
    )
    for r in results:
        status = "✓ Berhasil" if r["extractable"] else "✗ Gagal   "
        summary_text += (
            f"{r['qf']:5d} │ {r['ber']:7.4f} │ {r['nc']:+7.4f} │ "
            f"{r['psnr']:7.1f}dB │ {status}\n"
        )
    fig.text(0.015, 0.285, summary_text,
             fontsize=7.5, color='#cccccc',
             fontfamily='monospace',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#1a1d27',
                       edgecolor='#444', alpha=0.9))

    # Simpan grafik ke folder images/ (satu level di atas src/)
    src_dir    = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(src_dir, "..", "images")
    images_dir = os.path.normpath(images_dir)
    out_graph  = os.path.join(images_dir, "watermark_evaluation.png")

    plt.savefig(out_graph, dpi=150, bbox_inches='tight',
                facecolor='#0f1117', edgecolor='none')
    plt.close()
    print(f"\n[✓] Grafik evaluasi disimpan : {out_graph}")
    return results, out_graph, wm_img


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import glob

    # ── Tentukan path folder ───────────────────────────────────────────────
    src_dir    = os.path.dirname(os.path.abspath(__file__))   # .../src/
    images_dir = os.path.normpath(os.path.join(src_dir, "..", "images"))  # .../images/

    # Mode watermark: "binary" atau "random"  (argumen opsional)
    mode = sys.argv[1] if len(sys.argv) > 1 else "binary"

    print("=" * 60)
    print("  DCT Spread Spectrum Watermarking")
    print(f"  Folder input/output : {images_dir}")
    print(f"  Mode watermark      : {mode}")
    print("=" * 60)

    # ── Cari semua file .jpg/.jpeg di folder images/ ───────────────────────
    jpg_files = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.jpeg")) +
        glob.glob(os.path.join(images_dir, "*.JPG")) +
        glob.glob(os.path.join(images_dir, "*.JPEG"))
    )

    # Abaikan file yang sudah merupakan hasil watermark (suffix _watermarked)
    jpg_files = [f for f in jpg_files if not os.path.basename(f).endswith("_watermarked.jpg")]

    if not jpg_files:
        print(f"[!] Tidak ada file JPG ditemukan di: {images_dir}")
        print("    Pastikan folder images/ berisi file .jpg")
        sys.exit(1)

    print(f"\n[✓] Ditemukan {len(jpg_files)} file JPG:\n")
    for f in jpg_files:
        print(f"    • {os.path.basename(f)}")

    # ── Proses setiap file JPG ─────────────────────────────────────────────
    print()
    for host_path in jpg_files:
        base_name = os.path.splitext(os.path.basename(host_path))[0]
        out_jpg   = os.path.join(images_dir, f"{base_name}_watermarked.jpg")

        print(f"\n[>] Memproses : {os.path.basename(host_path)}")
        print(f"    Output    : {os.path.basename(out_jpg)}")

        try:
            results, out_graph, wm_img = run_full_evaluation(
                host_path=host_path,
                watermark_mode=mode
            )

            # Simpan gambar hasil watermark ke folder images/
            wm_bgr = cv2.cvtColor(wm_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_jpg, wm_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            print(f"[✓] Tersimpan : {out_jpg}")

        except Exception as e:
            print(f"[✗] Gagal memproses {os.path.basename(host_path)}: {e}")

    print("\n" + "=" * 60)
    print(f"  Selesai! Hasil tersimpan di folder:")
    print(f"  {images_dir}")
    print("=" * 60)