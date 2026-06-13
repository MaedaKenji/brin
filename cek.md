
---

### **Klasifikasi Family Pohon Berbasis Citra UAV dengan Pendekatan Deep Learning**
**Klasifikasi Family Pohon Berbasis Citra UAV dengan Pendekatan Deep Learning**
*(Tree Family Classification Based on UAV Imagery Using a Deep Learning Approach)*

**Oleh:**
Agus Fuad Mudhofar (5024 22 1021)

**Dosen Pembimbing:**
1. Dr. Eko Mulyanto Yuniarno, S.T., M.T.
2. Dr. Arief Kurniawan, S.T., M.T.

**Departemen Teknik Komputer**
Fakultas Teknologi Elektro dan Informatika Cerdas (FTEIC)
Institut Teknologi Sepuluh Nopember (ITS)

---

### **Slide 2: Latar Belakang**
* **Isu Global:** Perubahan iklim dan tingginya emisi gas rumah kaca menuntut pemantauan hutan yang lebih proaktif, akurat, dan efisien.
* **Konteks Indonesia:** Memiliki kawasan hutan tropis terluas ketiga di dunia, sehingga inventarisasi vegetasi (identifikasi komposisi pohon) sangat vital untuk mendukung:
  * Konservasi keanekaragaman hayati
  * Program reforestasi
  * Pengelolaan karbon
* **Masalah Metode Konvensional:** Pengamatan lapangan langsung oleh tenaga ahli membutuhkan waktu yang sangat lama, biaya tinggi, dan **tidak skalabel** untuk cakupan hutan yang sangat luas.

---

### **Slide 3: Solusi yang Diusulkan**
* **Penggunaan UAV (Drone):** Pengambilan citra udara beresolusi tinggi menggunakan *Unmanned Aerial Vehicle* untuk menjangkau area luas dengan cepat.
* **Pendekatan Deep Learning:** Menerapkan algoritma **YOLOv12** untuk mendeteksi dan mengklasifikasikan kanopi pohon secara otomatis dari citra udara.
* **Fokus Klasifikasi (3 Famili Dominan):**
  1. *Arecaceae* (Palem)
  2. *Fabaceae* (Polong-polongan)
  3. *Rubiaceae* (Kopi-kopian)

---

### **Slide 4: Pengumpulan Data & Lokasi**
* **Lokasi Pengambilan:** Kebun Raya Purwodadi.
* **Perangkat:** Drone DJI Phantom 4 Pro.
* **Ketinggian Terbang:** 91,4 meter.
* **Karakteristik Citra:** 191 citra RGB (*Red, Green, Blue*) beresolusi sangat tinggi (4864 × 3648 piksel).

---

### **Slide 5: Pra-pemrosesan & Augmentasi Data**
* **Anotasi Manual:** Dilakukan pelabelan kotak pembatas (*bounding box*) secara manual pada **2.365 objek kanopi**.
* **Augmentasi Data:** Untuk menghindari *overfitting* dan memperkaya variasi data pelatihan, digunakan pustaka *Albumentations*.
* **Hasil Akhir Dataset:** Jumlah objek yang siap dilatih meningkat signifikan menjadi **7.302 objek**.
* **Dimensi Input Model:** Citra diproses ke dalam resolusi 640 × 640 piksel.

---

### **Slide 6: Pelatihan Model (YOLOv12)**
* **Algoritma:** Menggunakan *State-of-the-art* dari keluarga YOLO, yaitu **YOLOv12**.
* **Siklus Pelatihan:** Model dilatih selama **300 *epoch***.
* **Skenario Pengujian:** Membandingkan berbagai varian YOLOv12 (seperti varian *n/nano* untuk efisiensi dan *l/large* untuk akurasi optimal).
* **Target Evaluasi:** Mencari titik keseimbangan (*trade-off*) terbaik antara **Akurasi (mAP)** dan **Efisiensi Komputasi (FPS)**.

---

### **Slide 7: Hasil Evaluasi Akurasi**
* **Model Terbaik (Akurasi):** Varian **YOLOv12l** (*Large*).
* **Skor mAP@0.5:** Mencapai **0,7486** (74,86%).
* **Analisis:** Model ini terbukti sangat baik dalam membedakan ciri khas visual dari ketiga famili pohon (*Arecaceae, Fabaceae, Rubiaceae*) pada lingkungan hutan yang saling tumpang tindih.
---

### **Slide 8: Hasil Evaluasi Kecepatan (Inferensi)**
* **Penerapan Edge Computing (Di Lapangan):**
  * Model yang diuji: **YOLOv12n** (*Nano*).
  * Perangkat: **Jetson Nano**.
  * Performa: Kecepatan mencapai **8,62 FPS**, membuktikan kelayakannya untuk komputasi terbatas dan pemantauan secara *real-time*.
* **Penerapan dengan Optimasi (Server/Komputer Kuat):**
  * Optimasi menggunakan: **TensorRT**.
  * Perangkat: **GPU NVIDIA RTX 4090**.
  * Performa: Kecepatan inferensi melonjak drastis hingga **108,58 FPS**.

---

### **Slide 9: Kesimpulan**
1. **Efektivitas Sistem:** Kombinasi citra UAV dan *Deep Learning* dapat secara efektif digunakan untuk mengawasi keanekaragaman hayati dan mengklasifikasikan famili pohon.
2. **Keseimbangan Model:** 
   * **YOLOv12l** direkomendasikan jika prioritas utama adalah **akurasi deteksi**.
   * **YOLOv12n** sangat ideal dan cocok diimplementasikan pada **perangkat lapangan yang terbatas** (*Edge Computing*) dengan performa *real-time*.
3. **Dampak Praktis:** Sistem ini membuka peluang bagi instansi terkait untuk melakukan inventarisasi hutan dengan cara yang jauh lebih masif, cepat, dan ekonomis dibanding survei tanah manual.

---

### **Slide 10: Penutup / Q&A**
**Terima Kasih atas Perhatiannya**
*Sesi Tanya Jawab*

---

> **💡 Tips Presentasi:** 
> Untuk melengkapi presentasi ini, Anda bisa menyisipkan beberapa **gambar tangkapan layar (screenshot) deteksi model** dari folder `Buku-TA/gambar/` pada **Slide 7** dan **Slide 8** agar audiens bisa melihat secara langsung *bounding box* hasil klasifikasi pada citra UAV.