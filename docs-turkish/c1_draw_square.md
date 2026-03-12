# C1: Cartesian Kare Çizimi — Öğrenme Notu

## Amaç

Önceki tüm modülleri (A2–B2) tek bir çalıştırılabilir script'te birleştirerek
2-link planar robotun MuJoCo ortamında **torque kontrol** ile XY düzleminde
bir kare çizmesini sağlamak.

## Zincir

```text
kare köşeleri ──► quintic Cartesian ──► analitik IK ──► Jacobian J⁻¹ ──► computed torque ──► MuJoCo sim ──► viewer trail
                  trajectory (B1)        (A4)             (A3)            M, bias (A5)
                                                                         Kp/Kd (B2)
```

Adım adım:

1. **Yol Planlama** — Karenin 4 köşesi tanımlanır. Her kenar boyunca `quintic_profile` ile
   x(t) ve y(t) ayrı ayrı enterpolasyon yapılır. Quintic profil başlangıç ve bitiş noktalarında
   hız ve ivmenin sıfır olmasını garanti eder (pürüzsüz duruş/kalkış).

2. **Ters Kinematik** — Her zaman adımında (x, y) hedefi için `analytic_ik` çağrılır.
   İki çözüm dalından (elbow-up / elbow-down) önceki çözüme en yakın olan seçilir
   (dal sürekliliği). İlk noktada elbow-down (q2 < 0) tercih edilir — bu dal kolu
   base platformundan uzak tutar.

3. **Jacobian ile Hız Eşlemesi** — İstenen Cartesian hız `[ẋ, ẏ]` Jacobian tersiyle
   joint hıza çevrilir: `q̇_des = J⁻¹ · [ẋ, ẏ]`. Singularite yakınında (det(J) < 1e-8)
   güvenli sıfır hız döndürülür — quintic profil zaten köşelerde sıfır hız verir.

4. **Joint İvme (Sayısal Türev)** — `q̈_des` merkezi sonlu farklar ile hesaplanır:
   `q̈[i] = (q̇[i+1] − q̇[i−1]) / (2·dt)`. Analitik J̇ hesabından daha basit, dt=0.002s
   ile yeterli doğruluk sağlar.

5. **Computed Torque Control** — Robotun hareket denklemi:

   ```
   M(q)·q̈ + C(q,q̇)·q̇ + g(q) = τ_applied + τ_passive
   ```

   MuJoCo terminolojisiyle:
   - `M` → `mj_fullM(model, M, data.qM)` — konfigürasyona bağlı kütle matrisi
   - `qfrc_bias` → Coriolis + gravity terimleri
   - `qfrc_passive` → joint damping (−d·q̇)
   - `ctrl` → doğrudan motor torku (N·m)

   Kontrol kuralı:
   ```
   u = q̈_des + Kp·(q_des − q) + Kd·(q̇_des − q̇)
   τ = M·u + qfrc_bias − qfrc_passive
   ```

   Bu kural tüm nonlineerliği iptal eder ve hata dinamiklerini şuna indirger:
   ```
   ë + Kd·ė + Kp·e = 0
   ```
   Kritik sönümlü seçim: `Kp = ωn² = 400`, `Kd = 2·ωn = 40` (doğal frekans 20 rad/s).

6. **MuJoCo Simülasyonu** — `mujoco.viewer.launch_passive` ile gerçek zamanlı
   görselleştirme yapılır. Her frame'de:
   - **Yeşil kare** — hedef yol (4 capsule çizgi, `user_scn` geom)
   - **Kırmızı izler** — end-effector'ün geçtiği noktalar (her 5 adımda bir küre)
   - **Sarı marker** — anlık EE konumu

## Konfigürasyon

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `SQUARE_CENTER` | (0.30, 0.30) | Karenin merkezi (m) |
| `SQUARE_SIDE` | 0.10 | Kenar uzunluğu (m) |
| `SEGMENT_DURATION` | 2.0 | Her kenar için süre (s) |
| `KP` | [400, 400] | Orantısal kazanç |
| `KD` | [40, 40] | Türevsel kazanç |
| `TORQUE_LIMIT` | 5.0 | Motor torque limiti (N·m) |
| `TRAIL_DECIMATE` | 5 | Trail noktası aralığı (adım) |

## Kazanç Tasarımı

Computed torque kontrolör ile hata dinamikleri doğrusallaştırılır:

```
ë + Kd·ė + Kp·e = 0
```

Bu ikinci dereceden sistemin karakteristik denklemi: `s² + Kd·s + Kp = 0`

- Doğal frekans: `ωn = √Kp = √400 = 20 rad/s`
- Sönüm oranı: `ζ = Kd / (2·ωn) = 40 / 40 = 1.0` → **kritik sönüm**
- Yerleşme süresi: `ts ≈ 4 / (ζ·ωn) = 0.2 s`

Kritik sönüm:
- Overshoot yok (aşma sıfır)
- Mümkün olan en hızlı salınımsız yakınsama
- Tracking hatasını minimize eder

## Keşfedilen Sorunlar ve Çözümler

### 1. MuJoCo Açı Birimi

**Sorun:** Model XML'de `range="-3.14 3.14"` yazılmıştı. MuJoCo varsayılan olarak
derece kullanır, bu yüzden bunu ±3.14° (±0.055 rad) olarak yorumladı.
Normal çalışma açılarında (±1-2 rad) joint limit constraint kuvvetleri devreye
girerek kontrolörü bozdu.

**Çözüm:** XML'e `<compiler angle="radian"/>` eklendi. Artık range doğru şekilde
±3.14 radyan.

### 2. Dekoratif Geom Çakışması

**Sorun:** Base platform ve joint visualizer geom'ları `contype=1` ile tanımlıydı.
Bazı konfigürasyonlarda (özellikle elbow-up, q2 > 1.5 rad) link1 capsule'ü
ile çakışarak büyük constraint kuvvetleri üretiyordu.

**Çözüm:** Script runtime'da tüm geom collision'ları devre dışı bırakır:
```python
model.geom_contype[:] = 0
model.geom_conaffinity[:] = 0
```

### 3. IK Dal Seçimi

**Sorun:** İlk nokta için `|q2|` en küçük olan dal seçiliyordu. Karenin köşeleri
origin'e yakın olduğunda her iki dal da `|q2| ≈ 1.92 rad` verir — pozitif dal
kolu base'e doğru katlar ve çakışma yaratır.

**Çözüm:** İlk noktada **elbow-down** (q2 < 0) tercih edilir. Bu dal kolu
workspace'in dış tarafına doğru açar.

## Doğrulama Sonuçları

Headless simülasyon (4001 adım, 8.0 saniye):

| Metrik | Değer |
|--------|-------|
| RMS Cartesian hata | 0.008 mm |
| Max Cartesian hata | 0.013 mm |
| Final Cartesian hata | 0.005 mm |
| Max uygulanan torque | 0.077 N·m |
| Torque satürasyonu | Yok |

Bu sonuçlar computed torque kontrolün etkinliğini gösterir:
- Sub-milimetre izleme doğruluğu
- Torque limitinin %2'si bile kullanılmıyor
- Pürüzsüz, salınımsız hareket

## Çıktılar

- `src/c1_draw_square.py` — ana script
- MuJoCo viewer'da gerçek zamanlı animasyon (yeşil hedef kare, kırmızı trail, sarı EE marker)

## Kullanılan Modüller

| Modül | Kullanılan Fonksiyon | Amaç |
|-------|---------------------|------|
| A2 (`a3_jacobian.py`) | `fk_endeffector` | FK doğrulama |
| A3 (`a3_jacobian.py`) | `analytic_jacobian` | J⁻¹ ile hız eşlemesi |
| A4 (`a4_inverse_kinematics.py`) | `analytic_ik` | Cartesian → joint-space |
| A5 (MuJoCo runtime) | `mj_fullM`, `data.qfrc_bias` | M(q), bias terimleri |
| B1 (`b1_trajectory_generation.py`) | `quintic_profile` | Pürüzsüz Cartesian yol |
| B2 (kontrol prensibi) | Kp/Kd hata terimleri | Computed torque içinde PD |

## Çalıştırma

```bash
cd src && python3 c1_draw_square.py
```

MuJoCo viewer penceresi açılır. Robot kolu kareyi çizerken:
- **Yeşil çizgiler** = hedef kare
- **Kırmızı noktalar** = end-effector trail
- **Sarı küre** = anlık EE konumu

Pencereyi kapatınca terminal'de özet metrikler yazdırılır.
