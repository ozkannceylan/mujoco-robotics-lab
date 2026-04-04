# Lab 7: Yurume Temelleri — Mimari

## 1. Sistem Genel Bakisi

Lab 7, sabit tabanlı manipulasyondan (Lab 1–6, UR5e) yüzen tabanlı yürümeye geçiş yapar. Platform: **Unitree G1 insansı robot** (29 aktüatörlü serbestlik derecesi, 33.34 kg). Lab, iki ayaklı yürümenin temel yapı taşlarını keşfeder: denge kontrolü, kütle merkezi takibi, tüm vücut ters kinematik ve ZMP tabanlı yürüme.

### Platform

| Özellik | Değer |
|---------|-------|
| Robot | Unitree G1 (MuJoCo Menagerie) |
| Toplam serbestlik derecesi (nq) | 36 (7 serbest eklem + 29 menteşe) |
| Hız serbestlik derecesi (nv) | 35 (6 taban hızı + 29 eklem hızı) |
| Aktüatörlü eklemler (nu) | 29 (12 bacak + 3 bel + 14 kol) |
| Aktüatör tipi | Konum servoları (Kp=500, Kd=0) |
| Toplam kütle | 33.34 kg |
| Ayakta pelvis yüksekliği | 0.79 m (anahtar kare), 0.757 m (oturmuş) |
| Ayakta KM yüksekliği | ~0.66 m |
| Simülasyon zaman adımı | 0.002 s |

### Mimari Katmanları

```
┌──────────────────────────────────────────────────────────┐
│  Planlayıcı Katmanı                                     │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ ZMP Referans   │→│ LIPM Önizleme│→│ Salınım Ayak │  │
│  │ Üreteci        │  │ Kontrolcüsü  │  │ Yörüngesi    │  │
│  └───────────────┘  └──────────────┘  └──────────────┘  │
├──────────────────────────────────────────────────────────┤
│  Ters Kinematik Katmanı                                  │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Tüm Vücut IK (Pinocchio)                        │    │
│  │ Görevler: ayaklar 6B×2 + KM XY + pelvis Z + R   │    │
│  │ Yöntem: Yığılmış Jacobian + DLS                  │    │
│  └──────────────────────────────────────────────────┘    │
├──────────────────────────────────────────────────────────┤
│  Kontrol Katmanı                                         │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Yerçekimi Kompanzasyonlu PD                      │    │
│  │ ctrl = q_hedef + qfrc_bias/Kp - K_hız*qvel/Kp   │    │
│  └──────────────────────────────────────────────────┘    │
├──────────────────────────────────────────────────────────┤
│  Simülasyon Katmanı                                      │
│  ┌────────────────────┐  ┌───────────────────────────┐   │
│  │ MuJoCo Motoru      │  │ Pinocchio Analitiği      │   │
│  │ (fizik, temas,     │  │ (FK, Jacobian, KM,       │   │
│  │  görüntüleme)      │  │  IK, çerçeve konumları)  │   │
│  └────────────────────┘  └───────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

### Çalışan ve Çalışmayan

| Yetenek | Durum | Kilometre Taşı |
|---------|-------|----------------|
| Yerçekimi kompanzasyonlu ayakta durma (10s) | BAŞARILI | M1 |
| 5N itme kurtarma | BAŞARILI | M1 |
| KM çapraz doğrulama (Pin vs MuJoCo) | 0.000mm hata | M2 |
| Destek poligonu hesaplama | BAŞARILI | M2 |
| Pinocchio Jacobian doğrulaması (sonlu fark) | 0/36 başarısızlık | M3a |
| FK çapraz doğrulama (10 rastgele konfigürasyon) | 0.000mm hata | M3b |
| Tüm vücut IK (KM kaydırma + ayak kısıtı) | <0.51mm ayak kayması | M3c |
| Simülasyonda ağırlık kaydırma (5cm yanal) | 53.5mm kayma, <1.4mm sapma | M3d |
| Dinamik yürüme (tek adım veya çoklu adım) | BAŞARISIZ | M3e |

---

## 2. Yüzen Taban Robotiği Teorisi

### Neden nq != nv

Sabit tabanlı robotlarda (Lab 1–6) `nq = nv`'dir çünkü her eklem tek bir açı ve tek bir hız kullanır. Yüzen tabanlı robotlar bu simetriyi **kuaterniyon temsili** nedeniyle bozar.

G1'in serbest eklemi katkıda bulunur:
- **qpos'a 7 eleman**: 3 konum (x, y, z) + 4 kuaterniyon (w, x, y, z)
- **qvel'e 6 eleman**: 3 doğrusal hız + 3 açısal hız

Kuaterniyonun 4 bileşeni vardır ancak sadece 3 serbestlik derecesi (birim küre S3 üzerinde). Açısal hız SO(3)'ün tanjant uzayında yaşar ve bu 3 boyutludur. Bu uyumsuzluğu yaratır: `nq = 36` ama `nv = 35`.

**Kritik sonuç:** Konfigürasyonları basit toplama ile güncelleyemezsiniz (`q += dq`). Bunun yerine `pin.integrate(model, q, dq)` kullanmalısınız — bu, hız tanjant vektörünü konfigürasyon manifolduna uygular ve üstel harita ile S3 üzerindeki kuaterniyonu günceller.

### MuJoCo vs Pinocchio Konvansiyonları

| Konvansiyon | MuJoCo | Pinocchio |
|------------|--------|-----------|
| Kuaterniyon sırası | (w, x, y, z) | (x, y, z, w) |
| Taban Z'si qpos'ta | Dünya konumu | Dünya konumu - 0.793 |
| Hız uzayı | Aynı | Aynı |
| Eklem sıralaması | MJCF ile aynı | MJCF ile aynı |

Z ofseti, G1'in MJCF'inin `<body name="pelvis" pos="0 0 0.793">` tanımlamasından kaynaklanır. Pinocchio'nun `buildModelFromMJCF` fonksiyonu `FreeFlyer` ile bu ofseti eklem yerleşimine dahil eder: `pelvis_dünya_z = pin_q[2] + 0.793`.

Dönüşüm fonksiyonları (`lab7_common.py`):
- `mj_qpos_to_pin()`: kuaterniyonu değiştirir ve Z ofsetini çıkarır
- `pin_q_to_mj()`: ters işlem
- `mj_quat_to_pin()` / `pin_quat_to_mj()`: sadece kuaterniyon dönüşümü

### Konum Aktüatör Modeli

Menagerie G1 konum aktüatörleri kullanır:

```
tau = Kp * (ctrl - qpos) - Kd * qvel
```

Menagerie varsayılanları: Kp=500, Kd=0 (sıfır `dof_damping` ile `dampratio=1`'den).

**Sıfır Kd kritik bir tasarım kısıtıdır.** Sönümleme olmadan aktüatörler yetersiz sönümlenmiş osilatörlerdir. Lab 7 açık hız sönümlemesi ekler:

```
ctrl = q_hedef + qfrc_bias[6+i] / Kp - K_HIZ * qvel[6+i] / Kp
```

Bu efektif tork verir:
```
tau = Kp*(q_hedef - qpos) + qfrc_bias - K_HIZ*qvel
```

K_HIZ=40 ile: sönüm oranı zeta = 40 / (2*sqrt(500)) = 0.89 (kritik yakını).

---

## 3. Pinocchio Analitik Boru Hattı

Pinocchio tüm analitik hesaplamaları yapar. MuJoCo simülasyonu yürütür.

### FK ve Jacobian Hesaplaması

```python
pin.computeJointJacobians(model, data, q)
pin.updateFramePlacements(model, data)
pin.centerOfMass(model, data, q)

# Ayak Jacobianları (6×nv, LOCAL_WORLD_ALIGNED)
J_sol = pin.getFrameJacobian(model, data, sol_id, pin.LOCAL_WORLD_ALIGNED)
J_sag = pin.getFrameJacobian(model, data, sag_id, pin.LOCAL_WORLD_ALIGNED)

# KM Jacobianı (3×nv)
J_km = pin.jacobianCenterOfMass(model, data, q)
```

**LOCAL_WORLD_ALIGNED** zorunludur. Bu, dünya çerçevesinde ifade edilen ancak çerçeve orijinindeki hızları verir — yığılmış Jacobian ile görev uzayı IK için doğru seçim.

### Tüm Vücut Ters Kinematik

IK çözücü (`m3c_static_ik.py`) sönümlenmiş en küçük kareler (DLS) ile yığılmış Jacobian kullanır:

**Görev yığını (serbest taban için 18 boyut, sabit taban için 14):**
1. Sol ayak konum + yönelim (6B)
2. Sağ ayak konum + yönelim (6B)
3. KM XY konumu (2B)
4. Pelvis yükseklik Z (1B) — sadece serbest taban
5. Pelvis yönelimi (3B) — sadece serbest taban

**DLS çözümü:**
```
J_yigin = [J_sol; J_sag; J_km_xy; J_pz; J_pyon]   (18×35)
dx_yigin = [dx_sol; dx_sag; dx_km_xy; dx_pz; dx_pyon]
dq = J^T (J J^T + lambda^2 I)^{-1} dx
q_yeni = pin.integrate(model, q, alpha * dq)
```

Parametreler: lambda=0.01, alpha=0.3, dq_maks=0.1 rad/iterasyon.

**Önemli içgörü:** Pelvis Jacobianı sadece taban serbestlik dereceleri (indeks 0–5) için sıfırdan farklıdır. Bu, pelvis görevlerinin yüzen tabanı kısıtlarken bacak eklemlerinin ayak/KM görevleri için serbest kalması anlamına gelir — görevler yarışmak yerine işbirliği yapar.

### Çapraz Doğrulama Sonuçları

Tüm Pinocchio-MuJoCo karşılaştırmaları makine hassasiyetinde uyum gösterir:

| Miktar | Yöntem | Maks Hata |
|--------|--------|-----------|
| Ayak FK (10 rastgele konfigürasyon) | Konum karşılaştırma | 0.000 mm |
| KM konumu | subtree_com vs pin.com | 0.000 mm |
| KM Jacobian sütunları | Merkezi sonlu farklar | 1.03e-08 |
| Ayak Jacobian sütunları | Merkezi sonlu farklar | 1.09e-07 |

---

## 4. LIPM ve ZMP Önizleme Kontrolü Teorisi

### Doğrusal Ters Sarkaç Modeli

LIPM, robotu sabit yükseklik z_c'de bir noktasal kütle olarak modeller ve yerle kütlesiz bir bacakla bağlar. Sabit yükseklik kısıtı altında:

```
x_iki_nokta = (g / z_c) * (x_km - x_zmp)
```

**Sıfır Moment Noktası (ZMP)**, yerçekimi artı atalet kuvvetlerinin net momentinin sıfır olduğu noktadır. Tek destek sırasında ZMP, basma ayağı içinde olmalıdır. Çift destek sırasında her iki ayağın dışbükey örtüsünde herhangi bir yerde olabilir.

LIPM, bağımsız X ve Y dinamiklerine ayrışır ve problemi çözülebilir kılar.

### Önizleme Kontrolü (Kajita 2003)

Araba-masa modeli, sarsıntı (üçüncü türev) girdi olarak ayrıklaştırılır:

```
Durum: [x, x_nokta, x_iki_nokta]
A = [[1, dt, dt^2/2], [0, 1, dt], [0, 0, 1]]
B = [[dt^3/6], [dt^2/2], [dt]]
C = [1, 0, -z_c/g]   (ZMP çıktısı: p = x - (z_c/g)*x_iki_nokta)
```

Sistem, ZMP takip hatası üzerinde bir integratör ile genişletilir, ardından ayrık Riccati denklemi çözülür:

- **Ki**: kümülatif ZMP hatasında integral kazancı
- **Gx**: durum geri besleme kazancı [1×3]
- **Gd[j]**: gelecek ZMP referansları için önizleme kazançları

Kontrol yasası 1.6s ileriye bakar (80 önizleme adımı × 0.02s):
```
u(k) = -Ki * toplam(e) - Gx @ durum(k) + toplam(Gd[j] * zmp_ref(k+j))
```

**Kritik özellik:** Önizleme kontrolü, destek geçişinden ÖNCE kütle merkezini hareket ettirmeye başlar. KM, ZMP geçişini "öncüler" — bu dinamik yürüme için temeldir.

### Adım Planı ve ZMP Referansı

Adım planlayıcı, yapılandırılabilir adım uzunluğu ve zamanlama ile değişen sol/sağ adımlar üretir:

```
Zaman çizelgesi:
  [0, t_baslangic)                — Başlangıç ÇD: ZMP orta noktadan ilk basma noktasına
  Her adım i için:
    [t_bas, t_bas+t_td)           — TD: ZMP basma ayağı merkezinde
    [t_td, t_td+t_cd)             — ÇD: ZMP sonraki basma noktasına enterpolasyon
  Son TD + Son ÇD                 — ZMP orta noktaya döner
```

Zamanlama: t_td=0.8s (tek destek), t_cd=0.2s (çift destek), t_bas=t_son=1.0s.

### Salınım Ayağı Yörüngesi

Tek destek sırasında salınım ayağı izler:
- **Yatay (X, Y):** Kübik düzgün enterpolasyon: `3s^2 - 2s^3`
- **Dikey (Z):** Parabolik tümsek: `4 * adim_yukseklik * s * (1-s)`, tepe s=0.5'te

Bu, başlangıç ve bitişte sıfır hız verir (düzgün kaldırma ve iniş).

---

## 5. Yürüme Başarısızlıkları — Dürüst Analiz

### Ne Denedik

Tüm LIPM + ZMP + IK boru hattını başarıyla uyguladıktan (tüm modüller bağımsız olarak doğrulandıktan) sonra, MuJoCo simülasyonunda tek bir ileri adım yürütmeyi denedik. Altı farklı yaklaşım denendi:

| Deneme | Anahtar Değişiklik | Sonuç | Düşme Zamanı |
|--------|-------------------|-------|--------------|
| 1 | Temel IK + PD (Kp=400, t_td=1.5s) | DÜŞTÜ | 3.64s |
| 2 | +İleri besleme + bilek geri beslemesi | DÜŞTÜ | 3.79s |
| 3 | Yüksek kazançlar (Kp=800), kısa TD (0.5s) | İlk TD geçti, hizalamada düştü | 3.78s |
| 4 | Hizalama adımını atla | DÜŞTÜ | 4.04s |
| 5 | Özel manuel ZMP referansı | DÜŞTÜ (daha kötü) | 2.84s |
| 6 | Ölçülen taban durumu ile çevrimiçi IK | DÜŞTÜ | 3.04s |

### Kök Neden Analizi

Başarısızlık tek bir bileşende değil — her modül (LIPM planlayıcı, ZMP referansı, IK çözücü, salınım yörüngesi) bağımsız olarak doğru çalışıyor. Başarısızlık, kinematik planlama ile dinamik simülasyon arasındaki **yürütme açığında** yatıyor.

**Temel problem: Konum aktüatörleri ZMP kontrolü sağlayamaz.**

ZMP tabanlı yürüme, yer tepki kuvvetlerinin — özellikle net momentin sıfır olduğu noktanın — hassas kontrolünü gerektirir. Bu, **bilek torkları** kontrolünü gerektirir — basma ayağı içinde ZMP'yi kaydırmanın birincil mekanizması.

Konum aktüatörleri (`tau = Kp*(q_ref - q) - Kd*qvel`) eklem açılarını takip eder, torkları değil. Robot tek desteğe girdiğinde:

1. IK planlayıcı, mükemmel takip varsayarak eklem açıları hesaplar
2. Konum aktüatörleri bu açıları biraz gecikme ve hata ile takip eder
3. Takip hataları istenmeyen torklar yaratır, özellikle bileklerde
4. İstenmeyen bilek torkları gerçek ZMP'yi planlanan ZMP'den uzaklaştırır
5. KM yanlış yönde ivmelenir (LIPM: `x_iki_nokta ∝ x_km - x_zmp`)
6. Hata birleşir — 0.5–1.5s içinde robot düşer

Bu **yapısal bir sınırlamadır**, ayar problemi değil. Hiçbir kazanç ayarı düzeltemez çünkü kontrol mimarisi doğrudan bilek torklarını komut edemez.

### Ne Düzeltirdi

1. **Tork kontrollü aktüatörler**: Konum servolarını doğrudan tork komutları ile değiştirmek. Farklı MuJoCo aktüatör modeli ve tüm vücut dinamik kontrolcüsü (ters dinamik, QP tabanlı) gerektirir.

2. **Pekiştirmeli öğrenme**: Durumdan konum aktüatör komutlarına eşleyen bir politika eğitmek. PÖ, aktüatör dinamiklerini telafi ederek istenen ZMP'den konum komutlarına eşlemeyi örtük olarak öğrenebilir.

3. **Model Öngörülü Kontrol (MPC)**: Aktüatör dinamiklerini ileriye tahmin etmek ve istenen tork profilini elde eden konum komutlarını optimize etmek.

4. **Sim-gerçek çerçeveleri (ör. Isaac Lab)**: GPU-paralel ortamlar kullanarak sağlam yürüme politikaları eğitmek. İnsansı yürüme için güncel endüstri standardı.

### Olumlu Tarafı

Ağırlık kaydırmaya kadar ve dahil her şey (M3d) güvenilir şekilde çalışır:
- Yerçekimi kompanzasyonlu ayakta durma çok kararlı (10s'de 1.6mm sapma)
- 5N yanal kuvvetten itme kurtarma çalışır
- Pinocchio IK, rastgele KM hedefleri için doğru eklem konfigürasyonları üretir
- Önceden hesaplanmış IK yörüngeleri simülasyonda düzgün yürütülür (53.5mm KM kayması, <1.4mm ayak sapması)

Başarısızlık, robotun tek ayak üzerinde dengelenmesi gerektiğinde — statikten dinamik dengeye geçişte — meydana gelir. Bu, iki ayaklı yürümenin temel zorluğudur.

---

## 6. Öğrenilen Dersler

### Ders 1: Pinocchio-MuJoCo Ayrımı Güçlüdür

Tüm analitik hesaplamalar için Pinocchio, fizik simülasyonu için MuJoCo kullanmak doğru mimaridir. Doğru kuaterniyon ve Z-ofseti dönüşümü ile iki motor makine hassasiyetinde uyuşur. Bu, kinematik boru hattına tam güven verdi ve dinamik yürütmeyi başarısızlık modu olarak izole etmemize olanak tanıdı.

### Ders 2: Önceden Hesaplanmış IK >> Çevrimiçi IK (Konum Aktüatörleri İçin)

Çevrimiçi IK geri beslemesine yapılan beş deneme (ölçülen duruma göre her zaman adımında IK hedeflerini güncelleme) başarısız oldu. Kinematik Jacobian, MuJoCo dinamiklerinin ürettiğinden farklı KM-eklem duyarlılığı tahmin ediyor çünkü servo tork reaksiyonu pelviste kinematik zincir tahminine baskın çıkıyor.

Tüm IK yörüngesini N ara noktada çevrimdışı önceden hesaplayıp PD takibi ile oynatmak çok daha sağlamdır.

### Ders 3: Hız Sönümlemesi En Önemli Parametredir

Menagerie G1 aktüatörleri Kd=0'dır. Açık hız sönümlemesi olmadan (K_HIZ=40, zeta=0.89 veren), robot salınım yapar ve düşer. Bu tek parametre değişikliği — `ctrl -= K_HIZ * qvel / Kp` ekleme — ağırlık kaydırma için başarı ile başarısızlık arasındaki fark oldu.

### Ders 4: Jacobianları Merkezi Sonlu Farklarla Doğrulayın

İleri farklar (O(eps)) büyük kaldıraç kollu eklemler için yanlış alarm verdi. Merkezi farklar (O(eps^2)) hataları 1e-7'ye indirdi ve Jacobianların doğru olduğunu onayladı. Bu doğrulama, herhangi bir yeni yüzen taban modeli için ilk adım olmalıdır.

### Ders 5: Yürütme Açığı Gerçektir

"Kinematik olarak doğru" ile "dinamik olarak kararlı" arasındaki açık, iki ayaklı yürüme için devasa boyuttadır. Her robotik ders kitabı ZMP + LIPM + IK'yı yürüme boru hattı olarak öğretir, ancak bunu konum aktüatörlü servolarla simülasyonda yürütmek, boru hattının mükemmel tork kontrolü varsaydığını ortaya koyar. Bu varsayım teoride görünmez ama pratikte ölümcüldür.

### Ders 6: Dürüst Dokümantasyon Önemlidir

Başarısızlıkları belgelemek, başarıları belgelemek kadar değerlidir. Yürüme başarısızlığı bize insansı kontrol hakkında tüm başarılı kilometre taşlarından daha fazlasını öğretti. Tork kontrolünün neden önemli olduğunu, PÖ'nün neden yürüme için baskın hale geldiğini ve Lab 8+'nın neyi ele alması gerektiğini anlamamızı sağladı.
