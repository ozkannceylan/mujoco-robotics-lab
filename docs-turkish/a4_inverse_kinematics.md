# A4: Inverse Kinematics — Öğrenme Notu

## Problem

IK sorusu:

```text
Hedef: (x_target, y_target)
Bul:   (theta1, theta2)
```

Bu repoda hedef, modeldeki `end_effector` site pozisyonudur. Bu yüzden ikinci link için etkin uzunluk:

```text
L2_eff = 0.3 + 0.015 = 0.315 m
```

İç erişim yarıçapı artık sıfır değil:

```text
r_min = |L1 - L2_eff| = 0.015 m
r_max = L1 + L2_eff = 0.615 m
```

## Analitik IK

Kapalı form çözüm iki branch üretir:

```text
cos(theta2) = (x^2 + y^2 - L1^2 - L2_eff^2) / (2 L1 L2_eff)
theta2 = +/- acos(cos(theta2))
theta1 = atan2(y, x) - atan2(L2_eff sin(theta2), L1 + L2_eff cos(theta2))
```

Yorum:
- `+acos(...)` ve `-acos(...)` iki farklı dirsek konfigürasyonu verir
- bu repoda branch'ler `elbow_down` ve `elbow_up` olarak etiketlenir

## Nümerik IK

`src/a4_inverse_kinematics.py` iki iteratif yöntem içerir:

1. `pinv`
   - kare ve tam rank durumda Jacobian tersini kullanır
   - singularity yakınında hassastır
2. `dls`
   - `J^T (J J^T + lambda^2 I)^-1 e`
   - singularity yakınında daha kararlıdır

## Doğrulama

Script şu çıktıları üretir:

- tek bir hedef için iki analitik branch ve FK hata kontrolü
- 20 reachable hedef nokta için `pinv` ve `dls` benchmark
- singularity yakınında ayrı stress testi
- MuJoCo varsa `site_xpos` ile opsiyonel son kontrol

Çıktı dosyası:

- `docs/a4_ik_benchmark.csv`

## Beklenen Yorum

- Analitik IK, reachable hedeflerde çok küçük FK hatasıyla iki çözüm döndürmelidir
- `pinv`, iyi başlangıç tahminiyle hızlı yakınsar
- `dls`, kötü koşullu Jacobian durumlarında daha güvenlidir

## Bu Oturumun Sonuçları

- Örnek hedef `(0.34, 0.28)` için iki analitik çözüm bulundu:
  - `elbow_down`: `theta1=-6.166 deg`, `theta2=88.552 deg`
  - `elbow_up`: `theta1=85.111 deg`, `theta2=-88.552 deg`
- Her iki analitik çözüm için FK hatası: `0.0`
- 20 hedef benchmark:
  - `pinv`: `%100.0` başarı, `12.50` ortalama iterasyon
  - `dls`: `%100.0` başarı, `13.45` ortalama iterasyon
- Singularity stress testi:
  - başlangıç: `(theta1, theta2) = (0, 0)`
  - hedef: `(0.612, 0.020)`
  - `pinv` ilk adımda singular Jacobian nedeniyle durdu
  - `dls` `161` iterasyonda `9.831e-06` hata ile yakınsadı
