# MuJoCo'da Temas Fiziği — Lab 5

## Genel Bakış

Güvenilir kavrama için sağlam temas modelleri gereklidir. MuJoCo, temasların davranışını belirleyen beş temel parametre sunar: `condim`, `friction`, `solref`, `solimp` ve `mass`. Bu belge her parametrenin ne kontrol ettiğini ve Lab 5'te seçilen değerleri açıklar.

---

## Temas Parametreleri

### `condim` — Temas Boyutluluğu

Bir temas noktasının uygulayabileceği kuvvet sayısını tanımlar.

| Değer | Dahil edilen kuvvetler | Kullanım alanı |
|-------|----------------------|----------------|
| 1 | Sadece normal | Sürtünmesiz yüzeyler |
| 3 | Normal + 2B teğetsel | Standart sürtünme |
| 4 | Normal + 2B teğetsel + burulma | Kavrama, dönen temas |
| 6 | Tam 6-DOF sarma kuvveti | Yumuşak temaslar |

**Lab 5 tercihi: `condim="4"`** — Hem parmak pedleri hem de kavranacak kutu condim=4 kullanır. Burulma bileşeni, kol hareket ederken kutunun tutuş içinde dönmesini önler.

---

### `friction` — Sürtünme Katsayıları

Üç katsayı: `[μ_kayma, μ_burulma, μ_yuvarlanma]`

- **μ_kayma** (birincil): kayan hareket için teğetsel sürtünme. ≥ 1.0 değerleri iyi tutuş sağlar.
- **μ_burulma**: eksenel döndürmeyi önler.
- **μ_yuvarlanma**: yuvarlanma direnci. Genellikle küçük (0.001–0.01).

**Lab 5 değerleri:**
```xml
friction="1.5 0.005 0.0001"
```

μ_kayma = 1.5, 150 g'lık bir kutu için güçlü tutuş sağlar.

---

### `solref` — Kısıt Referansı

`solref="zaman_sabiti rijitlik"` — kısıt kuvvetinin nasıl artacağını kontrol eder.

- **zaman_sabiti**: yükselme süresi (saniye). Küçük = daha sert.
- **rijitlik**: sönümleme oranı. Varsayılan 1.0 (kritik sönümlü).

**Lab 5 değeri: `solref="0.002 1"`** — çok sert temaslar (2 ms yükselme süresi). Parmak altında kutunun görünür şekilde sıkışmasını önler.

---

### `solimp` — Kısıt Empedansı

`solimp="dmin dmax genişlik orta_nokta kuvvet"` — temas yumuşaklığını kontrol eder.

**Lab 5 değeri: `solimp="0.99 0.99 0.001"`** — neredeyse rijit temaslar. 1 mm penetrasyon sonrasında tam rijitlik devreye girer.

---

### Kutu Kütlesi

Kutu kütlesi, kaldırmak için gereken parmak kuvvetini doğrudan etkiler.

**Lab 5 değeri: `mass="0.15"` (150 g)** — gerçekçi küçük nesne. Kp=200 konum aktüatörü ile tutulduğu doğrulandı.

---

## Geometrik Kısıt: Minimum Parmak Aralığı

Sık yapılan tasarım hatası: `GRIPPER_CLOSED` konumundaki minimum parmak aralığı, nesnenin genişliğinden **küçük** olmalıdır.

Lab 5'te:
- Parmak gövdesi Y-ofseti: **±0.015 m**
- Ped iç yüzü (joint=0): **0.019 m**
- Kutu yarı genişliği: **0.020 m**

1 mm örtüşme, MuJoCo'nun sert temasının tutuş kuvveti oluşturması için yeterlidir.

> **Ders:** Kontrol kodu yazmadan önce her zaman statik bir sahnede tutucu geometrisini prototipleyip `ped_iç_yüzü < nesne_yarı_genişliği` koşulunu doğrulayın.

---

## Referanslar

- [MuJoCo Temas Dokümantasyonu](https://mujoco.readthedocs.io/en/latest/computation.html#contacts)
- [MuJoCo XML Referansı — geom](https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-geom)
