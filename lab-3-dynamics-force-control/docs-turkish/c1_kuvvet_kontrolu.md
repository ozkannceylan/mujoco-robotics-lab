# C1: Hibrit Konum-Kuvvet Kontrolü

## Amaç

Hibrit kontrol uygulamak: XY düzleminde konum kontrolü, Z'de PI kuvvet kontrolü. Masa yüzeyinde sabit temas kuvveti korurken uç efektör konumunu kontrol etmek.

## Dosyalar

- Kuvvet denetleyicisi: `src/c1_force_control.py`
- Çizgi izleme: `src/c2_line_trace.py`
- Sahne: `models/scene_table.xml`
- Testler: `tests/test_force_control.py`

## Teori

### Hibrit Konum-Kuvvet Kontrolü

Temel fikir: kısıtlı görevlerde (cilalama, montaj vb.) bazı Kartezyen yönler konum kontrollü, diğerleri kuvvet kontrollü olmalıdır:

```
S_p = diag(1, 1, 0)  — XY'de konum kontrolü
S_f = diag(0, 0, 1)  — Z'de kuvvet kontrolü
```

### Kontrol Yasası

**Konum (XY):**
```
F_p = K_p · S_p · (x_d - x) + K_d · S_p · (ẋ_d - ẋ)
```

**Kuvvet (Z) — PI + hız sönümlemesi:**
```
e_f = F_istenen - F_ölçülen
F_f = -(K_fp · e_f + K_fi · ∫e_f dt) - K_dz · ẋ_z
```

**Birleşik:**
```
τ = J^T · (F_p + F_f) + g(q)
```

### Temas Kuvveti Ölçümü

Kuvvetler `mj_contactForce()` ile okunur, yalnızca probe_tip ve table_top geometrileri arasındaki temaslar filtrelenir. Ham kuvvetler EMA alçak geçiren filtre (α=0.2) ile yumuşatılır.

### Çarpışma Filtreleme

MuJoCo'nun `contype`/`conaffinity` bit maskeleri hangi geometrilerin çarpışacağını kontrol eder:
- Kol geometrileri: contype=1, conaffinity=1
- Masa + prob: contype=2, conaffinity=2

## Sonuçlar

### Statik Kuvvet Tutma

| Metrik | Değer |
|--------|-------|
| Hedef kuvvet | 5.0 N |
| Ortalama kuvvet | 4.95 N |
| Std kuvvet | 0.09 N |
| ±1N içinde | %100 |
| XY hatası | 0.62 mm |

### Sabit Kuvvetli Çizgi İzleme (50mm)

| Metrik | Değer |
|--------|-------|
| Hedef kuvvet | 5.0 N |
| Ortalama kuvvet | 4.99 N |
| Std kuvvet | 0.33 N |
| ±1N içinde | %98.1 |
| Ort. XY hatası | 1.96 mm |
| Maks XY hatası | 3.19 mm |

## Faz Durum Makinesi

```
YAKLAŞMA → OTURMA → İZLEME → TUTMA
    │          │        │        │
    ▼          ▼        ▼        ▼
  Empedans   Hibrit   Hibrit   Hibrit
  iniş      başlangıç + çizgi  bitiş
             XY       yörünge
```

## Nasıl Çalıştırılır

```bash
# Statik kuvvet tutma demosu
python3 src/c1_force_control.py

# Sabit kuvvetli çizgi izleme
python3 src/c2_line_trace.py
```

## Sınırlamalar

- Dikey alet konfigürasyonunda Jacobian X-satırı küçüktür (~0.01), bu temas sırasında XY takip bant genişliğini sınırlar. 50mm'den uzun çizgiler daha yavaş yörüngeler gerektirir.
