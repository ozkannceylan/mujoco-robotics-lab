# A1: Dinamik Temelleri

## Amaç

Pinocchio'nun analitik algoritmalarını kullanarak katı cisim dinamik büyüklüklerini (M(q), C(q,q̇), g(q)) hesaplamak ve MuJoCo ile çapraz doğrulama yapmak.

## Dosyalar

- Betik: `src/a1_dynamics_fundamentals.py`
- Ortak modül: `src/lab3_common.py`
- Model: `models/ur5e.xml` (tork modu), `models/ur5e.urdf`

## Teori

Katı cisim manipülatörü için hareket denklemleri:

```
M(q)q̈ + C(q,q̇)q̇ + g(q) = τ
```

Burada:
- **M(q)** — Kütle (atalet) matrisi, simetrik pozitif tanımlı (6×6)
- **C(q,q̇)** — Coriolis ve merkezkaç etkileri matrisi (6×6)
- **g(q)** — Yerçekimi vektörü (6×1)
- **τ** — Eklem torkları (6×1)

### Kütle Matrisi M(q)

Bileşik Katı Cisim Algoritması (CRBA) ile hesaplanır: `pin.crba(model, data, q)`. CRBA yalnızca üst üçgeni doldurur — manuel olarak simetrikleştirilir.

Doğrulanan özellikler:
- Simetrik: M = M^T
- Pozitif tanımlı: tüm özdeğerler > 0
- Konfigürasyona bağlı: köşegen elemanları q ile değişir

### Coriolis Matrisi C(q,q̇)

`pin.computeCoriolisMatrix(model, data, q, qd)` ile hesaplanır.

Temel özellik: Ṁ - 2C antisimetriktir (pasiflik özelliği), sayısal olarak doğrulanmıştır.

### Yerçekimi Vektörü g(q)

`pin.computeGeneralizedGravity(model, data, q)` ile hesaplanır.

MuJoCo'nun `qfrc_bias` değeri (sıfır hızda) ile çapraz doğrulama: maks hata < 0.0005 Nm.

## Çapraz Doğrulama Sonuçları

| Büyüklük | Yöntem | Maks Hata |
|----------|--------|-----------|
| g(q) | pin vs mj qfrc_bias | < 0.0005 Nm |
| M(q) | pin CRBA vs mj fullM | < 0.00004 |

## Nasıl Çalıştırılır

```bash
python3 src/a1_dynamics_fundamentals.py
```
