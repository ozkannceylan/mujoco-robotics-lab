# A1: Dinamik Temelleri

## Amaç

Pinocchio'nun analitik algoritmalarını kullanarak katı cisim dinamik büyüklüklerini (M(q), C(q,q̇), g(q)) hesaplamak ve MuJoCo ile çapraz doğrulama yapmak.

## Dosyalar

- Betik: `src/a1_dynamics_fundamentals.py`
- Ortak modül: `src/lab3_common.py`
- Analitik model: `models/ur5e.urdf`
- Yurutulen MuJoCo yigi: `lab3_common.py` uzerinden yuklenen Menagerie UR5e + Robotiq

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

MuJoCo'nun `qfrc_bias` degeri (sifir hizda) ile capraz dogrulama: maks hata `8.01e-06`.

## Çapraz Doğrulama Sonuçları

| Buyukluk | Yontem | Maks Hata |
|----------|--------|-----------|
| g(q) | Pinocchio vs MuJoCo `qfrc_bias` | `8.01e-06` |
| M(q) | Pinocchio CRBA vs MuJoCo `mj_fullM()` | `3.34e-05` |

## Nasıl Çalıştırılır

```bash
python3 src/a1_dynamics_fundamentals.py
```
