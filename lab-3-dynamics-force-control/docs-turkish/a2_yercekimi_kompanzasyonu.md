# A2: Yerçekimi Kompanzasyonu

## Amaç

En basit dinamik tabanlı denetleyiciyi uygulamak: τ = g(q). Robot, minimal sapma ile herhangi bir konfigürasyonda yerçekimine karşı durmalıdır.

## Dosyalar

- Betik: `src/a2_gravity_compensation.py`
- Testler: `tests/test_gravity_comp.py`

## Teori

Yerçekimi kompanzasyonu, her zaman adımında yerçekimi torkunu iptal eder:

```
τ = g(q)
```

g(q), Pinocchio'nun `computeGeneralizedGravity()` fonksiyonu ile hesaplanır. Bu, sonraki tüm denetleyicilerin temelidir — empedans ve kuvvet kontrolü yerçekimi kompanzasyonunu temel olarak ekler.

### Neden çalışır

τ = g(q) olduğunda, hareket denklemleri şuna indirgenir:

```
M(q)q̈ + C(q,q̇)q̇ = 0
```

Eklem sönümlemesi (modelimizde 1.0 Nm·s/rad) ile herhangi bir hız üstel olarak azalır ve robot başlangıç konfigürasyonunda kalır.

## Sonuçlar

| Konfigürasyon | Maks sapma (3s) |
|---------------|-----------------|
| Q_HOME | 0.0006 rad |
| Q_ZEROS | 0.002 rad |
| Rastgele | < 0.01 rad |

Omuz eklemine 20 Nm darbe pertürbasyonundan sonra kol ~2 saniyede orijinal pozisyonuna geri döner.

## Nasıl Çalıştırılır

```bash
python3 src/a2_gravity_compensation.py
```
