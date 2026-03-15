# Yorunge Takibi ve Empedans Kontrolu

Bu dokuman, TOPP-RA tarafindan uretilen zamanlandirilmis yorungenin MuJoCo
simulasyonunda nasil calistirildigini aciklar. Kontrol yontemi olarak
eklem-uzayi empedans kontrolu kullanilir.

---

## Kontrol Yasasi

Eklem-uzayi empedans kontrolu su formule dayanir:

```
tau = Kp * (q_d - q) + Kd * (qd_d - qd) + g(q)
```

Burada:
- `tau`: Uygulanan eklem torklari (6,)
- `Kp`: Oransal kazanc (konum hatasi)
- `Kd`: Turev kazanci (hiz hatasi)
- `q_d`: Istenen eklem konumu (yorungeden)
- `q`: Gercek eklem konumu (MuJoCo'dan)
- `qd_d`: Istenen eklem hizi (yorungeden)
- `qd`: Gercek eklem hizi (MuJoCo'dan)
- `g(q)`: Yercekimi telafi torku (Pinocchio'dan)

### Bilesenlerin Rolleri

**Oransal terim** `Kp * (q_d - q)`: Konum hatasini duzeltir. Robot istenen
konumdan ne kadar uzaksa, o kadar buyuk tork uretir. Bir yay gibi davranir.

**Turev terimi** `Kd * (qd_d - qd)`: Hiz hatasini duzeltir ve titresimi
sondurur. Bir damper gibi davranir.

**Yercekimi telafisi** `g(q)`: Robotun kendi agirligini tasimak icin gereken
torku saglar. Bu terim olmadan robot, yercekimi etkisiyle asagi duser ve
surekli bir konum hatasi olusur.

---

## Yercekimi Telafisi: Pinocchio RNEA

Yercekimi torku, Pinocchio'nun ters dinamik algoritmasiyla (RNEA) hesaplanir:

```python
pin.computeGeneralizedGravity(pin_model, pin_data, q)
g = pin_data.g.copy()
```

Bu fonksiyon, sifir hiz ve sifir ivme altindaki eklem torklerini hesaplar,
yani sadece yercekimini dengeleyen torku verir. MuJoCo bu hesaplamayi
yapmaz; bu nedenle Pinocchio'nun analitik beynini kullaniyoruz.

---

## Uygulama Detaylari

### Yorunge Enterpolasyonu

TOPP-RA ciktisi genellikle 1 kHz orneklemeyle uretilir. Ancak simulasyon
zaman adimi da 1 ms oldugu icin, her adimda dogrudan enterpolasyon yapilir:

```python
if t <= traj_duration:
    q_d = np.array([
        np.interp(t, times, q_traj[:, j]) for j in range(NUM_JOINTS)
    ])
    qd_d = np.array([
        np.interp(t, times, qd_traj[:, j]) for j in range(NUM_JOINTS)
    ])
else:
    q_d = q_traj[-1]       # son konumu tut
    qd_d = np.zeros(NUM_JOINTS)  # sifir hiz
```

Yorunge suresi bittikten sonra 0.5 saniye ek bekleme suresi eklenir.
Bu, robotun son konumda oturmasi (settling) icin zaman tanir.

### Tork Sinirlama

Hesaplanan torklar, UR5e'nin fiziksel tork sinirlarina kirpilir:

```python
TORQUE_LIMITS = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
tau = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)
```

Ilk uc eklem (omuz, kol) daha yuksek tork kapasitesine sahiptir. Bilek
eklemleri daha dusuk sinirlarla calisir.

---

## Kazanc Ayarlama

Laboratuvarda kullanilan kazanclar:

| Parametre | Deger | Birim    |
|-----------|-------|----------|
| `Kp`      | 400   | Nm/rad   |
| `Kd`      | 40    | Nm*s/rad |

Bu degerler su kriterleri dengeler:
- **Yuksek Kp:** Konum takibi daha siki, ancak titresim riski artar
- **Yuksek Kd:** Titresim soner, ancak sistem yavaslar
- **Kd/Kp orani:** Kritik sondurum icin `Kd = 2*sqrt(Kp*I)` yaklasimi
  kullanilabilir, ancak pratikte deneme ile ayarlanir

Sonuc olarak RMS takip hatasi 0.01 rad'in altinda tutulmustur.

---

## Simulasyon Dongusu

```python
def execute_trajectory(
    times: np.ndarray,
    q_traj: np.ndarray,
    qd_traj: np.ndarray,
    scene_path: Path | None = None,
    Kp: float = 400.0,
    Kd: float = 40.0,
) -> dict[str, np.ndarray]:
```

Dongu yapisi:

```
1. MuJoCo modelini yukle
2. Pinocchio modelini yukle
3. Ilk konfigurasyonu ayarla
4. Her zaman adiminda (1 kHz):
   a. Yorungeden q_d ve qd_d enterpolasyonu
   b. MuJoCo'dan q ve qd oku
   c. Pinocchio ile g(q) hesapla
   d. tau = Kp*(q_d-q) + Kd*(qd_d-qd) + g(q)
   e. Torklari sinirla ve uygula
   f. Verileri kaydet
   g. mj_step ile simulasyonu ilerlet
5. Sonuclari dondur
```

### Dondurulen Veriler

```python
{
    "time":      np.ndarray,  # (N,)    zaman [s]
    "q_actual":  np.ndarray,  # (N, 6)  gercek eklem konumlari
    "q_desired": np.ndarray,  # (N, 6)  istenen eklem konumlari
    "tau":       np.ndarray,  # (N, 6)  uygulanan torklar
    "ee_pos":    np.ndarray,  # (N, 3)  uc islevci konumu
}
```

Bu veriler, performans analizi ve grafik olusturma icin kullanilir.

---

## Pinocchio-MuJoCo Is Bolumlenmesi

Bu laboratuvarda mimarinin temel prensibi korunur:

```
Pinocchio = analitik beyin (FK, Jacobian, g(q), M, C)
MuJoCo    = fizik simulatoru (adim, render, temas, sensor)
```

Ozellikle:
- **Pinocchio:** Yercekimi torkunu (`g(q)`) hesaplar
- **MuJoCo:** Gercek eklem konumlarini ve hizlarini saglar, fizik adimini atar

Hesaplama asla ikisinde de tekrarlanmaz. Pinocchio'nun hesapladigi bir
buyukluk MuJoCo'da yeniden hesaplanmaz.

---

## Performans Sonuclari

Laboratuvar testlerinde elde edilen sonuclar:

| Metrik                        | Deger          |
|-------------------------------|----------------|
| RMS takip hatasi (eklem)      | < 0.01 rad     |
| Maksimum gecici hata          | < 0.05 rad     |
| Oturma suresi                 | < 0.3 s        |
| Tork siniri asilmasi          | Yok            |
| Hiz siniri asilmasi           | Yok (TOPP-RA)  |

TOPP-RA'nin hiz ve ivme sinirlarini ozen ile uygulamasi, empedans
kontrolorunun rahat calismasini saglar.

---

## Onemli Notlar

1. **Yercekimi telafisi zorunludur:** Bu terim olmadan, robot yercekimi
   etkisiyle surekli bir hata tasir.

2. **Enterpolasyon gereklidir:** TOPP-RA ornekleme frekansi ile simulasyon
   frekansi birebir esmese bile `np.interp` ile ara degerler hesaplanir.

3. **Oturma suresi:** Yorunge bitiminde 0.5 s ek bekleme, robotun son
   konumda dengelenmesini saglar.

4. **Tork sinirlama:** `np.clip` ile fiziksel sinirlarin asilmasi onlenir.
