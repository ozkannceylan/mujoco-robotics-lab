# Yol Isleme: Kisayol Alma ve TOPP-RA

Bu dokuman, RRT/RRT* planlamacisinin urdugu ham yolun nasil iyilestirildigini
ve zaman-optimal bir yorungeye donusturuldugunu aciklar.

---

## Neden Yol Isleme Gerekli?

RRT/RRT* tarafindan uretilen yollar genellikle:
- Gereksiz zigzag ve sapmalara sahiptir
- Cok sayida ara nokta icerir (ornegin 12 veya daha fazla)
- Zaman bilgisi tasimaz (sadece geometrik C-uzayi noktalardir)

Yol isleme boru hatti iki asamadan olusur:

```
Ham RRT yolu (12 nokta)
    |
    v
[Kisayol Alma]  -->  Sadeletirilmis yol (2-4 nokta)
    |
    v
[TOPP-RA]       -->  Zamanlandirilmis yorunge (t, q, qd, qdd)
```

---

## Asama 1: Kisayol Alma (Shortcutting)

Kisayol alma, gereksiz ara noktalari kaldirarak yolu kisaltir. Algoritma
basittir:

1. Yoldan rastgele iki bitisik olmayan nokta sec (indeks `i` ve `j`, `j >= i+2`)
2. Bu iki nokta arasindaki dogrudan segmentin carpismadan olup olmadigini kontrol et
3. Eger carpismadan ise, aradaki tum noktalari kaldir

```python
def shortcut_path(
    path: list[np.ndarray],
    collision_checker: CollisionChecker,
    max_iter: int = 200,
    seed: int | None = None,
) -> list[np.ndarray]:
```

### Algoritma Detayi

```python
for _ in range(max_iter):
    if len(result) <= 2:
        break
    i = rng.integers(0, len(result) - 2)
    j = rng.integers(i + 2, len(result))

    if collision_checker.is_path_free(result[i], result[j]):
        result = result[:i + 1] + result[j:]
```

### Sonuclar

Laboratuvar testlerinde tipik sonuclar:
- Giris: 12 ara nokta
- Cikis: 2 ara nokta (sadece baslangic ve hedef)
- Tum ara noktalar kaldirilmistir cunku dogrudan segment carpismadan gecmektedir

Bu azaltma, sonraki TOPP-RA asamasinin daha duz ve hizli bir yorunge
uretmesini saglar.

---

## Asama 2: TOPP-RA Zaman Parametrizasyonu

TOPP-RA (Time-Optimal Path Parameterization based on Reachability Analysis),
geometrik bir yolu eklem hiz ve ivme sinirlarini gozeterek mumkun olan en
kisa surede izleyen bir yorungeye donusturur.

### Matematiksel Temel

Geometrik bir yol `q(s)`, s in [0, 1] parametre uzerinde tanimlanir. Amac,
zamana bagli bir fonksiyon `s(t)` bulmaktir oyle ki:

- Eklem hizlari: `dq/dt = (dq/ds) * (ds/dt)` hiz sinirlarini asmasin
- Eklem ivmeleri: `d2q/dt2` ivme sinirlarini asmasin
- Toplam sure minimize edilsin

TOPP-RA bu problemi, erisebilirlik analizi ile verimli bir sekilde cozer.

### Uygulama

```python
def parameterize_topp_ra(
    path: list[np.ndarray],
    vel_limits: np.ndarray,
    acc_limits: np.ndarray,
    dt: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
```

Adimlar:

1. **Tekrar eden noktalarin kaldirilmasi:** Cok yakin noktalar spline
   hesaplamasinda sorun yaratir.

2. **Yay uzunlugu parametrizasyonu:** Ara noktalar arasindaki kumulatif
   L2 mesafesine gore [0, 1] araliginda parametreler atanir:

```python
ss = np.zeros(n_waypoints)
for i in range(1, n_waypoints):
    ss[i] = ss[i-1] + np.linalg.norm(waypoints[i] - waypoints[i-1])
ss /= ss[-1]  # [0, 1]'e normalize et
```

3. **Kubik spline enterpolasyonu:**

```python
path_spline = ta.SplineInterpolator(ss, waypoints)
```

4. **Kisitlarin tanimlanmasi:**

```python
vlim = constraint.JointVelocityConstraint(vel_limits)
alim = constraint.JointAccelerationConstraint(acc_limits)
```

5. **TOPP-RA cozucusunun calistirilmasi:**

```python
instance = ta.algorithm.TOPPRA(
    [vlim, alim], path_spline,
    parametrizer="ParametrizeConstAccel"
)
traj = instance.compute_trajectory()
```

6. **Yorungenin orneklenmesi:** `dt=0.001` aralikla zaman, konum, hiz ve
   ivme dizileri uretilir.

### Kisitlar

UR5e icin kullanilan sinirlar:

| Parametre        | Deger                                    |
|------------------|------------------------------------------|
| Hiz siniri       | [3.14, 3.14, 3.14, 6.28, 6.28, 6.28] rad/s |
| Ivme siniri      | [8.0, 8.0, 8.0, 16.0, 16.0, 16.0] rad/s^2  |

Bilek eklemleri (4-6) daha yuksek sinira sahiptir cunku daha hafiftirler.

### Cikti Formati

Fonksiyon dort dizi dondurur:

```python
times, positions, velocities, accelerations = parameterize_topp_ra(path, ...)
# times:          (N,)    zaman [s]
# positions:      (N, 6)  eklem konumlari [rad]
# velocities:     (N, 6)  eklem hizlari [rad/s]
# accelerations:  (N, 6)  eklem ivmeleri [rad/s^2]
```

Bu veriler dogrudan yorunge takip kontrolorune aktarilir.

---

## Hata Yonetimi

TOPP-RA uygun bir parametrizasyon bulamazsa (ornegin cok dar bir
konfigurasyon gecisi nedeniyle), `RuntimeError` firlatilir:

```python
if traj is None:
    raise RuntimeError("TOPP-RA failed to find a feasible parameterization")
```

Bu durum, yolun yeniden planlanmasini veya kisitlarin gevsetilmesini
gerektirir.

---

## Onemli Noktalar

1. **Kisayol alma rastgeledir:** Her calistirmada farkli sonuc verebilir.
   Tekrarlanabilirlik icin `seed` parametresi kullanilmalidir.

2. **TOPP-RA zaman-optimaldir:** Uretilen yorunge, verilen sinirlar dahilinde
   mumkun olan en kisa sureli hareketi saglar. Bu, endustriyel uygulamalarda
   dongu suresinin en aza indirilmesi icin kritiktir.

3. **Carpisma guvenligi korunur:** Kisayol alma sirasinda her yeni segment
   `is_path_free` ile kontrol edilir. TOPP-RA ise mevcut yol geometrisini
   degistirmez, sadece zamanlamasini ayarlar.

4. **Spline yumusakligi:** Kubik spline enterpolasyonu, noktalar arasinda
   surekliligi saglar. Bu, ani hiz degisimlerini onler ve mekanik
   bilesenlerin korunmasina yardimci olur.
