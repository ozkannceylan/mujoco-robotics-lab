# Carpisma Kontrolu Altyapisi

Bu dokuman, Lab 4'te kullanilan carpisma kontrol sistemini aciklar. Sistem,
Pinocchio ve HPP-FCL kutuphanelerini kullanarak UR5e robot kolunun hem
kendi-carpismalarini hem de cevresel engel carpismalarini tespit eder.

---

## Genel Bakis

Hareket planlamada en temel gereksinim, robotun engellere carpmadan hareket
edebilmesidir. Bunun icin her eklem konfigurasyonunda carpisma olup olmadigini
hizlica kontrol edebilen bir altyapiya ihtiyac vardir.

Sistemimiz iki tur carpismayi kontrol eder:

1. **Kendi-carpisma (self-collision):** Robotun uzuvlarinin birbirine carpmasi
2. **Cevre carpismasi:** Robot uzuvlarinin masa ve engel kutularina carpmasi

---

## Mimari

`CollisionChecker` sinifi tum carpisma sorgularini yonetir. Ilklendirme
sirasinda su adimlar gerceklesir:

1. URDF dosyasindan Pinocchio kinematik ve carpisma modelleri yuklenir
2. Cevre engelleri (kutu seklinde) `GeometryModel`'e eklenir
3. Carpisma ciftleri kaydedilir (robot-engel ve kendi-carpisma)

```python
class CollisionChecker:
    def __init__(
        self,
        urdf_path: Path | None = None,
        obstacle_specs: list[ObstacleSpec] | None = None,
        self_collision: bool = True,
        adjacency_gap: int = 1,
    ) -> None:
```

### Engel Tanimlama

Engeller `ObstacleSpec` veri sinifi ile tanimlanir:

```python
@dataclass
class ObstacleSpec:
    name: str                  # Engel adi
    position: np.ndarray       # (3,) dunya konumu
    half_extents: np.ndarray   # (3,) kutu yari-boyutlari
```

Ornegin, masa uzerindeki bir engel:

```python
ObstacleSpec("obs1", np.array([0.30, 0.15, 0.415]), np.array([0.06, 0.06, 0.10]))
```

Bu, merkezi (0.30, 0.15, 0.415) noktasinda, 12x12x20 cm boyutlarinda bir
kutuyu ifade eder.

---

## Engellerin Geometri Modeline Eklenmesi

Her engel, Pinocchio'nun `GeometryModel` yapisina HPP-FCL kutu geometrisi
olarak eklenir. Engeller sabit (evren eklemine bagli) olarak tanimlanir:

```python
full_extents = 2.0 * spec.half_extents
shape = hppfcl.Box(full_extents[0], full_extents[1], full_extents[2])
placement = pin.SE3(np.eye(3), spec.position.copy())
geom_obj = pin.GeometryObject(
    spec.name,       # engel adi
    0,               # ebeveyn eklem (evren)
    0,               # ebeveyn cerceve
    placement,       # SE3 konumu
    shape,           # carpisma geometrisi
)
gid = self.geom_model.addGeometryObject(geom_obj)
```

HPP-FCL, `Box` fonksiyonuna **tam boyut** (full extent) bekler; bu nedenle
`half_extents` degeri 2 ile carpilir.

---

## Carpisma Cifti Kaydi

### Robot-Engel Ciftleri

Her robot geometrisi ile her engel arasinda bir carpisma cifti kaydedilir:

```python
for robot_gid in range(self._n_robot_geoms):
    for obs_gid in self._obstacle_ids:
        self.geom_model.addCollisionPair(
            pin.CollisionPair(robot_gid, obs_gid)
        )
```

### Kendi-Carpisma Ciftleri ve Komsuluk Filtreleme

Komsuluk filtreleme, fiziksel olarak carpisamayacak bitisik uzuv ciftlerini
atlar. `adjacency_gap=1` ayari, ayni ekleme veya bitisik eklemlere bagli
geometri ciftlerini dislar:

```python
if abs(ji - jj) <= adjacency_gap:
    continue  # bitisik veya ayni eklem -- atla
```

Bu filtreleme olmadan, her zaman temas halinde olan bitisik uzuvlar surekli
yanlis pozitif carpisma uretir.

---

## Temel Sorgular

### Tekil Konfigurasyon Kontrolu

```python
def is_collision_free(self, q: np.ndarray) -> bool:
```

Verilen `q` eklem acilarinda carpisma var mi kontrol eder.
`stop_at_first_collision=True` parametresi ile ilk carpisma bulunur bulunmaz
durur; bu, performans acisindan onemlidir.

### Yol Segmenti Kontrolu

```python
def is_path_free(self, q1: np.ndarray, q2: np.ndarray,
                 resolution: float = 0.05) -> bool:
```

`q1` ile `q2` arasindaki dogrusal enterpolasyonu belirli bir cozunurlukle
ornekleyerek tum segment boyunca carpisma kontrolu yapar. `resolution`
parametresi, C-uzayinda L2 normunda maksimum adim buyuklugunu belirtir.

Ornegin, iki konfigurasyon arasindaki mesafe 1.0 rad ise ve cozunurluk
0.05 rad ise, en az 21 ara konfigurasyon kontrol edilir.

### Minimum Mesafe Hesaplama

```python
def compute_min_distance(self, q: np.ndarray) -> float:
```

Tum carpisma ciftleri arasindaki minimum mesafeyi hesaplar. Negatif deger,
geometrilerin ic ice gectigini (penetrasyon) gosterir.

---

## MuJoCo ile Capraz Dogrulama

Pinocchio carpisma sonuclarini dogrulamak icin MuJoCo'nun `data.ncon` degeri
kullanilir. Her iki sistem de ayni konfigurasyonda carpisma sonuclarini
karsilastirir. Laboratuvarda %92.8 uyum orani elde edilmistir.

Kucuk farkliliklarin nedenleri:
- Geometri temsilleri arasindaki hassasiyet farklari
- HPP-FCL ve MuJoCo'nun farkli carpisma algoritmalari kullanmasi
- Carpisma sinirindaki konfigurasyonlarda sayisal hassasiyet

---

## Performans Notlari

- `is_collision_free`: Tek sorgu ~0.1 ms (HPP-FCL optimize edilmistir)
- `is_path_free`: Mesafe ve cozunurlige bagli; tipik olarak 5-50 sorgu
- RRT planlamada binlerce carpisma sorgusu yapilir; bu nedenle HPP-FCL'nin
  hizi kritik oneme sahiptir
- `stop_at_first_collision=True` ayari, erken sonlanma ile hiz kazandirir

---

## Ozet

| Bilesken                | Aciklama                                        |
|-------------------------|-------------------------------------------------|
| `CollisionChecker`      | Ana sinif -- tum sorgulari yonetir              |
| `ObstacleSpec`          | Engel tanimlama veri sinifi                     |
| `is_collision_free(q)`  | Tekil konfigurasyon kontrolu                    |
| `is_path_free(q1, q2)`  | Yol segmenti kontrolu (enterpolasyon ile)       |
| `compute_min_distance`  | Minimum mesafe hesaplama                        |
| Komsuluk filtreleme     | Bitisik uzuv ciftlerini dislar                  |
| Capraz dogrulama        | MuJoCo ile %92.8 uyum                           |
