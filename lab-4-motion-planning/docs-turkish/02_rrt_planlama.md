# RRT ve RRT* Planlama Algoritmalari

Bu dokuman, Lab 4'te uygulanan ornekleme tabanli hareket planlama
algoritmalarini aciklar: temel RRT ve asimptotik olarak optimal RRT*.

---

## Problem Tanimi

Verilen:
- Baslangic konfigurasyonu `q_start` (6 eklem acisi)
- Hedef konfigurasyonu `q_goal` (6 eklem acisi)
- Engelli bir calisma ortami

Bul:
- `q_start`'tan `q_goal`'a carpismadan ulasan bir C-uzayi yolu

C-uzayi (konfigurasyon uzayi), robotun tum eklem acilarinin olusturdugu
6 boyutlu uzaydir. Her nokta, robotun tamamen tanimlanmis bir durusunu
temsil eder.

---

## RRT Algoritmasi

RRT (Rapidly-exploring Random Tree), C-uzayinda rastgele ornekleme ile
hizla buyuyen bir agac yapisi olusturur.

### Temel Adimlar

1. Agaci `q_start` ile baslatilir
2. Her iterasyonda:
   a. Rastgele bir `q_rand` konfigurasyonu orneklenir
   b. Agactaki en yakin dugum `q_nearest` bulunur
   c. `q_nearest`'ten `q_rand`'a dogru `step_size` kadar ilerlenip `q_new` elde edilir
   d. `q_nearest` -- `q_new` segmenti carpismadan geciyorsa dugum eklenir
   e. `q_new`, hedefe yeterince yakinsa yol bulunmustur

### Ornekleme ve Hedef Yanliligi

```python
def _sample(self, q_goal: np.ndarray) -> np.ndarray:
    if self._rng.random() < self.goal_bias:
        return q_goal.copy()
    return self._rng.uniform(self.lower, self.upper)
```

`goal_bias=0.1` ayari, orneklerin %10'unun dogrudan hedefe yonlendirilmesini
saglar. Bu, agacin hedefe dogru buyumesini tesvik eder.

### Yonlendirme (Steer)

```python
def _steer(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    diff = q_to - q_from
    dist = np.linalg.norm(diff)
    if dist <= self.step_size:
        return q_to.copy()
    return q_from + (diff / dist) * self.step_size
```

Uzaklik `step_size`'dan kucukse hedefin kendisi dondurulur. Aksi halde,
hedef yonunde `step_size` kadar ilerlenir. Laboratuvarda `step_size=0.3` rad
kullanilmistir.

---

## RRT* Algoritmasi

RRT*, temel RRT'nin asimptotik olarak optimal versiyonudur. Iki ek mekanizma
icerir:

1. **En iyi ebeveyn secimi:** Yeni dugum icin sadece en yakin dugum degil,
   belirli bir yaricap icindeki tum komsu dugumler arasinda en dusuk maliyetli
   ebeveyn secilir.

2. **Yeniden baglama (rewiring):** Yeni dugum eklendikten sonra, komsulardaki
   mevcut dugumlerin maliyetleri kontrol edilir. Eger yeni dugum uzerinden
   daha kisa bir yol varsa, ebeveyn guncellenir.

### En Iyi Ebeveyn Secimi

```python
near_indices = self._near(q_new)   # yaricap icindeki komsular
best_parent = nearest_idx
best_cost = self._tree[nearest_idx].cost + dist(nearest, q_new)

for idx in near_indices:
    candidate_cost = self._tree[idx].cost + dist(tree[idx].q, q_new)
    if candidate_cost < best_cost:
        if self.cc.is_path_free(self._tree[idx].q, q_new):
            best_parent = idx
            best_cost = candidate_cost
```

### Yeniden Baglama

```python
for idx in near_indices:
    new_cost = best_cost + dist(q_new, self._tree[idx].q)
    if new_cost < self._tree[idx].cost:
        if self.cc.is_path_free(q_new, self._tree[idx].q):
            self._tree[idx].parent = new_idx
            self._propagate_cost(idx, new_cost)
```

Maliyet guncellemesi, `_propagate_cost` ile tum alt dugumlere yayilir.

---

## RRT ve RRT* Karsilastirmasi

| Ozellik              | RRT                    | RRT*                        |
|----------------------|------------------------|-----------------------------|
| Optimallik           | Hayir                  | Asimptotik optimal          |
| Hesaplama maliyeti   | Dusuk                  | Daha yuksek (komsuluk arama)|
| Yol kalitesi         | Genellikle uzun/zigzag | Daha kisa, daha duz         |
| Ilk cozum hizi       | Hizli                  | Ayni (ilk cozum sonrasi iyilestirir) |

Laboratuvar sonuclarinda RRT* yollari, temel RRT'ye gore tutarli bir sekilde
daha kisa maliyet uretmistir.

---

## Parametreler

| Parametre       | Deger    | Aciklama                                      |
|-----------------|----------|-----------------------------------------------|
| `step_size`     | 0.3 rad  | Her adimda maksimum ilerleme mesafesi          |
| `goal_bias`     | 0.1      | Orneklerin %10'u hedefe yonlendirilir          |
| `rewire_radius` | 1.0 rad  | RRT* komsuluk yaricapi                         |
| `goal_tolerance`| 0.15 rad | Hedefe ulasilmis sayilma esigi                 |
| `max_iter`      | 5000     | Maksimum iterasyon sayisi                      |

---

## Planlama Arayuzu

```python
planner = RRTStarPlanner(
    collision_checker=cc,
    step_size=0.3,
    goal_bias=0.1,
    rewire_radius=1.0,
)

path = planner.plan(
    q_start=q_home,
    q_goal=q_target,
    max_iter=5000,
    rrt_star=True,    # False: temel RRT
    seed=42,
)
```

Dondurulen `path`, `q_start`'tan `q_goal`'a kadar siralanmis
`list[np.ndarray]` formatinda C-uzayi konfigurasyonlaridir.
Yol bulunamazsa `None` dondurulur.

---

## Agac Yapisi ve Gorsellestirme

Her dugum su bilgileri icerir:

```python
@dataclass
class RRTNode:
    q: np.ndarray          # eklem konfigurasyonu (6,)
    parent: int | None     # agactaki ebeveyn indeksi
    cost: float            # kokten toplam maliyet (L2 normu)
```

`visualize_plan` fonksiyonu, agac kenarlarini ve bulunan yolu 3B gorev
uzayinda (FK ile EE konumlarina donusturulerek) cizdirmek icin kullanilir.
Engeller tel kafes kutu olarak gosterilir.

---

## Onemli Tasarim Kararlari

1. **L2 mesafe metrigi:** C-uzayinda oklid mesafesi kullanilir. Bu, eklem
   agirliklarini esit sayar. Gelismis uygulamalarda agirlikli metrik
   kullanilabilir.

2. **Dogrusal enterpolasyon:** Kenar carpisma kontrolu, iki konfigurasyon
   arasinda dogrusal enterpolasyon ile yapilir. 0.05 rad cozunurluk,
   tipik engel boyutlari icin yeterli hassasiyet saglar.

3. **Temel RRT'de erken donus:** Temel RRT, hedefe ulastiginda hemen
   dondurur. RRT* ise tum iterasyonlari tamamlayarak daha iyi yol arar.

4. **Tekrarlanabilirlik:** `seed` parametresi, ayni sonuclari elde etmek
   icin rastgele sayi uretecini sabitler.

---

RRT ve RRT*, yuksek boyutlu C-uzaylarinda carpismadan yol bulmak icin etkili
algoritmalardir. Planlama sonuclari yol isleme (shortcutting + TOPP-RA)
asamasina aktarilir.
