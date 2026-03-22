# Çift Kol Kurulumu

Lab 6, iki ayrı UR5e kolunu tek bir MuJoCo sahnesi içinde koordineli biçimde çalıştırmayı hedefler.
Bu belge; Pinocchio modellerinin nasıl oluşturulduğunu, MuJoCo sahne yapısını, FK çapraz
doğrulamasını ve HPP-FCL tabanlı kol-kol çarpışma kontrolünü açıklar.

---

## 1. Genel Tasarım İlkesi

Tek kollu laboratuvarlardan çift kola geçişte temel zorluk, her iki kolu aynı dünya
koordinat sisteminde tutarlı biçimde temsil etmektir. Lab 6'da bu sorun iki yoldan çözülür:

- **Pinocchio tarafında:** Her kol için ayrı bir model yüklenir; kolun dünya içindeki konumu
  bir baz SE3 dönüşümü (`base_left`, `base_right`) ile belirtilir.
- **MuJoCo tarafında:** Her kol için ayrı bir MJCF dosyası hazırlanır; gövde ve eyleyici
  isimleri çakışmaması için `left_` / `right_` önekiyle ayrıştırılır.

Bu yaklaşım, mevcut tek-kol kodunu olduğu gibi yeniden kullanmayı mümkün kılar.

---

## 2. İki Ayrı Pinocchio Modeli ve Baz SE3 Dönüşümleri

### 2.1 Neden İki Ayrı Model?

Tek bir 12-DOF birleşik model yerine iki bağımsız 6-DOF model tercih edilir. Bunun
nedenleri:

1. Her kol standart bir UR5e zinciridir; dallanmalı bir kinematik ağaç gerektirmez.
2. FK, Jacobian ve yerçekimi hesaplamaları her kol için bağımsız çalışır.
3. Önceki laboratuvarlardaki IK kodu değiştirilmeden yeniden kullanılabilir.

### 2.2 Baz Dönüşümleri

Sol kol dünya orijininde sabit durur; sağ kol ise X ekseninde 1 m ötede ve Z ekseni
etrafında 180° döndürülmüş konumdadır:

```python
# lab6_common.py
import pinocchio as pin
import numpy as np

LEFT_BASE_SE3 = pin.SE3.Identity()

_R_right = pin.utils.rotate("z", np.pi)          # 180° yaw
_t_right = np.array([1.0, 0.0, 0.0])             # 1 m sağda
RIGHT_BASE_SE3 = pin.SE3(_R_right, _t_right)
```

### 2.3 DualArmModel Başlatma

`DualArmModel` sınıfı aynı URDF dosyasını iki kez yükler ve her birine farklı bir baz
dönüşümü uygular:

```python
# dual_arm_model.py (kısaltılmış)
class DualArmModel:
    def __init__(self, urdf_path, base_left, base_right):
        self.base_left  = base_left
        self.base_right = base_right

        self.model_left = pin.buildModelFromUrdf(str(urdf_path))
        self.data_left  = self.model_left.createData()
        self.ee_fid_left = self.model_left.getFrameId("ee_link")

        self.model_right = pin.buildModelFromUrdf(str(urdf_path))
        self.data_right  = self.model_right.createData()
        self.ee_fid_right = self.model_right.getFrameId("ee_link")
```

### 2.4 Dünya Çerçevesinde FK

Pinocchio, FK'yı kolun kendi yerel çerçevesinde hesaplar. Dünya çerçevesine dönüşüm
için baz SE3 dönüşümü soldan çarpılır:

```
T_ee_world = base * T_ee_local
```

```python
def fk_left(self, q: np.ndarray) -> pin.SE3:
    pin.forwardKinematics(self.model_left, self.data_left, q)
    pin.updateFramePlacements(self.model_left, self.data_left)
    local_pose = self.data_left.oMf[self.ee_fid_left]
    return self.base_left * local_pose   # dünya çerçevesi
```

Sağ kol için de aynı işlem yapılır; yalnızca `base_right` kullanılır.

---

## 3. MuJoCo Sahne Yapısı

### 3.1 Önek (Prefix) Kullanımı

MuJoCo, tüm gövde, eklem ve eyleyici isimlerinin tek bir isim uzayı içinde benzersiz
olmasını zorunlu kılar. İki aynı UR5e modelini aynı sahnede kullanmak için her eklemin
adı öneklenir:

| Pinocchio adı | Sol kol (MuJoCo) | Sağ kol (MuJoCo)  |
|---------------|------------------|-------------------|
| `shoulder_pan_joint` | `left_shoulder_pan_joint` | `right_shoulder_pan_joint` |
| `ee_link` | `left_ee_link` | `right_ee_link` |
| `shoulder_pan` | `left_shoulder_pan` | `right_shoulder_pan` |

`scene_dual.xml`, `ur5e_left.xml` ve `ur5e_right.xml` dosyalarını `<include>` etiketiyle
dahil eder. Eklem indeksleri sabitleri aracılığıyla dilimlenebilir:

```python
LEFT_JOINT_SLICE  = slice(0, 6)   # mj_data.qpos[0:6]
RIGHT_JOINT_SLICE = slice(6, 12)  # mj_data.qpos[6:12]
LEFT_CTRL_SLICE   = slice(0, 6)   # mj_data.ctrl[0:6]
RIGHT_CTRL_SLICE  = slice(6, 12)  # mj_data.ctrl[6:12]
```

### 3.2 Weld Kısıtlamaları (Equality Constraints)

Simülasyonda tutma kuvvetini modellemek için MuJoCo'nun equality (weld) kısıtlaması
kullanılır. Kavrama anında her iki uç-efektör kutuya sabitlenir:

```xml
<!-- scene_dual.xml (kısaltılmış) -->
<equality>
  <weld name="left_grasp"  body1="left_ee_link"  body2="box" active="false"/>
  <weld name="right_grasp" body1="right_ee_link" body2="box" active="false"/>
</equality>
```

Kısıtlamalar başlangıçta pasiftir (`active="false"`); durum makinesi GRASP durumuna
geçtiğinde etkinleştirilir:

```python
eq_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, "left_grasp")
mj_model.eq_active[eq_id] = 1   # etkinleştir
mj_model.eq_active[eq_id] = 0   # pasifleştir (bırakma)
```

### 3.3 Sahne Bileşenleri

`scene_dual.xml` aşağıdaki öğeleri içerir:

- Sol ve sağ UR5e kolları (`<include>` ile)
- İki kol arasında ortak bir masa
- Serbest gövde (free body) olarak büyük bir kutu (30 × 15 × 15 cm, ~2 kg)
- İki kolu ve çalışma alanını görecek şekilde konumlandırılmış kamera
- Zemin düzlemi, ışıklar ve gök kutusu

---

## 4. FK Çapraz Doğrulaması

Pinocchio ve MuJoCo'nun aynı konfigürasyonda aynı uç-efektör konumunu verdiğini
doğrulamak kritik öneme sahiptir. Baz dönüşümlerinin doğru uygulandığı bu şekilde
güvence altına alınır.

### 4.1 Doğrulama Protokolü

```python
import numpy as np
import mujoco
from dual_arm_model import DualArmModel
from lab6_common import load_mujoco_model, LEFT_JOINT_SLICE, RIGHT_JOINT_SLICE

dual_model = DualArmModel()
mj_model, mj_data = load_mujoco_model()

rng = np.random.default_rng(42)

for _ in range(5):
    q_left  = rng.uniform(-np.pi, np.pi, 6)
    q_right = rng.uniform(-np.pi, np.pi, 6)

    # MuJoCo FK
    mj_data.qpos[LEFT_JOINT_SLICE]  = q_left
    mj_data.qpos[RIGHT_JOINT_SLICE] = q_right
    mujoco.mj_forward(mj_model, mj_data)

    ee_left_mj  = mj_data.xpos[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_ee_link")]
    ee_right_mj = mj_data.xpos[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_ee_link")]

    # Pinocchio FK (dünya çerçevesi)
    ee_left_pin  = dual_model.fk_left(q_left).translation
    ee_right_pin = dual_model.fk_right(q_right).translation

    assert np.allclose(ee_left_pin,  ee_left_mj,  atol=1e-3), f"Sol FK uyuşmazlığı: {ee_left_pin} vs {ee_left_mj}"
    assert np.allclose(ee_right_pin, ee_right_mj, atol=1e-3), f"Sağ FK uyuşmazlığı: {ee_right_pin} vs {ee_right_mj}"
```

Başarı kriteri: Her iki kol için konum hatası < 1 mm.

### 4.2 Jacobian Doğrulaması

Jacobian doğrulamasında sonlu farklar yöntemi kullanılır:

```python
eps = 1e-7
q = np.zeros(6)
J_pin = dual_model.jacobian_left(q)      # (6, 6) analitik

J_fd = np.zeros((6, 6))
for i in range(6):
    dq = np.zeros(6); dq[i] = eps
    p_plus  = dual_model.fk_left(q + dq).translation
    p_minus = dual_model.fk_left(q - dq).translation
    J_fd[:3, i] = (p_plus - p_minus) / (2 * eps)

assert np.allclose(J_pin[:3, :], J_fd, atol=1e-5)
```

---

## 5. Kol-Kol Çarpışma Kontrolü (HPP-FCL)

### 5.1 Kontrol Edilen Çarpışma Çiftleri

`DualCollisionChecker`, dört çarpışma kategorisini yönetir:

| Kategori | Açıklama |
|----------|----------|
| Sol öz-çarpışma | Sol kolun kendi linkleri arasındaki çarpışmalar |
| Sağ öz-çarpışma | Sağ kolun kendi linkleri arasındaki çarpışmalar |
| Çapraz çarpışma | Sol kol linkleri ile sağ kol linkleri arası |
| Çevre çarpışması | Her kol ile masa ve diğer nesneler arası |

### 5.2 Temel Arayüz

```python
class DualCollisionChecker:
    def is_collision_free(self,
                          q_left: np.ndarray,
                          q_right: np.ndarray) -> bool:
        """Tüm çarpışma çiftleri için çarpışmasızlık kontrolü."""

    def get_min_distance(self,
                         q_left: np.ndarray,
                         q_right: np.ndarray) -> float:
        """Tüm çiftler arasındaki minimum mesafeyi döndürür (m)."""

    def is_path_free(self,
                     q_left_start, q_left_end,
                     q_right_start, q_right_end,
                     n_checks: int = 10) -> bool:
        """İki konfigürasyon arasındaki doğrusal yolu kontrol eder."""
```

### 5.3 Baz Dönüşümlerinin Geometri Nesnelerine Uygulanması

HPP-FCL, her linkin çarpışma geometrisini dünya koordinatlarında konumlandırmak için
baz SE3 dönüşümüne ihtiyaç duyar. Bu dönüşüm Pinocchio'dan çekilir:

```python
pin.forwardKinematics(model_left, data_left, q_left)
pin.updateFramePlacements(model_left, data_left)

for geom_obj in collision_model_left.geometryObjects:
    frame_id = geom_obj.parentFrame
    # Dünya çerçevesine dönüşüm: baz * yerel dönüşüm
    T_local = data_left.oMf[frame_id]
    T_world = base_left * T_local
    geom_obj.placement = T_world
```

### 5.4 Çarpışma Sonuçlarının Yorumlanması

```python
checker = DualCollisionChecker(...)

q_safe = Q_HOME_LEFT, Q_HOME_RIGHT
print(checker.is_collision_free(*q_safe))        # True beklenir
print(checker.get_min_distance(*q_safe))         # pozitif değer

# Kollar üst üste bindiğinde
q_collide_left  = np.array([0, -np.pi/4, np.pi/2, 0, 0, 0])
q_collide_right = np.array([0, -np.pi/4, np.pi/2, 0, 0, 0])
print(checker.is_collision_free(q_collide_left, q_collide_right))  # False beklenir
```

---

## 6. Bağımlılıklar ve Önceki Laboratuvarlarla İlişki

| Özellik | Kaynak |
|---------|--------|
| Impedans kontrolü deseni | Lab 3 |
| HPP-FCL geometri arayüzü | Lab 4 |
| Weld kısıtlama yönetimi | Lab 5 |
| Kuaterniyon dönüşüm yardımcıları | `lab6_common.py` içinde yeniden yazıldı |

Her lab bağımsız çalışabilmek için bağımlılıkları kendi içinde yeniden uygular;
çapraz-lab içe aktarma (cross-lab import) yapılmaz.

---

## Özet

Lab 6'nın temel altyapısı üç bileşenin bir araya gelmesinden oluşur:

1. **İki bağımsız Pinocchio modeli** — baz SE3 dönüşümleri ile dünya çerçevesinde FK,
   Jacobian ve yerçekimi hesabı sağlar.
2. **Önekli MuJoCo MJCF dosyaları** — isim çakışmalarını önler, weld kısıtlamaları
   aracılığıyla rijit kavramayı modelleştirir.
3. **HPP-FCL çarpışma kontrolü** — sol öz, sağ öz, çapraz ve çevre çarpışmalarını
   tek bir arayüzden sorgular.
