# Koordineli Hareket

Bu belge, iki kolun zaman içinde eş güdümlü olarak hareket ettirilmesini sağlayan
`CoordinatedPlanner` modülünü ve bunu mümkün kılan nesne merkezli çerçeve soyutlamasını
açıklar. Üç koordinasyon modu ele alınır: senkronize, master-slave ve simetrik.

---

## 1. Nesne Merkezli Çerçeve (Object-Centric Frame) Soyutlaması

### 1.1 Motivasyon

İki kolun bağımsız hedeflere doğru hareket etmesi bir koordinasyon değildir. Gerçek
koordinasyon, kolların bir nesne üzerindeki sabit kavrama noktalarını koruyarak
birlikte hareket etmesidir. Bu gereksinimi karşılamak için tüm planlamayı nesne
çerçevesine göre yapmak en doğru yaklaşımdır.

### 1.2 ObjectFrame Sınıfı

`ObjectFrame`, nesnenin dünya çerçevesindeki konumunu (SE3) ve her kolun nesneye
göre göreli hedefini (kavrama ofseti) bir arada tutar:

```python
@dataclass
class ObjectFrame:
    pose: pin.SE3            # Nesnenin dünya çerçevesindeki pozu
    grasp_offset_left: pin.SE3   # Sol EE'nin nesne çerçevesindeki ofseti
    grasp_offset_right: pin.SE3  # Sağ EE'nin nesne çerçevesindeki ofseti
```

Her kolun hedefi şu formülle bulunur:

```
T_ee_world = T_object_world * T_ee_object
```

Python'da:

```python
def get_left_target(self) -> pin.SE3:
    return self.pose * self.grasp_offset_left

def get_right_target(self) -> pin.SE3:
    return self.pose * self.grasp_offset_right
```

### 1.3 Kavrama Ofsetinin Hesaplanması

Mevcut uç-efektör konumlarından ve bilinen nesne konumundan ofset otomatik türetilebilir:

```python
@classmethod
def from_ee_poses(cls, object_pose, left_ee, right_ee) -> "ObjectFrame":
    # offset = object_pose^{-1} * ee_pose
    offset_left  = object_pose.inverse() * left_ee
    offset_right = object_pose.inverse() * right_ee
    return cls(pose=object_pose,
               grasp_offset_left=offset_left,
               grasp_offset_right=offset_right)
```

### 1.4 Nesne Taşındığında Hedeflerin Güncellenmesi

Nesne yeni bir konuma taşındığında kavrama ofsetleri sabit kalır, yalnızca `pose`
güncellenir:

```python
def moved_to(self, new_pose: pin.SE3) -> "ObjectFrame":
    return ObjectFrame(
        pose=new_pose,
        grasp_offset_left=self.grasp_offset_left,
        grasp_offset_right=self.grasp_offset_right,
    )
```

Bu tasarım sayesinde taşıma yörüngesi boyunca nesne için bir dizi SE3 pozu
üretmek yeterlidir; kol hedefleri `moved_to()` çağrısıyla otomatik türetilir.

---

## 2. SynchronizedTrajectory Veri Yapısı

Planlayıcı, her zaman damgasında her iki kola ait eklem açısı ve hızını birlikte
barındıran bir yapı döndürür:

```python
@dataclass
class SynchronizedTrajectory:
    timestamps: np.ndarray   # (T,)       saniye
    q_left:     np.ndarray   # (T, 6)     sol eklem açıları
    qd_left:    np.ndarray   # (T, 6)     sol eklem hızları
    q_right:    np.ndarray   # (T, 6)     sağ eklem açıları
    qd_right:   np.ndarray   # (T, 6)     sağ eklem hızları
    duration:   float        # toplam süre (s)
```

Her iki kol aynı `timestamps` dizisini paylaşır; bu sayede her adımda `q_left[i]`
ve `q_right[i]` eşzamanlı olarak uygulanabilir.

---

## 3. Koordinasyon Modu 1: Senkronize (Synchronized)

### 3.1 İlke

Her iki kol aynı anda başlar ve aynı anda hedeflerine ulaşır. Daha uzun yol
gerektiren kol toplam süreyi belirler; diğer kol bu süreye yetişecek şekilde
daha yavaş hareket eder.

### 3.2 Adımlar

1. Başlangıç ve bitiş pozları için IK çözülür.
2. Her kol için gerekli yörünge süresi hesaplanır (maksimum eklem hızına göre).
3. İki süreden büyük olanı ortak süre olarak seçilir: `duration = max(T_left, T_right)`.
4. Her kol, ortak süre boyunca doğrusal görev-uzayı (task-space) enterpolasyonu yapılarak
   planlanır.

### 3.3 Görev Uzayı Doğrusal Enterpolasyonu

Konum için doğrusal, oryantasyon için SLERP (Spherical Linear Interpolation)
kullanılır:

```
p(t) = p_0 + (t / T) * (p_f - p_0)

R(t) = R_0 * Exp( (t / T) * Log(R_0^T * R_f) )
```

Pinocchio ile:

```python
alpha = t / duration                         # [0, 1]

# Konum (doğrusal)
pos = (1 - alpha) * pose_start.translation + alpha * pose_end.translation

# Oryantasyon (SLERP via SE3 interpolation)
log_R = pin.log3(pose_start.rotation.T @ pose_end.rotation)
R = pose_start.rotation @ pin.exp3(alpha * log_R)

T_interp = pin.SE3(R, pos)
```

### 3.4 Planlayıcı Çağrısı

```python
traj = planner.plan_synchronized_linear(
    target_left=pose_L_goal,
    target_right=pose_R_goal,
    q_left_init=q_left_0,
    q_right_init=q_right_0,
    duration=None,   # None ise otomatik hesaplanır
)
```

---

## 4. Koordinasyon Modu 2: Master-Slave

### 4.1 İlke

Master kol (varsayılan olarak sol) bağımsız bir hedef yörüngesi izler. Slave kol
(sağ), nesne çerçevesi aracılığıyla master'ın konumundan türetilen hedefleri takip
eder. Nesnenin her iki kola göre göreli pozu sabit kalır.

### 4.2 Geometri

Her adımda:

1. Master EE'nin dünya çerçevesindeki pozu FK ile hesaplanır.
2. Nesne pozu, master EE'den kavrama ofseti tersine çevrilip türetilir:
   ```
   T_object = T_master_ee * T_master_offset^{-1}
   ```
3. Slave hedefi nesne çerçevesinden hesaplanır:
   ```
   T_slave_target = T_object * T_slave_offset
   ```

### 4.3 Planlayıcı Çağrısı

```python
traj = planner.plan_master_slave(
    master_waypoints=[pose_1, pose_2, pose_3],
    object_frame=obj_frame,
    q_left_init=q_left_0,
    q_right_init=q_right_0,
    master="left",
)
```

### 4.4 Doğrulama

İki uç-efektör arasındaki göreli poz zaman içinde sabit kalmalıdır. Bunu
`relative_ee_pose()` ile izlemek mümkündür:

```python
rel_pose = dual_model.relative_ee_pose(q_left, q_right)
# rel_pose.translation normu zamana göre sabit olmalı
```

---

## 5. Koordinasyon Modu 3: Simetrik (Symmetric)

### 5.1 İlke

Her iki kol da bir nesne yörüngesinden türetilmiş hedefleri izler. Ne master ne de
slave vardır; nesne çerçevesi tek otorite kaynağıdır. Bu mod taşıma ve bırakma
görevleri için uygundur.

### 5.2 Nesne Yörüngesi

Planlayıcıya SE3 pozlarından oluşan bir liste verilir; her eleman bir zaman
adımındaki nesne konumunu temsil eder:

```python
object_traj = [
    pin.SE3(R_id, np.array([0.5, 0.0, 0.80])),  # başlangıç
    pin.SE3(R_id, np.array([0.5, 0.0, 0.95])),  # kaldırma
    pin.SE3(R_id, np.array([0.8, 0.0, 0.95])),  # taşıma
    pin.SE3(R_id, np.array([0.8, 0.0, 0.80])),  # indirme
]
```

### 5.3 Kol Hedeflerinin Türetilmesi

Her `object_traj[i]` için:

```python
frame_i = object_frame.moved_to(object_traj[i])
target_left_i  = frame_i.get_left_target()
target_right_i = frame_i.get_right_target()
q_left_i  = dual_model.ik_left(target_left_i,  q_left_prev)
q_right_i = dual_model.ik_right(target_right_i, q_right_prev)
```

### 5.4 Planlayıcı Çağrısı

```python
traj = planner.plan_symmetric(
    object_trajectory=object_traj,
    object_frame=obj_frame,
    q_left_init=q_left_0,
    q_right_init=q_right_0,
)
```

Başarı kriteri: EE'den nesne merkezine göreli konum hatası < 2 mm.

---

## 6. IK Entegrasyonu

Tüm planlama modları görev uzayında enterpolasyon yapar ve ardından IK çözer.
IK, Levenberg-Marquardt sönümlü en küçük kareler yöntemiyle uygulanır:

```
Δq = (J^T J + λI)^{-1} J^T e
q  ← q + α Δq
```

Parametreler: `λ = 1×10⁻⁶` (sönümleme), `α = 0.5` (adım büyüklüğü), maks. 100 yineleme,
yakınsama toleransı `tol = 1×10⁻⁴`.

Dünya çerçevesindeki hedef, IK çağrılmadan önce kolun yerel çerçevesine dönüştürülür:

```python
def ik_left(self, target: pin.SE3, q_init) -> np.ndarray | None:
    # Dünya -> kol yerel çerçeve
    local_target = self.base_left.inverse() * target
    return self._solve_ik(self.model_left, self.data_left,
                          self.ee_fid_left, local_target, q_init, ...)
```

IK başarısız olursa (None döndürürse) planlayıcı bir uyarı kaydeder ve önceki
konfigürasyonu korur.

---

## 7. Çarpışma Denetimi

Planlayıcı, her üretilen konfigürasyonu `DualCollisionChecker` ile doğrular:

```python
for i, (q_l, q_r) in enumerate(zip(q_left_traj, q_right_traj)):
    if not collision_checker.is_collision_free(q_l, q_r):
        raise CollisionError(f"Adım {i}'de çarpışma tespit edildi")
```

Paralel yaklaşım testi (`is_path_free`) ise iki nokta arasındaki doğrusal yolu
`n_checks` ara noktayla kontrol eder.

---

## Özet

| Mod | Kontrol kaynağı | Tipik kullanım |
|-----|-----------------|----------------|
| Senkronize | Her kol bağımsız hedef | Eşzamanlı yaklaşım |
| Master-slave | Sol kol yönlendirir, sağ takip eder | Asimetrik görevler |
| Simetrik | Nesne yörüngesi her iki kolu yönlendirir | Taşıma ve bırakma |

Tüm modlar `SynchronizedTrajectory` döndürür; aynı yürütme döngüsü kullanılır.
