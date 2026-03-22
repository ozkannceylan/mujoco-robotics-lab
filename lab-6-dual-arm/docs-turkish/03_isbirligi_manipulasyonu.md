# İşbirliği Manipülasyonu

Bu belge, iki kolun bir nesneyi birlikte kavramasını, taşımasını ve bırakmasını
sağlayan üç temel bileşeni açıklar: çift empedans kontrolü, iç kuvvet yönetimi ve
`BimanualGraspStateMachine` durum makinesi.

---

## 1. Çift Empedans Kontrolü

### 1.1 Temel Denklem

Her kol için empedans torku aşağıdaki formülle hesaplanır:

```
τ = J^T · F + g(q)
```

Burada:
- `J` — uç-efektör Jacobian matrisi (6 × 6, dünya çerçevesi)
- `F` — görev uzayında sanal kuvvet/moment vektörü (6 boyutlu)
- `g(q)` — yerçekimi torkları (6 boyutlu)

Sanal kuvvet vektörü `F`, konum ve oryantasyon hatalarından oluşan bir PD denetleyicisi
tarafından üretilir:

```
F = K_p · e + K_d · ė
```

### 1.2 Hata Vektörünün Oluşturulması

6 boyutlu hata vektörü; 3 boyutlu konum hatası ve 3 boyutlu oryantasyon hatasından oluşur:

```python
# Konum hatası (doğrusal, 3)
pos_err = target.translation - ee_pose.translation

# Oryantasyon hatası (açısal, 3) — çarpım matrisinin simetrik olmayan kısmı
def orientation_error(R_des, R_cur):
    R_err = R_des.T @ R_cur - R_cur.T @ R_des
    return 0.5 * np.array([R_err[2,1], R_err[0,2], R_err[1,0]])

ori_err = orientation_error(target.rotation, ee_pose.rotation)
error   = np.concatenate([pos_err, ori_err])   # (6,)
```

### 1.3 Hız Hatası

EE hızı Jacobian ve eklem hızlarından türetilir:

```python
v_cur   = J @ qd          # mevcut EE hızı (6,)
vel_err = xd_des - v_cur  # hız hatası; xd_des = 0 ise damping etkisi
```

### 1.4 Tork Hesabı

Tam tork hesabı:

```python
F   = gains.K_p @ error + gains.K_d @ vel_err
tau = J.T @ F + g
```

### 1.5 Kazanç Yapısı

Her iki kol için kazançlar simetrik ve aynıdır. Bu tasarımın önemi bir sonraki
bölümde açıklanır:

```python
@dataclass
class DualImpedanceGains:
    K_p: np.ndarray = np.diag([400, 400, 400, 40, 40, 40])  # önt. değer
    K_d: np.ndarray = np.diag([ 60,  60,  60,  6,  6,  6])
    f_squeeze: float = 10.0   # N, iç kuvvet büyüklüğü
```

Konum rijitliği 400 N/m, oryantasyon rijitliği 40 Nm/rad olarak ayarlanır. Her iki
kol özdeş kazançlarla çalıştığında denge dışı iç kuvvetler oluşmaz.

---

## 2. İç Kuvvet Kontrolü

### 2.1 İç Kuvvet Nedir?

Bir nesneyi iki kol aynı anda tuttuğunda, kollar nesneye hem dıştan (nesneyi hareket
ettirmek için) hem de içten (kavrama kuvvetini korumak için) kuvvet uygular. Dış
kuvvetler nesneyi ivmelendirirken, iç kuvvetler nesnede net hareket oluşturmaz fakat
kavrama kuvvetini artırır.

Matematiksel olarak:

```
F_iç_toplam = F_sol_iç + F_sağ_iç = 0
```

Her iki iç kuvvetin büyüklükleri eşit ama yönleri zıttır; nesne üzerindeki etkileri
birbirini götürür.

### 2.2 Simetrik Kazanç ile Doğal Denge

Her iki kol aynı kazançlarla ve aynı hedefe göre çalıştığında empedans çerçevesi
iç kuvvetleri doğal olarak dengeler: sol kol sağa, sağ kol sola doğru eşit büyüklükte
çekilir. Açık bir kuvvet döngüsüne gerek yoktur.

### 2.3 Sıkma (Squeeze) Kuvveti Terimi

Kavrama kalitesini daha da artırmak için her iki kola ek bir "sıkma" kuvveti uygulanır.
Bu kuvvet, iki uç-efektörü birbirine doğru iten bir saf iç kuvvettir:

```
kavrama ekseni = (p_sağ - p_sol) / ||p_sağ - p_sol||

F_sıkma_sol  = +f_sıkma × kavrama_ekseni   (sola, sağa doğru iter)
F_sıkma_sağ  = -f_sıkma × kavrama_ekseni   (sağa, sola doğru iter)
```

Python uygulaması:

```python
def _compute_squeeze_torques(self, q_left, q_right):
    pos_left  = self.model.fk_left_pos(q_left)
    pos_right = self.model.fk_right_pos(q_right)

    diff = pos_right - pos_left
    dist = np.linalg.norm(diff)
    if dist < 1e-6:
        return np.zeros(6), np.zeros(6)

    grasp_axis = diff / dist   # birim vektör

    F_sq_left  = np.zeros(6); F_sq_left[:3]  = +self.gains.f_squeeze * grasp_axis
    F_sq_right = np.zeros(6); F_sq_right[:3] = -self.gains.f_squeeze * grasp_axis

    tau_left  = self.model.jacobian_left(q_left).T  @ F_sq_left
    tau_right = self.model.jacobian_right(q_right).T @ F_sq_right
    return tau_left, tau_right
```

### 2.4 Toplam Tork

Kavrama durumunda (`grasping=True`) toplam tork şöyle oluşur:

```
τ_sol  = J_sol^T · (K_p · e_sol  + K_d · ė_sol)  + g(q_sol)  + J_sol^T · F_sıkma_sol
τ_sağ  = J_sağ^T · (K_p · e_sağ  + K_d · ė_sağ)  + g(q_sağ)  + J_sağ^T · F_sıkma_sağ
```

Kavrama yoksa (`grasping=False`) sıkma terimi eklenmez:

```python
tau_left, tau_right = self.controller.compute_dual_torques(
    q_left, qd_left, q_right, qd_right,
    target_left, target_right,
    grasping=True,   # sıkma kuvvetini etkinleştir
)
```

---

## 3. BimanualGraspStateMachine

### 3.1 Durum Geçiş Şeması

```
IDLE
  │  (ilk adım)
  ▼
APPROACH  ─── her iki EE yaklaşma konumuna ulaştı ──▶  PRE_GRASP
                                                             │
                                                  her iki EE kavrama
                                                  konumuna ulaştı
                                                      + weld etkinleştir
                                                             │
                                                             ▼
                                                           GRASP
                                                             │
                                                     settle_time bekle
                                                             │
                                                             ▼
                                                           LIFT
                                                             │
                                                  her iki EE kaldırma
                                                  yüksekliğine ulaştı
                                                             │
                                                             ▼
                                                           CARRY
                                                             │
                                                  her iki EE taşıma
                                                  konumuna ulaştı
                                                             │
                                                             ▼
                                                           LOWER
                                                             │
                                                  her iki EE masa
                                                  yüksekliğine ulaştı
                                                      + weld pasifleştir
                                                             │
                                                             ▼
                                                          RELEASE
                                                             │
                                                     settle_time bekle
                                                             │
                                                             ▼
                                                          RETREAT
                                                             │
                                                  her iki EE geri çekilme
                                                  konumuna ulaştı
                                                             │
                                                             ▼
                                                           DONE
```

### 3.2 BimanualState Sayımı (Enum)

```python
class BimanualState(Enum):
    IDLE      = auto()   # Başlatılmamış
    APPROACH  = auto()   # Ön yaklaşma (clearance mesafesinde)
    PRE_GRASP = auto()   # Kavrama noktasına son yaklaşma
    GRASP     = auto()   # Weld etkin, yerleşme bekleniyor
    LIFT      = auto()   # Kaldırma (15 cm)
    CARRY     = auto()   # Yatay taşıma (30 cm)
    LOWER     = auto()   # Masa yüksekliğine indirme
    RELEASE   = auto()   # Weld pasif, bırakma
    RETREAT   = auto()   # Geri çekilme
    DONE      = auto()   # Görev tamamlandı
```

### 3.3 BimanualTaskConfig

Görev parametreleri tek bir veri sınıfında toplanır:

```python
@dataclass
class BimanualTaskConfig:
    object_pos: np.ndarray          # Nesnenin başlangıç merkez konumu (3,)
    target_pos: np.ndarray          # Nesnenin hedef merkez konumu (3,)
    lift_height: float = 0.15       # Kaldırma yüksekliği (m)
    approach_clearance: float = 0.08  # Ön yaklaşma ofseti X ekseninde (m)
    grasp_offset_x: float = 0.18    # Nesne merkezinden kavrama noktasına mesafe (m)
    position_tolerance: float = 0.01  # Durum geçiş toleransı (m)
    settle_time: float = 0.5        # GRASP ve RELEASE durumlarında bekleme süresi (s)
```

### 3.4 Sabit Hedeflerin Ön Hesabı

Tüm durumların SE3 hedefleri, `__init__` sırasında bir kez hesaplanır ve
önbelleğe alınır. Bu, simülasyon döngüsünde gereksiz hesaplama yapmaktan kaçınır:

```python
def _build_targets(self) -> None:
    obj = self.config.object_pos
    tgt = self.config.target_pos
    dx  = self.config.grasp_offset_x
    clr = self.config.approach_clearance

    # Kavrama konumları: nesne merkezinin ±X ofseti
    self._grasp_pos_left  = np.array([obj[0] - dx, obj[1], obj[2]])
    self._grasp_pos_right = np.array([obj[0] + dx, obj[1], obj[2]])

    # Yaklaşma konumları: kavramadan daha da uzak
    self._approach_pos_left  = np.array([obj[0] - dx - clr, obj[1], obj[2]])
    self._approach_pos_right = np.array([obj[0] + dx + clr, obj[1], obj[2]])

    # Kaldırma, taşıma, indirme ve geri çekilme konumları...
```

### 3.5 Oryantasyon Kısıtlamaları

Sol kol nesniye +X yönünden, sağ kol -X yönünden yaklaşır. Bunun için
EE Z ekseni sırasıyla +X ve -X dünya eksenine hizalanır:

```python
# Sol EE: Z ekseni +X'e bakar (nesniye doğru)
R_left = np.column_stack([
    np.array([0, 1, 0]),   # X_new
    np.array([0, 0, 1]),   # Y_new (yukarı)
    np.array([1, 0, 0]),   # Z_new (+X)
])

# Sağ EE: Z ekseni -X'e bakar (nesniye doğru)
R_right = np.column_stack([
    np.array([ 0, -1, 0]),
    np.array([ 0,  0, 1]),
    np.array([-1,  0, 0]),
])
```

---

## 4. Kavrama → Kaldırma → Taşıma → Bırakma Boru Hattı

### 4.1 Genel Döngü Yapısı

Simülasyon döngüsü şu şekilde çalışır:

```python
sm = BimanualGraspStateMachine(dual_model, config, gains)

while not sm.is_done:
    q_left  = mj_data.qpos[LEFT_JOINT_SLICE]
    qd_left = mj_data.qvel[LEFT_JOINT_SLICE]
    q_right  = mj_data.qpos[RIGHT_JOINT_SLICE]
    qd_right = mj_data.qvel[RIGHT_JOINT_SLICE]

    tau_left, tau_right = sm.step(
        q_left, qd_left, q_right, qd_right,
        t=mj_data.time,
        mj_model=mj_model,
        mj_data=mj_data,
    )

    mj_data.ctrl[LEFT_CTRL_SLICE]  = tau_left
    mj_data.ctrl[RIGHT_CTRL_SLICE] = tau_right
    mujoco.mj_step(mj_model, mj_data)
```

`step()` her çağrıda şunu yapar:
1. Geçiş koşullarını değerlendirir (`_maybe_transition`).
2. Aktif durum için hedefleri seçer (`_targets_for_state`).
3. `DualImpedanceController.compute_dual_torques()` çağırır.
4. `(tau_left, tau_right)` döndürür.

### 4.2 Weld Kısıtlamalarının Yönetimi

Weld kısıtlamaları iki kritik noktada değiştirilir:

**Kavrama aktivasyonu (PRE_GRASP → GRASP):**
```python
# Her iki EE kavrama konumuna ulaştığında
if self._both_near(q_left, q_right,
                   self._grasp_pos_left, self._grasp_pos_right):
    self._activate_welds(mj_model, mj_data)
    self._grasp_activate_time = t
    self._transition_to(BimanualState.GRASP, t)
```

**Bırakma (LOWER → RELEASE):**
```python
# Her iki EE masa yüksekliğine indiğinde
if self._both_near(q_left, q_right,
                   self._lower_pos_left, self._lower_pos_right):
    self._deactivate_welds(mj_model, mj_data)
    self._release_time = t
    self._transition_to(BimanualState.RELEASE, t)
```

Weld aktivasyon/pasifleştirme MuJoCo'nun equality constraint mekanizması üzerinden
gerçekleşir:

```python
def _activate_welds(self, mj_model, mj_data) -> None:
    for name in ("left_grasp", "right_grasp"):
        eq_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if eq_id >= 0:
            mj_model.eq_active[eq_id] = 1
```

### 4.3 Durum Geçiş Koşulları Özeti

| Geçiş | Koşul |
|-------|-------|
| IDLE → APPROACH | Her zaman (ilk adımda) |
| APPROACH → PRE_GRASP | Her iki EE yaklaşma konumunda (< `position_tolerance`) |
| PRE_GRASP → GRASP | Her iki EE kavrama konumunda + weld etkin |
| GRASP → LIFT | `settle_time` geçti |
| LIFT → CARRY | Her iki EE kaldırma yüksekliğinde |
| CARRY → LOWER | Her iki EE taşıma konumunda |
| LOWER → RELEASE | Her iki EE indirme konumunda + weld pasif |
| RELEASE → RETREAT | `settle_time` geçti |
| RETREAT → DONE | Her iki EE geri çekilme konumunda |

### 4.4 Sıkma Kuvvetinin Durumlara Göre Etkinliği

```
IDLE      — sıkma YOK
APPROACH  — sıkma YOK
PRE_GRASP — sıkma YOK
GRASP     — sıkma AÇIK  (weld yerleşirken tutma kuvveti sağlar)
LIFT      — sıkma AÇIK
CARRY     — sıkma AÇIK
LOWER     — sıkma AÇIK
RELEASE   — sıkma YOK   (nesne bırakılıyor)
RETREAT   — sıkma YOK
DONE      — sıkma YOK
```

### 4.5 Durum Geçiş Kaydı

Her geçiş, zaman damgasıyla birlikte `state_log` listesine eklenir:

```python
self._transition_to(BimanualState.LIFT, t)
# → state_log.append((t, "LIFT"))
```

Sonradan incelemek için:

```python
for ts, state_name in sm.get_state_log():
    print(f"t={ts:.3f}s  →  {state_name}")
```

---

## 5. Sıfırlama ve Yeniden Kullanım

Durum makinesi `reset()` metoduyla yeniden başlatılabilir:

```python
new_config = BimanualTaskConfig(
    object_pos=np.array([0.5, 0.0, 0.75]),
    target_pos=np.array([0.8, 0.0, 0.75]),
)
sm.reset(config=new_config)
```

Bu çağrı şunları yapar: durumu `IDLE`'a geri döndürür, zaman sayaçlarını temizler,
`state_log`'u siler ve `_build_targets()` ile yeni hedefleri yeniden hesaplar.

---

## 6. Başarı Kriterleri

| Metrik | Hedef Değer |
|--------|-------------|
| EE konum izleme hatası (her iki kol) | < 5 mm RMS |
| Taşıma sırasında nesne rotasyonu | < 5° |
| Nesne nihai yerleştirme hatası | < 1 cm |
| Durum geçişlerinde çarpışma | Sıfır |
| İç kuvvet büyüklüğü (kavrama sırasında) | ~10 N (ayarlanan `f_squeeze`) |

---

## Özet

İşbirliği manipülasyonu üç katmandan oluşur:

1. **DualImpedanceController** — her kol için `τ = J^T · F + g(q)` formülüyle tork
   üretir; sıkma kuvveti ile kavrama kalitesini artırır.
2. **Simetrik kazançlar** — iç kuvvetlerin doğal olarak dengelenmesini sağlar;
   açık döngü kuvvet kontrolüne gerek yoktur.
3. **BimanualGraspStateMachine** — on durumlu bir otomat aracılığıyla kavrama,
   kaldırma, taşıma ve bırakma aşamalarını orchestrate eder; weld kısıtlamalarını
   doğru zamanda etkinleştirir ve pasifleştirir.
