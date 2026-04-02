# Lab 6 --- Cift Kol Koordinasyonu: Mimari Belge

## 1. Sistem Genel Bakisi

Lab 6, iki bagimsiz UR5e robot kolunun esgudum icinde bir kutuyu kavrayip kaldirmasini, tasmasini ve yerlestirmesini gerceklestirir. Mimari, **Pinocchio** (analitik hesaplama motoru) ile **MuJoCo** (fizik simulatoru) arasindaki katman ayrimina dayanir.

### 1.1 Modul Bagimliliklari ve Veri Akisi

```
                        ┌─────────────────────────────────────────────┐
                        │          bimanual_state_machine.py          │
                        │   6 durumlu FSM (APPROACH → ... → PLACE)   │
                        └────────┬──────────┬──────────┬──────────────┘
                                 │          │          │
               ┌─────────────────┘          │          └──────────────────┐
               ▼                            ▼                            ▼
  ┌────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
  │ grasp_pose_         │     │  dual_arm_model.py   │     │ joint_pd_            │
  │ calculator.py       │     │  DualArmModel sinifi │     │ controller.py        │
  │                     │     │                      │     │ DualArmJointPD       │
  │ Kutu pozundan       │     │ 2x Pinocchio modeli  │     │ sinifi               │
  │ kavrama hedefleri   │     │ FK, Jacobian, DLS IK │     │                      │
  │ uretir              │     │                      │     │ tau = Kp*(q_des-q)   │
  └─────────┬──────────┘     └──────────┬───────────┘     │   + Kd*(0-qd)        │
            │                           │                  │   + qfrc_bias        │
            │                           │                  └──────────┬───────────┘
            │                           │                             │
            └─────────┬─────────────────┘                             │
                      ▼                                               ▼
            ┌──────────────────┐                          ┌───────────────────┐
            │  lab6_common.py  │                          │   MuJoCo Engine   │
            │  Sabitler, yol-  │                          │   mj_step(),      │
            │  lar, dilimler,  │ ◄────────────────────────│   qfrc_bias,      │
            │  model yukleme   │                          │   xpos, xmat,     │
            └──────────────────┘                          │   eq_data, ncon   │
                                                          └───────────────────┘
```

### 1.2 Sahne Yerlesimi

```
      Sol UR5e                              Sag UR5e
      taban (0, 0, 0)                      taban (1.0, 0, 0)
      ayni yonelim                         ayni yonelim (yaw yok)
            \                                    \
             \           Masa                     \
              \      (0.5, 0, 0.17)                \
               \       [ kutu ]                     \
                =========================================
                              Zemin
```

- Her iki taban ayni Menagerie kuaterniyonunu kullanir (mount rotasyonu yok).
- Kollarin birbirine bakmasi M3'te IK hedefleri ile saglanir, taban yonelimiyle degil.
- Kutu: 30x15x15 cm, masa uzerinde (z = 0.245 m).

### 1.3 Temel Tasarim Kararlari

| Karar | Gerekce |
|-------|---------|
| Motor aktuetorleri (tork kontrolu) | Dogrudan tork komutlari; pozisyon servolari degil. Yercekimi telafisi Pinocchio RNEA ile hesaplanir. |
| Iki ayri Pinocchio modeli | Her kol icin bir model + taban SE3 donusumu. Tek kompozit modelden daha basit. |
| Nesne merkezli cerceve | Kavrama hedefleri kutu pozundan turetilir; kol basina sabit kodlanmaz. |
| Indeks dilimleme | Sol kol: qpos[0:6], ctrl[0:6]. Sag kol: qpos[6:12], ctrl[6:12]. Kutu freejoint: qpos[12:19]. |

---

## 2. Modul Bazinda Inceleme

### 2.1 lab6_common.py --- Merkezi Sabitler Hub'i

Bu modul tum Lab 6 betiklerinin ithal ettigi yapitasidir. Temel icerikler:

| Sabit / Fonksiyon | Deger / Aciklama |
|--------------------|------------------|
| `DT` | 0.001 s (1 kHz simulasyon adimi) |
| `NUM_JOINTS_PER_ARM` | 6 |
| `NUM_JOINTS_TOTAL` | 12 |
| `Q_HOME_LEFT` | `[-pi/2, -pi/2, pi/2, -pi/2, -pi/2, 0]` |
| `Q_HOME_RIGHT` | `Q_HOME_LEFT` kopyasi (ayni taban yonelimi) |
| `TORQUE_LIMITS` | `[150, 150, 150, 28, 28, 28]` Nm |
| `VEL_LIMITS` | `[3.14, 3.14, 3.14, 6.28, 6.28, 6.28]` rad/s |
| `LEFT_JOINT_SLICE` | `slice(0, 6)` --- MuJoCo qpos/qvel/ctrl dilimi |
| `RIGHT_JOINT_SLICE` | `slice(6, 12)` |
| `TABLE_SURFACE_Z` | 0.17 m (masa ustu) |
| `BOX_HALF_EXTENTS` | `[0.15, 0.075, 0.075]` m |
| `BOX_INIT_POS` | `[0.5, 0.0, 0.245]` m |
| `mj_quat_to_pin()` | MuJoCo (w,x,y,z) -> Pinocchio (x,y,z,w) donusumu |
| `pin_quat_to_mj()` | Pinocchio (x,y,z,w) -> MuJoCo (w,x,y,z) donusumu |
| `load_mujoco_model()` | MjModel + MjData dondurur |
| `clip_torques()` | Torku `+/-TORQUE_LIMITS` arasinda kirpar |
| `get_mj_body_id()` | Isimden MuJoCo govde kimlik numarasi |
| `get_mj_site_id()` | Isimden MuJoCo site kimlik numarasi |

**Capraz-lab bagimliliklari:**
- URDF: `lab-3-dynamics-force-control/models/universal_robots_ur5e/ur5e.urdf` dosyasinin yerel kopyasi `models/ur5e.urdf` altinda. Lab 3'un elle ayarlanmis URDF'i MuJoCo Menagerie kinematik zincirine birebir eslesir (L4 dersine bakiniz).
- MJCF: `models/scene_dual.xml` her iki kolu, masayi, kutuyu ve kaynak kisitlamalarini icerir.

### 2.2 dual_arm_model.py --- DualArmModel Sinifi

Iki bagimsiz Pinocchio UR5e modelini dunya-cercevesi taban ofsetleriyle sarar.

**Baslica API:**

| Metot | Girdi | Cikti | Aciklama |
|-------|-------|-------|----------|
| `fk(arm, q)` | "left"/"right", q(6,) | `pin.SE3` | Dunya cercevesinde EE pozu (taban ofseti dahil) |
| `jacobian(arm, q)` | "left"/"right", q(6,) | `ndarray(6,6)` | Dunya cercevesinde geometrik Jakobiyen |
| `ik(arm, pos, rot, ...)` | Kol adi, hedef pos/rot | `(q, converged, info)` | Coklu baslangiçli DLS IK |

**Taban donusumu mantigi:**
- FK: Pinocchio `oMf.translation` + `base_pos` = dunya cercevesi EE konumu.
- IK: Hedef pozisyondan `base_pos` cikarilir, Pinocchio yerel cercevesinde cozulur.
- Jakobiyen: Dogrusal kısım zaten dunya cercevesinde; taban öteleme Jakobiyen'i degistirmez.

### 2.3 joint_pd_controller.py --- DualArmJointPD Sinifi

Iki bagimsiz UR5e kolu icin eklem-uzay PD kontrolcusu.

**Kontrol yasasi (kol basina):**

```
tau = Kp * (q_hedef - q) + Kd * (0 - qvel) + qfrc_bias
```

- `qfrc_bias`: MuJoCo'dan yercekimi + Coriolis telafisi. Pinocchio RNEA ile esdeger.
- Tork siniri: `np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)`.
- `saturated` ozelligi: Son cagridaki herhangi bir tork kirpildiysa `True`.

**Uygulama:** Her kol icin `(q_target, jslice, cslice)` uzerinde yineleme. Dogrudan `mj_data.ctrl` uzerine yazar.

### 2.4 grasp_pose_calculator.py --- Kavrama Pozu Hesaplayici

Kutu pozisyonundan her iki kol icin yaklasma ve kavrama standoff pozlari uretir.

**Temel fonksiyonlar:**
- `compute_grasp_poses(mj_model, mj_data)` -> `dict[str, pin.SE3]`
- `_rotation_facing(direction)` -> `ndarray(3,3)` --- Z-ekseni verilen yonu gosteren, Y-ekseni ~dunya +Z'ye yakin rotasyon matrisi

**Standoff mesafeleri:**
- `APPROACH_STANDOFF = 0.10` m (kutu yuzeyinden 10 cm)
- `GRASP_STANDOFF = 0.05` m (kutu yuzeyinden 5 cm)

### 2.5 bimanual_state_machine.py --- BimanualStateMachine Sinifi

6 durumlu sonlu durum makinesi: APPROACH -> CLOSE -> GRASP -> LIFT -> CARRY -> PLACE -> DONE.

Yardimci fonksiyonlar:
- `_wrap_joints(q, q_ref)`: Her eklemi `q_ref +/- pi` araligina sarar.
- `_find_collision_free_ik(...)`: 300 rastgele baslangiçli, MuJoCo carpisma kontrollu IK arama.
- `_compute_ee_targets_from_box(...)`: Kutu pozundan sol/sag EE hedeflerini hesaplar.
- `_solve_ik_pair(...)`: Her iki kol icin IK'yi cozer (carpisma kontrollü veya kontrol suz).

---

## 3. Kontrolcu Tasarimi

### 3.1 Neden Eklem PD Kontrolu?

Lab 3'te Kartezyen empedans kontrolu basariyla uygulanmis olmasina ragmen, Lab 6'da **eklem PD** tercih edilmistir. Gerekce:

1. **Buyuk yeniden yapilandirmalar:** HOME -> yaklasma geçisi ~150 derece eklem degisimi gerektirir. Empedans kontrolunde Kp=100 Nm/m bu mesafeyi asma kapasitesine sahip degildir.
2. **Basitlik:** Iki kol x 6 eklem = 12 serbestlik derecesi. Empedans kontrolunde her kol icin ayri Kartezyen uzay ve tekil deger ayrismasi gerekir; eklem PD dogrudan qpos uzerinde calisir.
3. **Yerçekimi telafisi:** MuJoCo'nun `qfrc_bias` verisi Pinocchio RNEA ile tutarli sekilde yercekimi + Coriolis kompanzasyonu saglar. PD + ileri besleme = kararlı izleme.

### 3.2 PD Kazanc Secimi

| Faz | Kp | Kd | Kullanim Alani |
|-----|----|----|----------------|
| M1 (kucuk hareket) | 100 | 10 | Q_HOME civarinda kuçuk sapma regülasyonu |
| M3 (buyuk hareket) | 500 | 50 | HOME -> yaklasma geçisi (~150 derece) |
| M4/M5 (rampa ile) | 300 | 40 | Smoothstep interpolasyon, gorsel akim |

**Kritik oranlar:**
- Kd/Kp ~ 0.1 --- kritik sonumleme oranina yakin.
- Kp=500'de buyuk eklem hatasi (2.5 rad) -> 1250 Nm komut, 150 Nm'ye kirpilir -> hizli ama kararlı yaklasma.
- Kp=300 + 2 s rampa -> suratin yaninda yumusak hareket.

### 3.3 Smoothstep Interpolasyon

Step-komut PD sarsintili hareket uretir (L12). Cozum: smoothstep (kubik Hermite):

```
alpha = min(1.0, t / T_rampa)
alpha_smooth = 3*alpha^2 - 2*alpha^3
q_hedef(t) = q_baslangic + alpha_smooth * (q_bitis - q_baslangic)
```

- T_rampa = 2.0 s
- Baslangicta ve bitte sifir hiz (sifir turev) saglar.
- PD kontrolcusu suresince `q_target_left/right` guncellenir.

### 3.4 Yerleşme Kriterleri

Her faz iki kosuldan biriyle sona erer:
1. **Pozisyon yakinsama:** Maksimum eklem hatasi < 0.005 rad, 0.15 s boyunca tutarlı.
2. **Hiz yakinsama:** Rampa tamamlandiktan sonra, maks eklem hizi < 0.02 rad/s, 0.15 s boyunca. Kaynak yuklu fazlarda (LIFT, CARRY) kalan pozisyon hatasi mevcuttur; hiz kriteri erken çikis saglar.

---

## 4. IK Boru Hatti

### 4.1 Sonumlu En Kucuk Kareler (DLS) Algoritmasi

Temel formul:

```
dq = J^T * (J * J^T + lambda^2 * I)^{-1} * e
```

Burada:
- **J** (6x6 veya 3x6): Geometrik Jakobiyen (dunya cercevesi)
- **e** (6,) veya (3,): Gorev-uzay hatasi `[e_pos; e_rot]` veya sadece `e_pos`
- **lambda = 0.01**: Sonumleme faktoru --- tekillige yakin dayaniklilik ile hassasiyet arasinda denge
- **dq_max = 0.5 rad**: Adim basina maksimum eklem degisimi --- asiri adim onleme
- **max_iter = 200**: Maksimum iterasyon sayisi
- **tol_pos = 1e-4 m**, **tol_rot = 1e-3 rad**: Yakinsama esikeleri

### 4.2 Adim Kisitlama

Her iterasyonda:
```python
dq_norm = ||dq||
if dq_norm > dq_max:
    dq = dq * (dq_max / dq_norm)
```

Bu, tekillik yakininda Jakobiyen'in buyuk pseudoinvers adimlari uretmesini onler. lambda=0.01 zaten bir miktar sonumleme saglar, ancak adim kisitlama ek guvenlik katmani ekler (L5).

### 4.3 Coklu Baslangiç Stratejisi

Tek baslangiç noktasi (Q_HOME) yerel minimumlara takilabilir. Coklu baslangiç:

1. **Ilk deneme:** `q_init` (verilmisse) veya `Q_HOME`.
2. **Basarisizsa:** `n_restarts` kez `Q_HOME +/- uniform(-1, +1)` rastgele perturbasyonlar denenir.
3. **6DOF IK:** `n_restarts=8` genellikle yeterlidir (20/20 yakinsamis).
4. **Sadece pozisyon (az kisitli):** `n_restarts=20` gerekebilir.

### 4.4 Taban Donusumu

Pinocchio modeli kol tabani orijininde calisir. Dunya cercevesindeki hedefler icin:

- **IK cozumunde:** `target_pos_local = target_pos_world - base_pos`
- **FK ciktisinda:** `ee_pos_world = oMf.translation + base_pos`

Rotasyon, taban oteleme donusumuyle etkilenmez (her iki taban ayni yonelime sahiptir).

### 4.5 Carpismadan Kacinma IK Araması

`_find_collision_free_ik()` standart IK'nin uzerine carpisma kontrolu ekler:

1. 300 rastgele baslangiç noktasi dene (ilki `q_init`).
2. DLS IK ile coz.
3. Yakinsan cozumu `_wrap_joints` ile sargila.
4. Sargilanmis cozumde FK dogrulamasi yap (pos hatasi < 1 mm).
5. MuJoCo `mj_forward` ile statik carpisma kontrolu:
   - `mj_data.qpos[jslice] = q_w` ayarla, `mj_forward` cagir.
   - Her teması kontrol et: masa-kutu temaslarini atla (beklenen).
   - Ilgili kolun herhangi bir geomu ile sahne carpismasi varsa reddet.
6. Carpismasiz cozumler arasindan `q_ref`'e en yakin olani sec.

Bu yaklasim APPROACH fazinda kullanilir. CLOSE ve sonrasi fazlarda temas beklendigi icin carpisma kontrolu devre disidir.

---

## 5. Kavrama Pozu Hesaplama

### 5.1 Nesne-Merkezli Cerceve

Tum kavrama hedefleri kutunun anlik pozisyon ve yoneliminden turetilir:

```python
box_pos = mj_data.xpos[box_id]          # Kutu merkezi (3,)
box_rot = mj_data.xmat[box_id] -> (3,3) # Kutu rotasyon matrisi
box_x_axis = box_rot[:, 0]              # Kutunun en uzun ekseni (+x)
box_half_x = BOX_HALF_EXTENTS[0]        # 0.15 m
```

### 5.2 Sol ve Sag Kol Hedefleri

**Sol kol** (kutunun -x tarafindan yaklasir):
```
sol_poz  = kutu_merkez - kutu_x * (yarim_x + standoff)
sol_rot  = _rotation_facing(+kutu_x)   # EE z-ekseni kutunun +x yonunu gosterir
```

**Sag kol** (kutunun +x tarafindan yaklasir):
```
sag_poz  = kutu_merkez + kutu_x * (yarim_x + standoff)
sag_rot  = _rotation_facing(-kutu_x)   # EE z-ekseni kutunun -x yonunu gosterir
```

### 5.3 Standoff Miktarlari

| Faz | Standoff | Aciklama |
|-----|----------|----------|
| APPROACH | +10 cm | Carpismasiz guvenli mesafe |
| GRASP standoff | +5 cm | Son yaklasma oncesi ara poz |
| CLOSE | -2 cm | Kutu icine 2 cm penetre --- temas guvencesi |
| LIFT/CARRY/PLACE | -2 cm | Kaynak kilitli ofset ile uyumlu (`-CONTACT_PENETRATION`) |

**Onemli:** Kaynak aktif fazlarda standoff degeri, CLOSE aninda kilitlenen EE-kutu goreli donusumu ile eslesmelidir. Farkli standoff kullanilirsa PD kontrolcusu ve kaynak kisitlamasi birbirine karsi calisir (L10).

### 5.4 Rotasyon Matrisi Olusturma

`_rotation_facing(direction)` fonksiyonu:

1. Z-ekseni = verilen yon (yaklasma yonu).
2. Y-ekseni = dunya +Z'ye mumkun oldugunca yakin (z neredeyse dikey ise +Y'ye gec).
3. X-ekseni = Y x Z capraz carpim.

Bu, EE site cercevesinin MuJoCo'daki `quat="0.7071 -0.7071 0 0"` (R_x(-90 derece)) ile ayarlanan site yonelimi ile tutarlidir (L2).

---

## 6. Durum Makinesi

`BimanualStateMachine` sinifi 6 ana durum + DONE durumundan olusur.

### 6.1 Durum-Kontrolcu Haritasi

| Durum | Kontrolcu Modu | IK Tipi | Kaynak Aktif? |
|-------|----------------|---------|---------------|
| APPROACH | PD + smoothstep | Carpismasiz (300 deneme) | Hayir |
| CLOSE | PD + smoothstep | Basit IK (temas bekleniyor) | Hayir |
| GRASP | PD (sabit hedef) | --- | Evet (etkinlestirilir) |
| LIFT | PD + smoothstep | Basit IK | Evet |
| CARRY | PD + smoothstep | Basit IK | Evet |
| PLACE | PD + smoothstep | Basit IK | Evet -> Hayir |

### 6.2 APPROACH (Durum 1)

**Amac:** Her iki kolu HOME'dan kutunun yanina getirmek.

**Iki asamali yaklasma:**
1. **HOME -> yaklasma pozu** (10 cm standoff): `_find_collision_free_ik()` ile carpismasiz IK. Bu en buyuk yeniden yapilandirma (~150 derece).
2. **Yaklasma -> kavrama standoff** (5 cm standoff): Yaklasma cozumunden tohumlanmis (seeded) IK --- gecis yalnizca ~0.68 rad.

**Teknik detaylar:**
- 300 rastgele baslangiçli carpisma kontrollu IK (sol/sag ayri ayri).
- Eklem sarmalama: `_wrap_joints()` ile q_ref +/- pi araligina sarmala --- 200 derecelik yol yerine 160 derecelik kisa yolu bulmak icin (L6).
- Tohumlu zincir: Yaklasma cozumunu kavrama IK'nin baslangiç noktasi olarak kullan.

### 6.3 CLOSE (Durum 2)

**Amac:** EE'leri kutu yuzeyinin 2 cm icine itmek.

- Standoff = `-CONTACT_PENETRATION` = -0.02 m (kutu icinde).
- Carpisma kontrolu yok: Temas bekleniyor.
- 4 s maks sureli PD + 0.15 s ek bekleme (kararli temas icin).

**Gecis kriterı:** Sol ve sag EE kutu yuzeyinden 3 cm icinde.

### 6.4 GRASP (Durum 3)

**Amac:** Kaynak kisitlamalarini etkinlestirmek.

**Kritik sira:**
1. `_set_weld_relpose()` ile guncel goreli donusumu `eq_data`'ya yaz.
2. `eq_active = 1` ile kaynak kisitlamalarini etkinlestir.
3. 0.15 s tutma.

**`eq_data` yerlesimi (11 float):**
```
[0:3]   body1 uzerinde capa noktasi (yerel cerceve)  = [0, 0, 0]
[3:6]   body1 cercevesinde body2 goreli konum
[6:10]  body1 cercevesinde body2 goreli kuaterniyon (w,x,y,z)
[10]    torquescale = 1.0
```

**Goreli donusum hesabi:**
```python
rel_pos = mat1.T @ (pos2 - pos1)
rel_mat = mat1.T @ mat2
rel_quat = mju_mat2Quat(rel_mat)
```

**UYARI:** Kaynak kisitlamalarini `eq_data` guncellemeden etkinlestirmek, kutuyu model derleme zamanindaki goreli poza teleport etmeye calisir ve kutuyu firlatir (L8).

### 6.5 LIFT (Durum 4)

**Amac:** Kutuyu 15 cm yukari kaldirmak.

```python
kaldırılmıs_kutu_poz = mevcut_kutu_poz + [0, 0, 0.15]
```

- Standoff: `-CONTACT_PENETRATION` (kaynak kilitli ofset ile uyumlu).
- Basit IK (carpisma kontrolu yok --- kaynak aktif, temas zaten mevcut).
- PD + smoothstep ile yumusak yukseltme.

### 6.6 CARRY (Durum 5)

**Amac:** Kutuyu 20 cm +y yonunde tasimak.

```python
tasinmis_kutu_poz = mevcut_kutu_poz + [0, 0.20, 0]
```

**Neden +y yonu?** Kol tabanlari x-ekseni boyunca 1 m aralikli (x=0, x=1.0). +x yonunde tasima asimetriktir --- sag kolun EE'si kendi tabanina yaklasir ve kinematik sinirlara takilir (L9). +y yonu her iki kol icin simetriktir.

### 6.7 PLACE (Durum 6)

**Amac:** Kutuyu masaya indirmek, kaynaklari birak, kollari geri cek.

**Uc adim:**

1. **Indirme:** Mutlak z hedef = `TABLE_SURFACE_Z + BOX_HALF_EXTENTS[2]` = 0.17 + 0.075 = 0.245 m. Goreli delta degil, mutlak pozisyon (L11).

2. **Kaynak birakma:** `eq_active = 0` ile her iki kaynak kisitlamasi devre disi.

3. **Kol geri cekilmesi:** EE'leri `APPROACH_STANDOFF` (10 cm) mesafesine geri cek. Kaynak birakildiginda kollar temas mesafesindeyse kutuyu itebilir (L11).

### 6.8 Durum Gecis Sema

```
APPROACH                CLOSE               GRASP
  HOME -> yaklasma       yaklasma -> temas    kaynak etkinlestir
  yaklasma -> kavrama    (4s maks + 0.15s)    (0.15s tutma)
  (carpismadan kacinma)                     
        |                     |                    |
        v                     v                    v
      LIFT                  CARRY               PLACE
  +15cm z kaldirma        +20cm y tasima       z=0.245'e indir
  (kaynak aktif)          (kaynak aktif)       kaynak birak
                                               kollar geri cek
                                                   |
                                                   v
                                                 DONE
```

---

## 7. Alinan Dersler

### L1: Taban yaw kullanma, IK hedefleriyle yon belirle

- **Belirti:** Sag kol tabanina 180 derece yaw mount uygulamak Q_HOME aynalamasini olanaksiz kildi --- uc farkli eklem isaret cevirmesi ve grid aramasina ragmen simetrik asagi bakan EE'ler uretilemedı.
- **Temel neden:** Taban yaw rotasyonu eklem uzayi-Kartezyen uzay iliskisini degistirir. Farkli taban yonelimleri varken eklem degerlerini aynalama Kartezyen pozlari aynalamiyor.
- **Cozum:** Sag koldan 180 derece yaw mount tamamen kaldirildi. Her iki taban ayni yonelim (sadece Menagerie kuaterniyonu). Q_HOME_RIGHT = Q_HOME_LEFT.
- **Cikarim:** Taban cercevelerini her iki kol icin ayni tut. Birbirine bakma yonunu IK/yol planlamasi ile coz, taban rotasyonu ile degil.

### L2: UR5e EE site cercevesi icin acik rotasyon gerekli

- **Belirti:** EE site z-ekseni govde Z'si boyunca (yana dogru) gosteriyordu, alet yaklasma yonu boyunca degil. `dot(z, -z_dunya)` kriteri hep ~0 okuyordu.
- **Temel neden:** MuJoCo siteleri varsayilan olarak govde cercevesini miras alir. UR5e alet yaklasma yonu govde +Y, +Z degil.
- **Cozum:** Her iki EE sitesine `quat="0.7071 -0.7071 0 0"` (R_x(-90 derece)) eklendi. Site z-ekseni = govde +Y = yaklasma yonu.
- **Cikarim:** EE site yonelimini her zaman acikca tanimla. Govde cercevesi eksenlerinin gorev-uzay kurallarinizla esledigini varsayma.

### L3: Masa carpismasi sag kol PD yakinsamasini engelledi

- **Belirti:** Sag kol omuz ekleminde ~0.6 rad kararli-hal hatasi. Sol kol < 0.001 rad'a yakinsamis. Ayni Kp/Kd.
- **Temel neden:** Masa x yari-genisligi 0.40 idi (masa kenari x=0.9). Sag kol tabani x=1.0'da --- UR5e omuz/ust-kol carpisma kapsulleri (yaricap 0.06) fiziksel olarak masanin icindeydi. Temas kuvvetleri omuz ekleminin hedefe ulasimasini engelledi.
- **Cozum:** Masa x yari-genisligi 0.40'tan 0.20'ye daraltildi (masa x=0.3..0.7). Her kol tabanindan 0.3 m mesafe.
- **Cikarim:** Bir eklem-uzay kontrolcusu bir kolda yakinsamazken digerinde yakinsamistsa, `data.ncon` ve temas geom ciftlerini kontrol et. Carpisma en muhtemel suçlu.

### L4: Lab 4 URDF MuJoCo Menagerie'den farkli kinematik kullaniyor

- **Belirti:** Lab 4 URDF ile Pinocchio FK, MuJoCo EE pozisyonlarindan ~99 mm sapma gosterdi. Ofset omuz_pan ile birlikte donuyordu (kol-yerel cercevede sabit).
- **Temel neden:** Lab 4'un URDF'i standart DH kurali kullanir. MuJoCo Menagerie farkli govde yerlesimi kullanir. Lab 3'un URDF'i Menagerie'ye birebir eslesmek icin elle ayarlanmisti: (1) dunya_eklemi'nde 180 derece Z rotasyonu, (2) omuz_kaldirma `rpy="0 pi/2 0" xyz="0 0.138 0"`, (3) eksenler `0 1 0`.
- **Cozum:** Lab 6 URDF'i Lab 3'un Menagerie-eslesimli kinematik zinciriyle degistirildi. FK hatasi 99 mm'den 0.000 mm'ye dustu.
- **Cikarim:** Hangi URDF'in MuJoCo modelinize esledigini her zaman dogrulayin. universal_robots_description'dan gelen "standart" DH-kurali URDF, MuJoCo Menagerie ile eslesmiyor. Lab 3'un elle ayarlanmis URDF'i Pinocchio capraz dogrulamasi icin kanonik olandir.

### L5: DLS IK icin adim kisitlama ve coklu baslangiç gerekli

- **Belirti:** DLS IK (sonumleme=0.01, maks_iter=100) 20 6DOF hedeften 4'unde ve tum sadece-pozisyon hedeflerinde basarisiz oldu. Cozucu 500-990 mm kalinti ile yerel minimumlara takildi.
- **Temel neden:** Adim kisitlama olmadan buyuk Jakobiyen pseudoinvers adimlari asim yapar. Tek baslangiç noktasi (Q_HOME) bazi hedeflerden eklem uzayinda cok uzak.
- **Cozum:** Iterasyon basina dq_max=0.5 rad adim kisitlama + coklu baslangiç (Q_HOME etrafinda n_restarts=8 rastgele perturbasyonlar). 6DOF: 20/20 yakinsamis. Sadece-pozisyon (n_restarts=20): 5/5 yakinsamis.
- **Cikarim:** 6-DOF kollar icin DLS IK her zaman adim kisitlama ve coklu baslangiç icermelidir. Sadece-pozisyon (az kisitli) 6DOF'tan daha fazla yeniden baslatma gerektirir.

### L6: IK cozumleri carpisma kontrolu ve eklem sarmalama gerektirir

- **Belirti:** Kol PD kontrolcusu IK hedeflerine yakinsamadi (hata 1.0+ rad'da takildi). Sol kol savrulma sirasinda kutuyla carpisti, sag kol ust_kol_baglantisi masaya yaslanmis durumda. 9-11 MuJoCo temasi.
- **Temel neden:** (1) IK cozucu sahne geometrisini bilmiyor. (2) Eklem sarmalama olmadan IK, 200+ derece eklem hareketi gerektiren cozumler buluyor ("uzun yoldan" gidiyor). (3) HOME -> yaklasma geçisi gercekten buyuk (~150 derece).
- **Cozum:** Uc parcali cozum: (a) Eklem sarmalama --- IK sonrasi her eklemi `q_ref +/- pi` araligina sarmala. (b) Carpismasiz IK aramasi --- 300 rastgele baslangiç degerlendır, MuJoCo temaslarini kontrol et, en yakin carpismasiz cozumu sec. (c) Tohumlu zincir --- yaklasma cozumunu kavrama IK'nin tohumu olarak kullan.
- **Cikarim:** Engellerle cift-kol sahnelerinde IK'yi carpisma kontrolu takip etmelidir. Eklem sarmalama PD kontrol icin zorunludur. Sirali IK hedeflerini (yaklasma -> kavrama) her zaman zincirlı tohumla.

### L7: Buyuk yeniden yapilandirmalar yuksek PD kazanci gerektirir

- **Belirti:** Kp=100, Kd=10 (M1 kazanclari) ~150 derecelik HOME -> yaklasma gecisini suremedi. Kollar 8 saniyede neredeyse hic hareket etmedi.
- **Temel neden:** Kucuk hareketlerde Kp=100 esik degerinde yalnizca 0.5 Nm uretir. Buyuk hareketlerde (2.5 rad hata) 250 Nm uretir --- bu 150 Nm'ye kirpilir. Asil sorun: Kd=10 hizli izleme icin Kp=100'e kiyasla fazla sonumleme --- kritik sonumlu yanitlamasi bu mesafeler icin cok yavas.
- **Cozum:** M3 icin Kp=500, Kd=50'ye yukseltildi. Her iki kol 0.62 s'de yerlesimini tamamladi, 2 ms senkronizasyon hatasi.
- **Cikarim:** PD kazanclari goreve gore olçeklenmelidir. Kucuk-hata regülasyonu (0.001 rad) dusuk kazanc kullanabilir. Buyuk yeniden yapilandirmalar (2.5 rad) daha yuksek Kp ve orantili Kd gerektirir.

### L8: Kaynak kisitlamalari etkinlestirmeden once eq_data guncellenmelidir

- **Belirti:** Kaynak kisitlamalarini `eq_active=1` ile etkinlestirmek kutuyu 90+ cm firlatti. Kutu 0.5 s'de `[0.5, 0, 0.245]`'ten `[1.12, 0.65, 0.37]`'ye gitti.
- **Temel neden:** MuJoCo kaynak kisitlamalari `eq_data`'da saklanan goreli pozu uygular. Bu deger model derleme zamaninda baslangic govde konumlarindan hesaplanir. Kavrama aninda bilek pozisyonlari kutuya gore tamamen farklidir. Kaynagi etkinlestirmek, kutuyu baslangictaki goreli poza teleport etmeye calisir.
- **Cozum:** Etkinlestirmeden once, iki govde arasindaki guncel goreli donusumu (poz + kuaterniyon) hesapla ve `mj_model.eq_data[weld_id]`'ye yaz. Yerlesim: `[anchor(3), rel_pos(3), rel_quat(4), torquescale(1)]`. Kuaterniyon (w,x,y,z) formatinda, `mju_mat2Quat` ile.
- **Cikarim:** Calisma zamani kaynak kisitlamalarini guncel goreli pozu `eq_data`'ya yazmadan asla etkinlestirme. Derlemeden gelen varsayilan eq_data neredeyse hicbir zaman istediginiz sey degildir.

### L9: Tasima yonu cift-kol calisma alani geometrisi ile sinirlidir

- **Belirti:** +x yonunde 20 cm tasima sirasinda CARRY IK basarisiz oldu. Sag kol IK cozum uretmedi.
- **Temel neden:** Her iki kol tabani y=0'da, x yonunde 1.0 m aralikla. Kutu merkezi x=0.5. +x tasima kutuyu asimetrik olarak hareket ettirir --- sag EE (kutunun x + yarim_x + standoff konumunda) sag kol tabanina (x=1.0) yaklasir. z=0.41'de UR5e, gereken -x gosterme yonelimi ile x>0.75'e ulasamaz.
- **Cozum:** Tasima yonu +y olarak degistirildi --- her iki kol icin simetrik (her ikisi de y=0'da). Calisma alani y yonunde 20 cm yer degistirme icin bol.
- **Cikarim:** Cift-kol kurulumlarinda taban-arasi eksen boyunca tasima ciddi sekilde kisitlidir. Yanal (taban eksenine dik) tasima simetrik erisilebirligini korur.

### L10: Kaynak-aktif fazlarda standoff kilitli ofset ile eslesmelidir

- **Belirti:** LIFT/CARRY IK hedefleri `GRASP_STANDOFF` (5 cm) kullanirken kaynak kisitlamalari EE'yi `-CONTACT_PENETRATION` (-2 cm, kutu icinde) pozisyonda kilitlemisti. Bu, PD hedefi ile kaynak-zorunlu pozisyon arasinda 7 cm ofset olusturdu ve ic kuvvetlere yol acti.
- **Temel neden:** Kaynak, CLOSE anindaki goreli donusumu dondurur (EE kutu yuzeyinin 2 cm icinde). Farkli standoff ile IK hedefi hesaplamak, PD kontrolcusu ile kaynak kisitlamasinin birbirine karsi calismasi demektir.
- **Cozum:** Tum kaynak-aktif fazlarda (LIFT, CARRY, PLACE) EE hedefleri kutu pozisyonundan hesaplanirken `-CONTACT_PENETRATION` standoff degeri kullanildi.
- **Cikarim:** Kaynak goreli bir pozu kilitlediginde, sonraki tum IK hedefleri ayni goreli ofset ile hesaplanmalidir.

### L11: Kutu yerlestirme mutlak z hedefi ve kol geri cekilmesi gerektirir

- **Belirti:** Kaynak birakildiktan sonra kutu masadan kaydi (z 0.26'dan 0.10'a dustu, x/y 10 cm sapmis). Rotasyon hatasi 120 derece.
- **Temel neden:** (1) PLACE goreli delta (`-LIFT_DZ`) kullandi --- kaldirma asimini kompanse etmedi, kutuyu masanin 1.5 cm uzerinde birakti. (2) Kaynak birakildiktan sonra CLOSE pozisyonundaki kollar kutuyu temas kuvvetleri ile itti.
- **Cozum:** (1) Mutlak yerlestirme z = `TABLE_SURFACE_Z + BOX_HALF_EXTENTS[2]` = 0.245 m. (2) Kaynak birakildiktan hemen sonra kollar `APPROACH_STANDOFF` (10 cm) mesafesine geri cekildi.
- **Cikarim:** Kritik yerlestirmeler icin her zaman mutlak hedef pozisyon kullanin. Kaynak birakildiktan sonra hemen geri cekilin --- temas mesafesindeki kollar nesneyi itecektir.

### L12: Step-komut PD sarsintili hareket uretir, smoothstep interpolasyon kullan

- **Belirti:** Son eklem hedefini PD kontrolcusune anlik olarak vermek hizli, sarsintili kol hareketleri uretiyordu.
- **Temel neden:** Yuksek Kp'de (500) step komut aninda maksimum tork uretir. Kol maksimum ivmeyle hizlanir, asim yapar ve salinim uretir.
- **Cozum:** Mevcut ve hedef eklem konfigurasyonlari arasinda 2 saniyelik smoothstep (3*alpha^2 - 2*alpha^3) interpolasyon eklendi. Dusuk kazanclar (Kp=300, Kd=40) rampa ile birlestirildiginde yavas, yumusak hareket uretir.
- **Cikarim:** Gorsel olarak hosnut edici robot hareketi icin PD hedefini her zaman rampa ile verin, step-komut vermeyin. Smoothstep interpolasyon basit ve etkilidir.

---

## MuJoCo <-> Pinocchio Esleme Tablosu

| MuJoCo Adi | Pinocchio Cercevesi | Notlar |
|-------------|---------------------|--------|
| `left_ee_site` | `ee_link` (sol model) | Dunya cercevesi = Pinocchio FK + `LEFT_BASE_POS` |
| `right_ee_site` | `ee_link` (sag model) | Dunya cercevesi = Pinocchio FK + `RIGHT_BASE_POS` |
| `qpos[0:6]` | `q` (sol model) | Birebir esleme, ayni eklem sirasi |
| `qpos[6:12]` | `q` (sag model) | Birebir esleme, ayni eklem sirasi |
| `qfrc_bias[0:6]` | `pin.rnea(model, data, q, 0, 0)` | Yercekimi + Coriolis (sol kol) |
| `qfrc_bias[6:12]` | `pin.rnea(model, data, q, 0, 0)` | Yercekimi + Coriolis (sag kol) |

---

## Dosya Yapisi

```
lab-6-dual-arm/
├── models/
│   ├── assets/              # Menagerie UR5e OBJ aglar (paylasimli)
│   ├── scene_dual.xml       # Ana MJCF: sol + sag kollar, masa, kutu
│   ├── ur5e_left.xml        # Sol kol govde hiyerarsisi + motor aktuetorleri
│   ├── ur5e_right.xml       # Sag kol govde hiyerarsisi + motor aktuetorleri
│   └── ur5e.urdf            # UR5e URDF (Pinocchio icin, kol basina)
├── src/
│   ├── lab6_common.py       # Sabitler, yollar, model yukleyiciler, indeks dilimleme
│   ├── dual_arm_model.py    # DualArmModel: FK, Jakobiyen, DLS IK
│   ├── joint_pd_controller.py # DualArmJointPD: yercekimi telafili PD kontrolcu
│   ├── grasp_pose_calculator.py # Kutu pozundan kavrama hedefleri
│   ├── bimanual_state_machine.py # 6 durumlu FSM
│   ├── m0_validate_scene.py # M0: sahne dogrulama
│   ├── m1_independent_motion.py # M1: bagimsiz eklem hareketi
│   ├── m2_fk_validation.py  # M2: FK capraz dogrulamasi
│   ├── m2_ik_validation.py  # M2: IK dogrulamasi
│   ├── m2_ik_visual.py      # M2: IK gorsellestirmesi
│   ├── m3_coordinated_approach.py # M3: esgudumlü yaklasma
│   ├── m4_cooperative_carry.py    # M4: isbirlikci tasima
│   └── m5_capstone_demo.py  # M5: tam demo + video
├── tests/
├── docs/
├── docs-turkish/
│   └── ARCHITECTURE_TR.md   # Bu belge
├── media/
└── tasks/
    ├── PLAN.md
    ├── ARCHITECTURE.md
    ├── TODO.md
    └── LESSONS.md
```
