# A1: MuJoCo Ortam Kurulumu — Öğrenme Notu

## MuJoCo Nedir?

**MuJoCo** = "Multi-Joint dynamics with Contact". Robotik simülasyon için yazılmış bir fizik motoru.

Gazebo veya Isaac Sim'den farkı: daha hafif, Python API ile doğrudan erişilebilir, headless batch simülasyon için optimize. ML pipeline'larında tercih sebebi hız — binlerce simülasyonu paralel çalıştırabilirsin.

---

## MJCF: Robotun Tanım Dili

MuJoCo'da her şey **MJCF** (MuJoCo XML Format) ile tanımlanır. Mental model:

```
<mujoco>
  ├── <option>        ← fizik parametreleri (gravity, timestep)
  ├── <worldbody>     ← sahne hiyerarşisi (body → joint → geom)
  ├── <actuator>      ← motorlar (joint'lere bağlanır)
  └── <sensor>        ← ölçümler (pozisyon, hız, kuvvet)
```

### Üç Temel Kavram

| Kavram | Ne İşe Yarar | Analoji |
|--------|-------------|---------|
| **Body** | Koordinat frame'i. Görünmez referans noktası. | Bir iskelet eklemi |
| **Geom** | Fiziksel şekil (collision + görsel). Body'ye bağlı. | O ekleme bağlı kemik/et |
| **Joint** | İki body arası hareket serbestliği. Body'nin *içinde* tanımlanır. | Eklemin hareket yönü |

**Hiyerarşi:** Body'ler iç içe (parent-child). Joint, child body'nin parent'a göre nasıl hareket edeceğini tanımlar. Geom, body'nin fiziksel varlığı.

---

## Bizim Robot: 2-Link Planar Manipulator

```
         joint1          joint2         end-effector
  (base) ──○──── Link1 ────○──── Link2 ────●
           │    (0.3m)      │    (0.3m)
           │ z-ekseni       │ z-ekseni
           │ hinge          │ hinge
```

- **2 link**, her biri 0.3m
- **2 hinge joint**, z-ekseni etrafında döner (XY düzleminde çalışır)
- **2 motor**, her joint'e bir tane (ctrl range: [-10, 10])
- **Gravity kapalı** — düzlemsel robot, yerçekimi komplikasyonu istemiyoruz (şimdilik)

### MJCF'deki Kritik Detaylar

```xml
<!-- Body hiyerarşisi: base → link1 → link2 -->
<body name="link1" pos="0 0 0">
  <joint name="joint1" type="hinge" axis="0 0 1"/>
  <geom fromto="0 0 0  0.3 0 0"/>       ← link geometrisi

  <body name="link2" pos="0.3 0 0">     ← link1'in ucunda başlar
    <joint name="joint2" type="hinge" axis="0 0 1"/>
    <geom fromto="0 0 0  0.3 0 0"/>

    <site name="end_effector" pos="0.3 0 0"/>  ← takip noktası
  </body>
</body>
```

**`pos="0.3 0 0"`** → link2, link1'in ucundan (0.3m) başlıyor. Bu parent-child ilişkisi sayesinde joint1 döndüğünde link2 de birlikte döner — kinematik zincir budur.

**`site`** → Fiziksel etkisi yok, sadece bir takip noktası. End-effector pozisyonunu `data.site_xpos` ile okuruz.

---

## MuJoCo API — Temel Kullanım

```python
import mujoco

# 1. Model yükle
model = mujoco.MjModel.from_xml_path("models/two_link.xml")
data  = mujoco.MjData(model)

# 2. Simülasyon adımı
mujoco.mj_step(model, data)

# 3. Veri oku
data.qpos          # joint açıları [θ₁, θ₂]
data.qvel          # joint hızları [θ̇₁, θ̇₂]
data.site_xpos[i]  # site'ların Cartesian pozisyonları

# 4. Motor komutu ver
data.ctrl[0] = 5.0   # joint1 motoru
data.ctrl[1] = -3.0  # joint2 motoru

# 5. Kinematiği güncelle (step atmadan)
mujoco.mj_forward(model, data)
```

### model vs data

- **model** (`MjModel`): Sabit parametreler — link uzunlukları, kütle, joint limitleri. Simülasyon boyunca değişmez.
- **data** (`MjData`): Anlık durum — joint açıları, hızlar, kuvvetler. Her `mj_step()` ile güncellenir.

---

## Doğrulama Sonuçları

Çalıştırdığımız testten:

| Kriter | Sonuç |
|--------|-------|
| `mj_step()` hatasız çalışıyor | ✓ 100 adım, 0.2s simülasyon süresi |
| `data.qpos` 2 elemanlı | ✓ shape=(2,) |
| Motor komutu joint açılarını değiştiriyor | ✓ ctrl=[5, -3] → Δθ=[+0.055, -0.055] rad |
| End-effector pozisyonu okunuyor | ✓ site_xpos ile x=0.60, y=0.02 |

---

## Sonraki Adım: Forward Kinematics (A2)

Şimdi joint açıları verildiğinde end-effector'ın nerede olduğunu *kendi hesabımızla* bulacağız ve MuJoCo'nun `site_xpos` değeriyle karşılaştıracağız.
