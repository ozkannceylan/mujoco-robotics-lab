# Paralel Çeneli Tutucu Tasarımı — Lab 5

## Genel Bakış

Lab 5, MJCF XML olarak oluşturulmuş özel bir paralel çeneli tutucu kullanır. Tutucu, UR5e'nin `tool0` flanşına bağlanmıştır ve simetrik parmak hareketini aynalayan bir eşitlik kısıtıyla tek bir konum aktüatörü tarafından kontrol edilen iki kayan parmağa sahiptir.

---

## Fiziksel Yapı

```
tool0 (bilek flanşı)
  └── gripper_base  [pos="0 0 0.020"]
        ├── gripper_adapter  (montaj plakası)
        ├── left_finger      [pos="0 +0.015 0.060"]
        │     ├── left_finger_geom  (yapısal gövde)
        │     └── left_pad         (sürtünme pedi)
        └── right_finger     [pos="0 -0.015 0.060"]
              ├── right_finger_geom
              └── right_pad
```

`GRIPPER_CLOSED` (eklem konumu = 0) durumunda temel boyutlar:
| Eleman | Merkeze göre konum (Y ekseni) |
|--------|-------------------------------|
| Sol ped iç yüzü | +0.019 m |
| Sağ ped iç yüzü | -0.019 m |
| **İç aralık** | **38 mm** |

`GRIPPER_OPEN` (eklem konumu = 0.030 m) durumunda:
| Eleman | Merkeze göre konum (Y ekseni) |
|--------|-------------------------------|
| Sol ped iç yüzü | +0.049 m |
| Sağ ped iç yüzü | -0.049 m |
| **İç aralık** | **98 mm** |

---

## Eklem Mekaniği

Her iki parmak da tutucu Y ekseni boyunca hareket eden `slide` eklemlerdir:

```xml
<joint name="left_finger_joint" type="slide"
       axis="0 1 0" range="0 0.030"
       damping="2.0" armature="0.0001"/>
```

### Eşitlik Kısıtı (Ayna)

Tek bir eşitlik kısıtı, sağ parmağı sola aynalar:

```xml
<equality>
  <joint name="finger_mirror"
         joint1="left_finger_joint"
         joint2="right_finger_joint"
         polycoef="0 1 0 0 0"/>
</equality>
```

`polycoef="0 1 0 0 0"` → `q_sağ = 1 × q_sol`. Bu, tutucuyu 2 aktüatörlü DOF'tan 1'e indirir.

---

## Aktüasyon

Tutucuyu tek bir konum aktüatörü kontrol eder:

```xml
<position name="gripper" joint="left_finger_joint"
          kp="200" ctrllimited="true" ctrlrange="0 0.030"/>
```

- **kp=200**: yay sabiti (N/m). 20 mm'lik bir nesnede ~4 N kapama kuvveti sağlar.
- `ctrl[6] = 0.000` → **KAPALI** (kavrama)
- `ctrl[6] = 0.030` → **AÇIK** (bırakma)

---

## Koordinat Kuralları

Q_HOME konumunda:
- `tool0` Z ekseni = dünya **-Z** (kol aşağıya bakıyor)
- `tool0` Y ekseni ≈ dünya **Y** (parmaklar dünya Y yönünde kayıyor)
- `gripper_site` (parmak ucu merkezi) tool0 kökeninden 0.090 m aşağıda

IK, kutu konumuna `GRIPPER_TIP_OFFSET = 0.090 m` ekleyerek tool0 hedefini hesaplar:
```
tool0_hedefi = kutu_konumu + [0, 0, GRIPPER_TIP_OFFSET]
```

---

## Tutucu Kontrolör API

`src/gripper_controller.py` dosyasındaki tam uygulama için:

| Fonksiyon | Açıklama |
|-----------|----------|
| `open_gripper(mj_data)` | `ctrl[6] = 0.030` ayarlar |
| `close_gripper(mj_data)` | `ctrl[6] = 0.000` ayarlar |
| `step_until_settled(m, d, max_steps)` | Parmak hızı eşiğin altına düşene kadar adımlar |
| `get_finger_position(m, d)` | Sol parmak eklem konumunu döndürür (m) |
| `is_gripper_settled(m, d)` | Hız < 5×10⁻⁴ m/s ise True döndürür |
| `is_gripper_in_contact(m, d)` | Herhangi bir parmak geom'u nesneye dokunuyorsa True |
