# B1: Kartezyen Empedans Kontrolü

## Amaç

Görev uzayı empedans kontrolü uygulamak: uç efektörün Kartezyen uzayda ayarlanabilir uyumluluk ile yay-sönümleyici sistemi gibi davranmasını sağlamak.

## Dosyalar

- Betik: `src/b1_impedance_controller.py`
- Uyumluluk demosu: `src/b2_compliance_demo.py`
- Testler: `tests/test_impedance.py`

## Teori

### Empedans Kontrol Yasası

Empedans denetleyicisi, Kartezyen yay-sönümleyici davranışını eklem torklarına eşler:

```
F = K_p · (x_d - x) + K_d · (ẋ_d - ẋ)
τ = J^T · F + g(q)
```

Burada:
- K_p: Kartezyen rijitlik (N/m veya Nm/rad)
- K_d: Kartezyen sönümleme (N·s/m veya Nm·s/rad)
- J: LOCAL_WORLD_ALIGNED çerçevesinde Jacobian
- g(q): yerçekimi kompanzasyonu

### 6-DOF Genişletme

Tam poz kontrolü için (konum + oryantasyon):

```
F_6d = [K_p_lin · e_pos; K_p_rot · e_rot] + [K_d_lin · ė_pos; K_d_rot · ė_rot]
τ = J_6d^T · F_6d + g(q)
```

Oryantasyon hatası antisimetrik çıkarma ile hesaplanır:
```
e_R = 0.5 · vee(R_d^T · R - R^T · R_d)
```

### Uyumluluk Ayarı

| K_p (N/m) | Davranış | Sapma (40N) |
|-----------|----------|-------------|
| 100 | Yumuşak | 104 mm |
| 500 | Orta | 43 mm |
| 2000 | Sert | 17 mm |

## Sonuçlar

- **Konum takibi**: Kartezyen hedeflere < 1mm hata
- **Oryantasyon takibi**: < 1° hata
- **Pertürbasyon geri kazanımı**: dış kuvvet kaldırıldığında hedefe döner
- **Rijitlik ölçeklemesi**: sapma K_p ile ters orantılı

## Nasıl Çalıştırılır

```bash
python3 src/b1_impedance_controller.py
python3 src/b2_compliance_demo.py
```
