# B2: PD Controller — Öğrenme Notu

## Kontrol Yasası

Bu adımda kullanılan temel kontrol:

```text
tau = Kp (q_des - q) + Kd (qd_des - qd) + g(q)
```

- `Kp`: pozisyon hatasına tepki
- `Kd`: hız hatasına damping
- `g(q)`: gravity compensation terimi

## Bu Repodaki Yaklaşım

`src/b2_pd_controller.py` iki katman içerir:

1. Saf Python fallback plant
   - deterministik ve bağımlılıksız
   - overshoot ve tracking hatasını hızlıca ölçmek için var
2. Opsiyonel MuJoCo doğrulama kancası
   - bağımlılıklar varsa gerçek torque modeline bağlanabilir

## Denenen Senaryolar

1. Sabit hedef açı
   - hedef: `q = [90°, -45°]`
   - gravity compensation açık/kapalı kıyaslanır
2. Cubic joint trajectory takibi
   - `B1`'de üretilen joint-space trajectory takip edilir
   - RMS tracking error ölçülür

## Çıktılar

- `docs/b2_fixed_target_no_gc.csv`
- `docs/b2_fixed_target_gc.csv`
- `docs/b2_tracking_no_gc.csv`
- `docs/b2_tracking_gc.csv`

`matplotlib` varsa ayrıca:

- `docs/b2_pd_controller.png`

## Beklenen Yorum

- PD controller hedefe gider ve durur
- gravity compensation açılınca steady-state hata ve tracking hatası azalmalıdır
- Bu fallback plant gerçek MuJoCo dynamics değildir; kontrol mantığını doğrulamak için kullanılır
