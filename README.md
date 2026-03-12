# MuJoCo Robotics Crash Course Sandbox

2-link planar robot üzerinden kinematik, IK, trajectory generation, PD control ve entegre pipeline çalışmaları için küçük bir çalışma deposu.

## Neler Tamamlandı?

- `A1`: MuJoCo model kurulumu ve temel simülasyon kontrolü
- `A2`: Forward kinematics ve workspace analizi
- `A3`: Analitik/nümerik Jacobian ve singularity gözlemi
- `A4`: Analitik IK, `pinv`, `dls`
- `A5`: Dynamics temelleri, `qfrc_bias`, `qM`
- `B1`: Cubic, quintic ve cartesian trajectory generation
- `B2`: PD controller ve gravity compensation karşılaştırması
- `B3`: Full pipeline demo'ları
- `B4`: Opsiyonel ROS2 bridge iskeleti
- `B5`: README, test özeti ve VLA bağlantı notları

## Dizin Yapısı

- [`models/`](./models): MuJoCo XML robot modelleri
- [`src/`](./src): Her adımın Python implementasyonları
- [`docs/`](./docs): öğrenme notları, CSV çıktıları, test kayıtları
- [`ros2_bridge/`](./ros2_bridge): opsiyonel ROS2 entegrasyon iskeleti
- [`PROJECT-PLAN.md`](./PROJECT-PLAN.md): takip edilen ana plan

## Hızlı Çalıştırma

```bash
python3 src/a3_jacobian.py
python3 src/a4_inverse_kinematics.py
python3 src/b1_trajectory_generation.py
python3 src/b2_pd_controller.py
python3 src/b3_full_pipeline.py
```

## Test Özeti

Detaylı kayıt: [`docs/testing.md`](./docs/testing.md)

- `A3`: analitik vs numerik Jacobian farkı `1e-10`
- `A4`: 20 hedefte `pinv` ve `dls` başarı oranı `%100`
- `B1`: cubic/quintic sınır koşulları doğru, cartesian path sapması `0`
- `B2`: tracking RMS error `0.008507 rad` (`gravity compensation` açıkken)
- `B3`: pick-place RMS error `0.003990 m`, circle RMS error `0.002734 m`

## Humanoid VLA Bağlantısı

1. FK/IK bilgisi, VLA action space ile joint-space arasındaki dönüşümü anlamak için doğrudan temel sağlar.
2. Jacobian, joint velocity limitleri ve güvenlik kısıtlarını task-space üzerinden yönetmek için gereklidir.
3. Trajectory generation ve PD tracking, veri üretim pipeline'ında motion planning ve düşük seviye takip mantığını somutlaştırır.

## Not

Bu sandbox'ta `numpy`, `matplotlib`, `mujoco` ve `rclpy` görünmüyor. Kodlar mümkün olduğunca saf Python fallback içerir; gerçek MuJoCo/ROS2 entegrasyonu bağımlılıklar erişilebilir ortamda yeniden çalıştırılmalıdır.
