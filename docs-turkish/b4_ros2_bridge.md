# B4: ROS2 Bridge — Opsiyonel

## Amaç

MuJoCo simülasyonunu bir ROS2 node gibi kullanmak:

- `/joint_command` ile komut almak
- `/joint_state` ile joint durumunu yayınlamak
- `/ee_pose` ile end-effector pozisyonunu yayınlamak

## Eklenen İskelet

- `ros2_bridge/mujoco_bridge.py`
  - MuJoCo modelini yükler
  - command subscriber açar
  - her timer adımında `mj_step()` çalıştırır
  - `JointState` ve `Pose` yayınlar
- `ros2_bridge/commander.py`
  - örnek komut publisher

## Çalıştırma Mantığı

Terminal 1:

```bash
python3 ros2_bridge/mujoco_bridge.py
```

Terminal 2:

```bash
python3 ros2_bridge/commander.py
```

Gerçek ROS2 ortamında bunu daha temiz biçimde bir paket, `setup.py`, launch dosyaları ve mesaj bağımlılıkları ile paketlemek gerekir. Buradaki amaç entegrasyon akışını net bırakmaktır.
