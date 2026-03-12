# B1: Trajectory Generation — Öğrenme Notu

## İki Temel Yaklaşım

1. Joint-space trajectory
   - joint açıları doğrudan interpolate edilir
   - basit ve ucuzdur
   - end-effector genelde düz çizgi izlemez
2. Cartesian trajectory
   - end-effector hedefleri doğrudan interpolate edilir
   - her örnekte IK çözmek gerekir
   - end-effector path'i düz çizgiye daha yakındır

## Bu Repoda Ne Var?

`src/b1_trajectory_generation.py` şu parçaları içerir:

- cubic joint-space trajectory
- quintic joint-space trajectory
- cartesian straight-line interpolation + analitik IK branch seçimi
- CSV çıktı üretimi
- `matplotlib` varsa path ve profile plot kaydı

## Sınır Koşulları

Cubic için:

```text
q(0) = q0
q(T) = qf
qd(0) = 0
qd(T) = 0
```

Quintic için:

```text
q(0) = q0
q(T) = qf
qd(0) = qd(T) = 0
qdd(0) = qdd(T) = 0
```

Bu yüzden quintic profil cubic'e göre daha smooth'tur.

## Çıktılar

Script aşağıdaki dosyaları üretir:

- `docs/b1_cubic_joint_traj.csv`
- `docs/b1_quintic_joint_traj.csv`
- `docs/b1_cartesian_traj.csv`

`matplotlib` varsa ayrıca:

- `docs/b1_trajectory_paths.png`
- `docs/b1_trajectory_profiles.png`

## Beklenen Yorum

- cubic trajectory başta ve sonda sıfır hız vermelidir
- quintic trajectory başta ve sonda sıfır hız ve sıfır ivme vermelidir
- cartesian trajectory düz çizgiye çok yakın kalmalıdır
- joint-space trajectory aynı başlangıç ve bitiş noktaları arasında görünür bir eğri çizer
