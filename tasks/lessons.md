# Lessons

> **Scope note (2026-08-13):** This file covers **Labs 1–2 only** (written before the
> per-lab convention existed). Lessons for Labs 3–7 live in each lab's
> `tasks/LESSONS.md`. Content below is still valid and worth keeping.

## Lab 1: 2-Link Planar Arm

- Hata: MuJoCo XML'de `range="-3.14 3.14"` yazıldı ama MuJoCo varsayılan olarak derece kullanır. Sonuç: ±3.14° (±0.055 rad) limit, büyük constraint kuvvetleri.
  - Kural: MuJoCo XML dosyalarında radyan kullanılacaksa `<compiler angle="radian"/>` zorunlu. Yoksa tüm açı değerleri derece olarak yorumlanır.

- Hata: Base platform ve joint visualizer geom'ları `contype=1` ile tanımlı. Bazı konfigürasyonlarda link1 ile çakışıp büyük constraint kuvvetleri üretiyor.
  - Kural: Dekoratif geom'lara `contype="0" conaffinity="0"` verin veya runtime'da `model.geom_contype[:] = 0` ile devre dışı bırakın.

## Lab 2: UR5e 6-DOF Arm

- Hata: Menagerie UR5e position servo'larında naif `ctrl = q_desired` yaklaşımı 133 mm RMS hata verdi.
  - Sebep 1: Yerçekimi düşmesi — PD servo yerçekimini pozisyon hatası üzerinden yenmek zorunda, kalıcı offset oluşuyor.
  - Sebep 2: Hız takip gecikmesi — sönümleme terimi (Kd*qvel) istenen harekete karşı direnç oluşturuyor.
  - Sebep 3: Aktüatör doyması — PD servo forcerange sınırını aşınca (150 Nm) robot takip otoritesini kaybediyor.
  - Kural: Position servo'lu modellerde `ctrl = q_des + qfrc_bias/Kp + Kd*qd_des/Kp` kullanın (gravity compensation + velocity feedforward). Bu sayede RMS 0.088 mm'ye düştü — 1500x iyileşme.

- Hata: IK çözücü masa ile çarpışan joint konfigürasyonları buluyor ama bunun farkında değil.
  - Sebep: IK çözücüler sadece kinematik hatayı minimize eder, sahne geometrisini bilmezler.
  - Kural: Her IK çözümünde `data.qpos = q_ik` ayarlayıp `mujoco.mj_forward()` çağırın ve `data.ncon > 0` kontrolü yapın. Çarpışma varsa hedef pozisyonu değiştirin veya Y-offset ekleyerek kolu masanın üzerinde tutun.
  - Örnek: Küp merkezi [0.4, 0.0, 0.50]'den [0.3, 0.2, 0.55]'e taşındığında tüm çarpışmalar ortadan kalktı.

- Hata: Geometric Jacobian hesaplamasında tüm UR5e eklemlerinin z-ekseni etrafında döndüğü varsayıldı (Frobenius error 3.09).
  - Sebep: UR5e URDF'inde eklem 2,3,4,6 RY ekseni, eklem 1,5 RZ ekseni kullanır — hepsi z-ekseni değil.
  - Kural: `model.joints[joint_id].shortname()` ile gerçek ekseni okuyun (RX, RY, RZ). Hardcoded z-axis varsayımı yapmayın.

- Hata: MuJoCo offscreen renderer 640px varsayılan framebuffer boyutundan dolayı 1280x720 render yapamıyordu.
  - Kural: lab_scene.xml'de `<visual><global offwidth="1920" offheight="1080"/></visual>` ekleyin.

- Bilgi: Menagerie UR5e aktüatör modeli: `general` type, `gainprm=[Kp,0,0]`, `biasprm=[0,-Kp,-Kd]`.
  - Aktüatör formülü: `tau = Kp*(ctrl - qpos) - Kd*qvel`
  - Kp = [2000, 2000, 2000, 500, 500, 500], Kd = [400, 400, 400, 100, 100, 100]
  - forcerange = [-150, 150] (eklem 1-3), [-28, 28] (eklem 4-6)

- Bilgi: Lab scene'deki masa yüzeyi z ≈ 0.41 m'de. Masa body'si pos=[0.4, 0, 0.19], table_top geom'u relative pos=[0, 0, 0.2], size (half-extents) = [0.4, 0.35, 0.02].
