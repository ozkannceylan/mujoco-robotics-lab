# Kavrama Pipeline'ı — Lab 5

## Genel Bakış

Lab 5, tam bir al-yerleştir pipeline'ı uygular: UR5e kolu, A konumundaki (dünya [0.35, +0.20, 0.335]) 40mm küpü alıp B konumuna (dünya [0.35, -0.20, 0.335]) yerleştirir. Pipeline şunları entegre eder:

- **Pinocchio**: FK ve Sönümlü En Küçük Kareler (DLS) IK
- **Lab 4 RRT\***: çarpışmasız hareket planlaması
- **Lab 4 TOPP-RA**: zaman-optimal yörünge parametreleştirmesi
- **Lab 3 empedans kontrolü**: yörünge takibi
- **Lab 3 yerçekimi telafisi**: kalıcı hal hatasını elimine etmek için ileri besleme

---

## Durum Makinesi

`GraspStateMachine` sınıfı 11 durumdan geçer:

```
IDLE
  → PLAN_APPROACH    (Q_HOME'dan q_pregrasp'a RRT*)
  → EXEC_APPROACH    (eklem-uzay empedans takibi)
  → DESCEND          (Kartezyen empedans inişi)
  → CLOSE            (tutucu kapanır, yerleşmeyi bekler)
  → LIFT             (Kartezyen empedans yükselişi)
  → PLAN_TRANSPORT   (q_pregrasp'tan q_preplace'e RRT*)
  → EXEC_TRANSPORT   (eklem-uzay empedans takibi)
  → DESCEND_PLACE    (Kartezyen empedans inişi)
  → RELEASE          (tutucu açılır)
  → RETRACT          (Q_HOME'a geri dön)
  → DONE
```

---

## IK Çözücü

**Dosya:** `src/grasp_planner.py :: compute_ik()`

**Sönümlü En Küçük Kareler (DLS)**:

```
Δq = α · Jᵀ (J Jᵀ + λ² I)⁻¹ · e
```

Parametreler:
- **J** (6×6): çerçeve Jakobiyen'i
- **e** (6,): görev-uzay hatası = [Δkonum, Δyönelim]
- **α = 0.5**: adım boyutu
- **λ² = 1e-4**: sönümleme (tekillik koruması)
- **tol = 1e-4**: yakınsama eşiği (‖e‖ normunda)

---

## Kavrama Konfigürasyonları

**Dosya:** `src/grasp_planner.py :: compute_grasp_configs()`

| Konfigürasyon | Tool0 hedefi (dünya, m) | Amaç |
|---------------|------------------------|------|
| `q_home` | — (doğrudan Q_HOME) | Dinlenme pozu |
| `q_pregrasp` | BOX_A + [0, 0, 0.240] | A kutusunun üstünde yaklaşım |
| `q_grasp` | BOX_A + [0, 0, 0.090] | Parmak uçları A kutusu merkezinde |
| `q_preplace` | BOX_B + [0, 0, 0.240] | B üzerinde taşıma hoveri |
| `q_place` | BOX_B + [0, 0, 0.090] | Parmak uçları B kutusu merkezinde |

`GRIPPER_TIP_OFFSET = 0.090 m` ve `PREGRASP_CLEARANCE = 0.150 m` sabitlerini kullanır.

---

## Hareket Planlaması (Lab 4 Entegrasyonu)

**Dosya:** `src/grasp_state_machine.py :: _plan_and_smooth()`

```python
# 1. RRT* çarpışmasız yol (6000 iterasyon)
path = rrt_planner.plan(q_start, q_goal, max_iter=6000, rrt_star=True)

# 2. Kısayol (200 iterasyon, gereksiz ara noktaları kaldırır)
path = shortcut_path(path, collision_checker, max_iter=200)

# 3. TOPP-RA zaman parametreleştirmesi
times, q_traj, qd_traj, _ = parameterize_topp_ra(path, VEL_LIMITS, ACC_LIMITS)
```

---

## Yörünge Yürütme (Lab 3 Entegrasyonu)

### Eklem-Uzay Empedans (EXEC_* durumları)

```python
τ = Kp(q_d - q) + Kd(qd_d - qd) + g(q)
```

- **Kp = diag([200, 200, 200, 100, 100, 100])** N·m/rad
- **Kd = 2√Kp** (kritik sönümlü)
- **g(q)**: Pinocchio RNEA yerçekimi vektörü

---

## Hata Modları

| Mod | Belirti | Neden | Çözüm |
|-----|---------|-------|-------|
| IK sapması | RuntimeError "IK başarısız" | Hedef erişilemez | GRIPPER_TIP_OFFSET işaretini kontrol et |
| RRT* zaman aşımı | RuntimeError "RRT* başarısız" | Dar geçit veya kötü limitler | max_iter artır |
| Kutu kayması | Durum makinesi LIFT'te donar | Düşük sürtünme | μ_kayma artır |
