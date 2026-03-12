# A3: Jacobian — Öğrenme Notu

## Jacobian Nedir?

Jacobian, joint-space hızlarını end-effector hızına çeviren yerel lineer dönüşümdür:

```text
x_dot = J(theta) * theta_dot
```

- `theta_dot = [theta1_dot, theta2_dot]`
- `x_dot = [vx, vy]`
- `J(theta)` 2x2 matristir ve FK'nin açılara göre türevinden gelir.

## 2-Link Planar Analitik Jacobian

End-effector için etkin ikinci link uzunluğu:

```text
L2_eff = L2 + 0.015
```

Çünkü modelde `end_effector` site'ı link ucundan 1.5 cm ileridedir.

```text
J = | -L1*sin(theta1) - L2_eff*sin(theta1 + theta2)   -L2_eff*sin(theta1 + theta2) |
    |  L1*cos(theta1) + L2_eff*cos(theta1 + theta2)    L2_eff*cos(theta1 + theta2) |
```

Yorum:
- `J[0,0] = dx / dtheta1`
- `J[0,1] = dx / dtheta2`
- `J[1,0] = dy / dtheta1`
- `J[1,1] = dy / dtheta2`

## Doğrulama Stratejisi

`src/a3_jacobian.py` üç doğrulama yolu içerir:

1. Analitik Jacobian ile finite-difference Jacobian karşılaştırması
2. `det(J)` sweep'i ile singularity gözlemi
3. MuJoCo kuruluysa `mj_jacSite` ile karşılaştırma

Determinant yorumu:
- `det(J) -> 0` ise kol singularity'ye yaklaşıyor
- 2-link için tipik singularity durumları: `theta2 ~= 0` ve `theta2 ~= pi`

## Üretilen Çıktı

- `docs/a3_det_sweep.csv`
  - sabit `theta1` için `theta2` boyunca determinant verisi
  - grafik üretmek istersen sonradan bu CSV kolayca çizilebilir

## Bu Orturumdaki Kısıt

Çalışma ortamında şu anda `numpy`, `matplotlib` ve `mujoco` kurulu değil. Bu yüzden:
- kod standart kütüphane ile yazıldı
- MuJoCo karşılaştırması korumalı olarak eklendi
- tam simülasyon doğrulaması bağımlılıklar kurulduğunda yeniden çalıştırılmalı
