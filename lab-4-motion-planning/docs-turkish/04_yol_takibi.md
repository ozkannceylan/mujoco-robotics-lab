# Yorunge Takibi

## Son Yurutme Yolu

Lab 4 yorunge takibi artik Lab 3 ile ayni kanonik yigin uzerinde calisir:

- MuJoCo Menagerie UR5e + monte Robotiq 2F-85
- Pinocchio ile `g(q)` yercekimi telafisi
- Menagerie aktuatore uygun torktan-kontrole esleme

## Kontrol Yasasi

```text
tau = Kp * (q_d - q) + Kd * (qd_d - qd) + g(q)
```

Varsayilan kazanclar:

- `Kp = 400`
- `Kd = 40`

## Onemli Fark

Eski Lab 4 yolu torklari dogrudan `mj_data.ctrl[:6]` icine yaziyordu. Kanonik
yiginda bu dogru degildir. Son uygulama, Lab 3'teki gibi Menagerie aktuatore
uygun tork eslemesini kullanir.

## Son Dogrulama

Standart capstone sahnesi:

- RMS takip hatasi: `0.0125 rad`
- son konum hatasi: `0.0016 rad`

Bloklu dogrulama sahnesi:

- RMS takip hatasi: `0.0124 rad`
- son konum hatasi: `0.0041 rad`
- dogrudan yol ozgur mu: `False`
