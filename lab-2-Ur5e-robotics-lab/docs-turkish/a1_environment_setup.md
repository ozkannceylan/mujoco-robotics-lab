# A1: Ortam Kurulumu

## Hedef

UR5e varliklarinin dogru yuklendigini, analitik model ile simulasyon modelinin ileri kinematikte ayni sonucu verdigini ve labin geri kalani icin gerekli bagimliliklarin hazir oldugunu dogrulamak.

## Dosyalar

- Script: `src/a1_model_setup.py`
- Ortak yardimcilar: `src/ur5e_common.py`
- MuJoCo wrapper'i: `src/mujoco_sim.py`
- MuJoCo sahnesi: `models/lab_scene.xml` (Menagerie `ur5e.xml` dosyasini include eder)
- URDF: `models/ur5e.urdf`
- Cikti snapshot'i: `docs-turkish/a1_environment_status.csv`

## Bagimlilik Siniri

Bu scriptin erken hata vermesi normaldir ve faydalidir. Lab boyunca kullanilan temel paketleri import eder:

- `numpy`
- `mujoco`
- `pinocchio`
- `matplotlib`

Bu importlardan biri eksikse once onu duzeltmek gerekir. Sonraki moduller ayni yiginin uzerine kurulur.

## Script Neleri Dogrular

Script bes kontrolu sirasiyla yapar:

1. **Python bagimlilik kontrolu**
   Temel Python paketlerinin import edilip edilemedigini ve surumlerini kontrol eder.
2. **Model dosya kontrolu**
   Menagerie MJCF sahnesinin ve URDF dosyasinin `models/` altinda oldugunu dogrular.
3. **MuJoCo kontrolu**
   Sahneyi yukler, temel model bilgisini yazdirir, eklemleri listeler, `Q_HOME` ayarlar ve simulasyonu ilerletir.
4. **Pinocchio kontrolu**
   URDF'yi yukler, joint ve frame bilgisini yazdirir, home konfigurasyonunda FK hesaplar.
5. **Capraz dogrulama**
   MuJoCo'daki `attachment_site` ile Pinocchio'daki `ee_link` pozisyonlarini birden cok konfigurasyonda karsilastirir. Gecme kosulu 1 mm icinde uyumdur.

## Bu Modul Neden Onemli

- `ur5e_common.py`, labin geri kalanina ortak bir sozlesme saglar: path'ler, joint isimleri, limitler ve referans konfigurasyonlari burada merkezilesir.
- `mujoco_sim.py`, ham MuJoCo dizi erisimini tek bir wrapper arkasina alir.
- FK karsilastirmasi projedeki ilk guclu dogruluk sinyalidir. Burada uyumsuzluk varsa A1 sonrasi tum moduller kuskulu hale gelir.

## Nasil Calistirilir

```bash
python3 src/a1_model_setup.py
```

## Neye Bakmali

1. URDF'deki `ee_link` ile MuJoCo sahnesindeki `attachment_site` eslesmesini netlestir.
2. `ur5e_common.py` icinde sonraki modullerin dayandigi sabitleri bul: `Q_HOME`, `Q_ZEROS`, joint limitleri, hiz limitleri, tork limitleri ve DH tablosu.
3. `mujoco_sim.py` icinde `UR5eSimulator` sinifinin durum, tork ve end-effector pozunu nasil sundugunu takip et.
4. FK capraz kontrolunu bir guven capasi gibi dusun. Bu asama gectiginde labin geri kalani daha saglam temele oturur.

## Sonraki Adim

A2'ye gecip UR5e kinematik zincirini DH perspektifinden incele; sonra bunu URDF ve MJCF tabanli gercek modellere karsi yorumla.
