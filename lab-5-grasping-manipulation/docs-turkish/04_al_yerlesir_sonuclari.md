# Al-Yerleştir Sonuçları — Lab 5

## Özet

Lab 5 al-yerleştir pipeline'ı, MuJoCo simülasyonunda UR5e + paralel çeneli tutucu kullanarak 40mm alüminyum küpü (150 g) A konumundan (dünya [0.35, +0.20, 0.335]) B konumuna (dünya [0.35, -0.20, 0.335]) başarıyla taşır.

---

## Sistem Konfigürasyonu

| Parametre | Değer |
|-----------|-------|
| Simülasyon zaman adımı | 1 ms (1 kHz) |
| Kol eklemleri | 6 (UR5e) |
| Tutucu eklemleri | 2 (aynalı → 1 aktüatörlü) |
| Kutu boyutu | 40 mm küp |
| Kutu kütlesi | 150 g |
| Tutucu aktüatör kp | 200 N/m |
| Eklem Kp (omuz/dirsek) | 200 N·m/rad |
| Eklem Kp (bilek) | 100 N·m/rad |

---

## Durum Zaman Çizelgesi

Her al-yerleştir döngüsü için yaklaşık zamanlama:

| Durum | Süre (sim adımları) | Açıklama |
|-------|---------------------|----------|
| IDLE | 1 | Başlangıç |
| PLAN_APPROACH | ~200–600 ms hesaplama | RRT* |
| EXEC_APPROACH | ~3000–8000 adım | TOPP-RA yörüngesi |
| DESCEND | ~500 adım | Kartezyen iniş |
| CLOSE | ~300 adım | Tutucu kapanır |
| LIFT | ~500 adım | Kartezyen yükseliş |
| PLAN_TRANSPORT | ~200–600 ms hesaplama | RRT* |
| EXEC_TRANSPORT | ~3000–8000 adım | TOPP-RA yörüngesi |
| DESCEND_PLACE | ~500 adım | Kartezyen iniş |
| RELEASE | ~300 adım | Tutucu açılır |
| RETRACT | ~3000–5000 adım | Q_HOME'a dön |

---

## Temel Sonuçlar

### IK Doğruluğu
DLS IK, tüm dört hedef konfigürasyonu için < 0.1 mm konum hatasına yakınsıyor. Üstten aşağı yönelim kısıtı tüm konfigürasyonlarda 1e-4 rad içinde sağlanıyor.

### Yol Planlaması
6000 iterasyonlu RRT*, tüm kavrama konfigürasyonları arasında çarpışmasız yolları güvenilir biçimde buluyor. Kısayol sonrasında yollar genellikle 5–15 ara nokta içeriyor.

### Temas Algılama
Kapamadan 10 simülasyon adımı içinde tutucu parmak geom'ları kutuyla temas kuruyor. 150 g'lık kutu, LIFT ve EXEC_TRANSPORT aşamalarında kayma olmadan güvenle tutuluyor.

### Yörünge Takibi
TOPP-RA parametreleştirmesi, tüm 6 eklemi hız ve ivme limitleri içinde tutuyor. Empedans kontrolörü + yerçekimi telafisi ile eklem takip hatası (‖q_d - q‖) yörünge boyunca 5 mrad'ın altında kalıyor.

---

## Bilinen Sınırlamalar

1. **Görsel geri besleme yok**: konfigürasyonlar IK ile çevrimdışı hesaplanıyor.
2. **Yalnızca sabit yönelim**: pipeline yalnızca üstten aşağı yaklaşımı destekler.
3. **Tek kutu**: çoklu nesne sahneleri desteklenmiyor.
4. **Kavrama kalite metriği yok**: sarma kapatması veya kuvvet elipsoidi değerlendirilmiyor.
5. **Yeniden kavrama yok**: LIFT sırasında kutu kayarsa durum makinesi kurtarma yapmıyor.

---

## Grafikler

`pick_place_demo.py` tarafından `media/` klasörüne kaydedilen grafikler:

- `media/ee_trajectory.png` — 3B EE yörüngesi (yaklaşım + taşıma yayları)
- `media/joint_tracking.png` — 6 eklem takip hatası vs. zaman
- `media/gripper_contact.png` — tutucu konumu + temas boolean'ı vs. zaman
- `media/state_timeline.png` — durum geçişleri zaman çizelgesi üzerinde
