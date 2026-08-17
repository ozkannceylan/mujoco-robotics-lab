# Lab 8 — Mimari

Tork kontrollü bir Unitree G1 üzerinde tüm vücut loko-manipülasyon: parçaların
nasıl birleştiği ve her birinin neden bu biçimde olduğu. Kilometre taşı bazlı
anlatım için [`../tasks/LESSONS.md`](../tasks/LESSONS.md), kodun rehberli
okuması için [`../docs/CODE_WALKTHROUGH.md`](../docs/CODE_WALKTHROUGH.md)
dosyalarına bakın. Bu belge [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md)
dosyasının Türkçe karşılığıdır.

---

## Tek paragraflık özet

Pinocchio modeli hesaplar — kütle matrisi, KM (kütle merkezi) Jacobian'ı,
sentroidal momentum matrisi, çerçeve Jacobian'ları. Bir kuadratik program (QP)
her milisaniyede bir, eklem ivmelerini **ve** temas kuvvet-momentlerini birlikte
çözer; kısıtlar: tahrik edilmeyen taban dinamiği, basan ayaklar, sürtünme,
basınç merkezi (CoP) sınırları ve tork limitleri. Eklem torkları QP'nin tahrikli
satırlarından okunur ve MuJoCo'ya uygulanır. Bunun üstündeki her şey — yürüme,
uzanma, taşıma — QP'nin maliyet fonksiyonunda hangi **görevlerin** bulunduğu ve
onlardan ne istendiği meselesidir.

```
yürüyüş planı (DCM)   el hedefleri   yük
        │                  │           │
        ▼                  ▼           ▼
   görev yığını:  DCM · salınım ayağı · pelvis · momentum · eller · duruş
        │        (Pinocchio, LOCAL_WORLD_ALIGNED, J̇q̇ sürüklenmesi + ileri besleme)
        ▼
   tüm vücut ters dinamik QP  (OSQP, ~0.07 ms)
      değişkenler:  q̈ (35)  +  temas kuvvet-momentleri f (basan ayak başına 6)
      eşitlikler:   tahrik edilmeyen taban satırları · basan ayakların sabitlenmesi
      eşitsizlikler: sürtünme piramidi · CoP ayak içinde · f_z ≥ f_min · |τ| ≤ τ_maks
        ▼
   τ = M[6:]q̈ + h[6:] − J_cᵀ[6:] f     →  MuJoCo, tork aktüatörleri, 1 kHz
```

---

## Neden ivme seviyesinde bir QP

Bu, laboratuvarın en önemli yapısal kararıdır ve tercihle değil ölçümle
verilmiştir.

Önce bir hız seviyesi ("kinematik") QP yazıldı — `min ‖J q̇ − ẋ_hedef‖²` — ve
`wb_qp.py` içinde "denge için uygun değil" etiketiyle duruyor. Yüzen tabanlı bir
robotu dengeleyemez ve bunun sebebi ayar değil: **KM hareketi temas kuvvetleri
tarafından üretilir; kinematik bir QP ise bu kuvvetleri hiç temsil etmez.** Böyle
bir QP, robot devrilirken `J_com q̇ = 0` denklemini tam olarak sağlayabilir. G1
üzerinde ölçülen ele veren belirti: **el** görevi güçlendirildikçe robot **daha
erken** düşüyordu.

Temas kuvvet-momentlerini karar değişkeni yaparak ivme seviyesinde çözmek, "KM'yi
ayakların üstünde tut" isteğini çözücünün gerçekten uygulayabileceği bir kısıta
dönüştürür — çünkü bunu yapacak kuvvetler artık onun sahip olduğu değişkenlerdir.

Bundan laboratuvar boyunca iki sonuç doğar:

- **Taban satırları bir komut değil, bir kısıttır.** `M q̈ + h = Sᵀτ + J_cᵀ f`
  denkleminin ilk altı satırında hiçbir aktüatör yer almaz. Bunlar `(q̈, f)`
  üzerinde bir eşitlik olarak girer; kalan 29 satır τ'yu **tanımlar** ve τ
  sonradan okunur.
- **Temas geometrisi kontrolcünün parçasıdır.** CoP eşitsizliği, robotun
  ayaklarının matematiğe girdiği yerdir; bu geometriyi yanlış yazmak bir kontrol
  hatasından ayırt edilemez — aşağıdaki "Temas modeli" bölümüne bakın.

## Neden DCM, neden bir KM yörüngesi değil

Doğrusal ters sarkaç altında KM `c̈ = ω²(c − p)` bağıntısına uyar; burada
`ω = √(g/z_c)` ve `p` ZMP'dir. Bu şu şekilde ayrışır:

```
ξ = c + ċ/ω        ıraksak       ξ̇ = ω(ξ − p)      ← kararsız, yönlendirilmeli
η = c − ċ/ω        yakınsak                         ← kararlı, kontrol gerektirmez
```

Yalnızca `ξ` kaçabilir ve o da QP'nin destek poligonunun her yerine
yerleştirebildiği ZMP ile yönlendirilir. Bu yüzden kontrolcü ıraksak bileşeni
komutlar ve KM'nin serbestçe ilerlemesine izin verir.

Alternatif — KM **konumunu** komutlamak — yerinde adımlama için çalışır ve
yürüme için kanıtlanmış biçimde çalışmaz: her ayağın üstünde bir an dinlenmeyi
gerektirir, ileri yürüyüş ise bunu asla sunmaz. Ölçüldüğünde, yarı-statik
referans denenen her adım uzunluğu, çift destek süresi ve KM ofsetinde 10 adımın
3'üne ulaştı.

`dcm_planner.py` ayak izleri boyunca parçalı doğrusal bir ZMP planlar ve
`ξ̇ = ω(ξ − p)` denklemini bir uçtaki dinlenme koşulundan **geriye doğru**
integre eder. Geriye doğru integrasyon tek kararlı yöndür: kararsız bir sistemin
ileri integrasyonu sınır hatasını `e^{ωT}` ile büyütür, geri özyineleme ise
`e^{−ωT}` ile küçültür.

`wb_tasks.DCMTask` ardından şunu komutlar:

```
p_cmd  = ξ − ξ̇_ref/ω + (k/ω)(ξ − ξ_ref)          basan ayakların içine kırpılır
c̈_des = ω²(c − p_cmd)  =  −ω·ċ + ω·ξ̇_ref − ω·k·(ξ − ξ_ref)
```

Neyin **olmadığına** dikkat edin: KM'yi komutlanan bir konuma çeken hiçbir terim
yok.

## Sentroidal momentum görevi neden var

Yürüme sentroidal bir problemdir ve KM Jacobian'ı kolları da içerir; dolayısıyla
bir el görevi bağımsız bir ekleme değildir — QP'nin el hedefini sağlamak için
ivmelendirdiği her kilogram, robotu ayakta tutan büyüklüğün içine düşer.

Hiçbir el görevi ağırlığı hem yürüyüp hem takip edemedi. Başarısızlıklar
**monoton değildi** (ağırlık 1e1: 46 mm sarkmayla yürüdü, 2e1: 5. adımda düştü,
1e2: 7. adımda, 3e2: 5. adımda) — bu, yanlış ayarlanmış bir terimin değil, eksik
bir terimin imzasıdır.

`CentroidalAngularMomentumTask`, `L = A_g(q) q̇` büyüklüğünü düzenler ve QP'nin
"kollar hareket edebilir ama gövdeyi döndüremez" diyebilmesini sağlar — kolları
momentum sönümleyicisi görevi de gören bir eklem uzayı duruş çekmesiyle
sınırlamak yerine. 7. adımda düşen aynı el görevi, bu terimle tüm mesafeyi üç kat
daha iyi takiple yürüdü.

Zor yoldan öğrenilen iki kapsam kuralı:

- Bu bir **kol görevi yardımcısıdır, küresel bir dengeleyici değil.** Sade bir
  yürüyüş boyunca etkinleştirildiğinde, yürümenin kendisinin ürettiği açısal
  momentumu (yürüyüş ±2 kg·m²/s yuvarlanma üretir) iptal eder ve robotu 2. adımda
  yere serer.
- Referansı, **tutulan** bir duruş için sıfırdır; kütleyi kasten hareket ettiren
  bir görev için ise `L_ref`'tir (çözümlenmiş momentum kontrolü, Kajita ve ark.
  2003) — aksi hâlde terim, mümkün kılmak için eklendiği yörüngeyle savaşır.

## Temas modeli

QP'nin CoP satırları ayağı tarif eder. Menagerie'nin G1 tabanı, ayak bileği
yuvarlanma çerçevesinde x ∈ {−0.05, 0.12}, y ∈ {±0.025, ±0.03}, z = −0.03
konumlarındaki dört temas küresidir; yani dürüst tarif, yarı-uzunluğu 0.085 olan
ve çerçevenin **0.035 m önünde** merkezlenmiş, 0.035 m altında oturan bir
yamadır.

Daha önceki, ayak bileğinde merkezlenmiş simetrik ±0.08 m kutu aynı anda üç
şekilde yanlıştı ve her biri önemliydi:

| hata | sonuç |
|---|---|
| geriye doğru 30 mm fazla CoP iddia etti | QP, MuJoCo'nun üretmeyi reddettiği kuvvet-momentler planladı — sabit bir kuvvet hatası |
| ileriye doğru 40 mm CoP'yi attı | KM'yi ayak basmadan önce yavaşlatan yetkiyi çöpe attı |
| çerçevenin yerden yüksekliğini yok saydı | `CoP = −m_y/f_z` yalnızca sıfır kayma kuvvetinde doğrudur; yürüme tam da kaymanın sıfır olmadığı andır |

Yama ofseti ve `h·f` kayma terimiyle
(`CoP_x = (−m_y − h·f_x)/f_z`), komutlanan-gerçekleşen KM ivmesi ilişkisi
eğim 0.78 ve −0.09 m/s² sapmadan, eğim 0.95 ve 0.995 korelasyona geçti.

Ayakta durmak bu modelleri birbirinden ayıramaz. Yürüme ise her adımda ayağın iki
ucunu da kullanır.

---

## Modüller

| Dosya | Rolü |
|---|---|
| `g1_torque_model.py` | Menagerie'den `MjSpec` ile tork kontrollü G1'i kurar: 29 `<position>` servosu → `<motor>`, ctrlrange her eklemin `actuatorfrcrange` değerinden, anahtar kare ctrl sıfırlanır, zemin + ışık eklenir |
| `lab8_common.py` | Yollar, sabitler, yükleyiciler, MuJoCo↔Pinocchio durum dönüşümü, LIPM ilkelleri, KM/temas/destek poligonu yardımcıları, yük iliştirme |
| `standing_controller.py` | Eklem PD + seçilebilir yerçekimi modu; yalnızca QP devralmadan önce robotu oturtmak için kullanılır |
| `wb_tasks.py` | Görev tanımları ve yığın: çerçeve konumu/pozu/yönelimi, KM, **DCM**, **sentroidal açısal momentum**, duruş |
| `wb_id_qp.py` | Kontrol yolu: temas kuvvet-momentli, ivme seviyesinde ters dinamik QP |
| `wb_qp.py` | Hız seviyesi QP — yalnızca kinematik alt problemler, denge **değil** |
| `gait_planner.py` | Faz zaman çizelgesi, temas kümeleri, ileri beslemeli salınım referansları, ayak izi yerleşimi (`step_length`, `step_width`, `first_swing`, `close_stance`) |
| `dcm_planner.py` | Ayak izleri boyunca ZMP + ondan doğan DCM yörüngesi |
| `locomotion_controller.py` | Yürüyüş → QP bağlantısı: ölçülmüş temasla basan ayak kümesi, salınım rampası, ZMP kırpma, telemetri |
| `capstone_scene.py` | M5 sahnesi: kaideler, serbest eklemli yük, canlı bağıl poz yakalamalı iki weld kavrama |
| `mN_*.py` | Kilometre taşı başına bir çalıştırılabilir gate demosu; her biri kendi kanıtını `media/` altına yazar |

Tork modeli **çalışma zamanında üretilir, depoya işlenmez**: Menagerie tek doğru
kaynağı olarak kalır ve `export_xml()` inceleme için bir anlık görüntü üretebilir.

---

## Veri akışı, bir kontrol adımı

```
mj_data.qpos, qvel
   │  mj_state_to_pin — pelvis z ofseti, kuaterniyon sırası, taban hızı dünya→gövde,
   │                     robota dilimlenir (bir sahne serbest cisimler ekleyebilir)
   ▼
q (36), v (35)
   │  TaskStack.update_dynamics — sıfır ivmeyle FK, böylece raporlanan her
   │                               çerçeve/KM ivmesi J̇q̇ sürüklenmesinin ta kendisidir
   ▼
Jacobian'lar · sürüklenmeler · A_g · J_com
   │  her görev: desired_acceleration = ẍ_ref + k_p·e + k_d·(ẋ_ref − ẋ)
   ▼
WholeBodyIDQP.solve  →  q̈, f, τ
   │
   ▼
mj_data.ctrl = τ   →   mujoco.mj_step
```

Kinematik, görev başına değil **adım başına bir kez** hesaplanır: 35 serbestlik
dereceli bir model üzerinde altı görevle, tekrarlanan geçişler adımın süresine
hâkim olurdu.

### Pazarlığa kapalı kurallar

- Jacobian'lar `pin.LOCAL_WORLD_ALIGNED`'dır — öteleme satırları dünya
  hizalıdır, böylece dünya çerçevesindeki bir konum hatası doğrudan onlara
  eşlenir.
- Pinocchio'nun dünyası MuJoCo'nunkinin `PELVIS_MJCF_Z` = 0.793 m altındadır.
  Görev hedefleri **MuJoCo dünya koordinatlarında** verilir ve içeride
  dönüştürülür; böylece çağıranlar iki çerçeveyle boğuşmaz.
- Yüzen tabanda `nq ≠ nv`'dir (36'ya 35). Konfigürasyon güncellemeleri
  `pin.integrate` üzerinden gider, asla `q += dq` ile değil.
- Her görev Jacobian'ı `tests/test_wb_tasks.py` içinde sonlu farklarla
  doğrulanır.

---

## Yürüyüş planı

`GaitSchedule` `DS → SS → DS → SS …` dizisini kurar ve herhangi bir `t` için
temas kümesini, salınım ayağı referansını (konum, hız, ivme) ve faz indeksini
döndürür. `DCMPlan` aynı zaman çizelgesini bir ZMP'ye ve ondan doğan DCM yayına
çevirir.

Davranışın büyük kısmını üç parametre taşır:

- **`step_width`** — baskın yürüyüş parametresi. ZMP her adımda ayaklar arasında
  geçiş yapar ve yanal DCM onunla salınır; dolayısıyla yanal dengenin bedeli
  ayakların ne kadar açık olduğuyla belirlenir. G1'in 0.237 m'lik dinlenme duruşu
  12 adımın 7'sini ve bir düşüşü verir; 0.18 m ise 12'de 12 verir.
- **`first_swing`** — **arkadaki** ayakla adım at. Düz bir duruştan bu, "her
  zaman sol" ile eşdeğerdir; bir yürüyüşü adım ortasında sürdürürken ise yürümek
  ile öndeki ayağı zaten bastığı yere yeniden basmak arasındaki farktır.
- **`close_stance`** — yürüyüşü ayaklar bitişik hâlde bitir, gerçek bir yürüyüşün
  bittiği gibi; böylece **sonraki** yürüyüş kontrolcünün gördüğü bir duruştan
  başlar.

Son ikisi yalnızca birden fazla kez yürüyen bir dizide gözlenebilir — M3 ve M4
boyunca gizli kaldılar ve capstone'da ortaya çıktılar.

---

## Manipülasyon ve yük

Kavrama bir MuJoCo `mjEQ_WELD` kısıtıdır; brief buna izin verir ("kavrama BASİT
kalsın"): buradaki G1 modelinin eli yoktur ve Lab 5 gerçek bir paralel çeneli
kavramayı zaten doğrulamıştır. Lab 8'in sınadığı şey, **tüm vücut**
kontrolcüsünün kütle alma, taşıma ve bırakmayı atlatıp atlatamadığıdır.

Weld'lerle ilgili yanlış yapılması kolay ve ikisi de bir düşüşe mal olan iki şey:

- Bir weld, **derleme zamanındaki** bağıl pozunu korur. `eq_active` bir
  anahtardır, "burayı kavra" talimatı değil — saf biçimde etkinleştirmek 0.42 m'lik
  bir sıçrama komutladı. `CapstoneScene.set_weld` önce canlı el→yük dönüşümünü
  `eq_data[3:10]` içine yazar ve bir boşluğun üstünden weld kapatmayı reddeder.
- İki weld, QP'nin modellemediği **kapalı bir kinematik zincir** oluşturur. Bu
  yüzden capstone tek elle alır, iki elle taşır ve bırakmadan önce ikinci weld'i
  açar.

Kavrama anında `attach_payload_to_pinocchio`, yükün eylemsizliğini bileğin ana
eklemine katar ve taze bir `pin.Data` döndürür. Çerçeve kimlikleri, `nq` ve `nv`
değişmez — yük, bir serbestlik derecesi değil mevcut bir ekleme eylemsizlik
ekler — dolayısıyla görevler ve QP boyutları dokunulmadan kalır; yalnızca `M`,
`J_com` ve `A_g` değişir. Ardından yürüyüş **yeniden planlanır**, çünkü bir plan
belirli bir konfigürasyonu tarif eder ve kütle almak o konfigürasyonda bir
değişikliktir.

---

## Çözücü ayarları

OSQP, adımlar arası sıcak başlatmalı; temas kümesinin biçimi değiştiğinde yeniden
kurulur.

`tolerance = 1e-4`, `max_iterations = 2000`. Önceki `1e-6`, problemin
koşullanmasının verebileceğinin altındaydı — 1e4…1e1 aralığındaki görev
ağırlıklarına karşılık 1e-4'lük bir düzenlileştirme — ve kontrol adımlarının
%38'i çözüm başına 12.6 ms ile `maximum iterations reached` döndürüyordu. `1e-4`
ile her adım ~25 iterasyonda ve **0.073 ms**'de yakınsıyor, üstelik taban
dinamiği kısıt artığı 0.021'den 8.5e-5 N·m'ye **düşüyor**. Daha az doğruluk
istemek daha doğru bir cevap üretti.

İterasyon sınırına çarpmak bir performans notu değil, bir doğruluk uyarısıdır:
dönen nokta, çözücünün o an nerede olduğudur.

---

## Laboratuvarlar arası bağımlılıklar

Lab 7, G1 kurallarını sağlar — eklem sıralaması, qpos/qvel dilimleri, pelvis MJCF
ofseti, kuaterniyon dönüşümleri — ve bunlar `lab8_common` üzerinden yeniden
dışa aktarılır ki alt modüller tek bir ad alanı içe aktarsın.

Yabancı laboratuvarlar `sys.path`'e **`append` ile eklenir, asla `insert(0)`
ile değil**. Laboratuvarlar modül adlarını paylaşır (`standing_controller`,
`record_demo`) ve yabancı bir `src/` dizinini bu laboratuvarınkinin önüne koymak,
yerel modülleri sessizce başka bir laboratuvarın gerçeklemesiyle gölgeler.

Lab 9'un veri hattı bu kontrolcülere bağlıdır; yürüme ve taşıma rejimlerinin
burada **ne amaçlandığı** değil **ne ölçüldüğü** cinsinden belgelenmesinin sebebi
budur.
