# Lab 9 — Mimari

Yürüyen bir insansı robotu süren, dile koşullanmış bir politika: parçaların
nasıl birleştiği ve her birinin neden bu biçimde olduğu. Kilometre taşı bazlı
anlatım [`../tasks/LESSONS.md`](../tasks/LESSONS.md), kodun rehberli okuması
[`../docs/CODE_WALKTHROUGH.md`](../docs/CODE_WALKTHROUGH.md) dosyasında. Bu belge
[`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) dosyasının Türkçe
karşılığıdır.

---

## Tek paragraflık özet

Dondurulmuş bir CLIP metin kulesi, talimatı 512 boyutlu bir vektöre çevirir. Bir
ACT politikası bu vektörü, iki adet 128 piksellik ego-merkezli kamera görüntüsünü
ve bir propriyosepsiyon vektörünü alır ve 10 Hz'de yirmi adımlık bir eylem öbeği
üretir. Bu eylemler eklem torku da değildir, eklem açısı da: bunlar Lab 8'in tüm
vücut ters dinamik QP'sinin **zaten tükettiği referanslardır** — el hedefleri, bir
yürüyüş komutu, bir kavrama biti. QP altta 1 kHz'de çalışır ve robotu ayakta
tutar. Politika **ne yapılacağına**, Lab 8 **nasıl düşülmeyeceğine** karar verir.

```
      "kırmızı bardağı al"
              │
     ┌────────▼─────────┐   dondurulmuş; her talimatın gömülmesi checkpoint'in
     │ CLIP metin kulesi│   içine yazılır, böylece değerlendirme ne transformers
     └────────┬─────────┘   ne de ağ erişimi ister
              │ 512
kafa kamerası ┐   │
bilek kamerası┤   │    ┌──────────────────────────────┐
   (128²)     ├───┴───►│  ACT politikası ~15.8M param │
propriyosepsiyon┘      │  ResNet18, yalnız layer4     │
    (62-d)             │  kamera başına 16 uzamsal jeton│
                       │  + durum + talimat jetonu    │
                       │  → çıkarım başına 20 eylem   │
                       └───────────────┬──────────────┘
                                       │ 10 Hz, 9 boyutlu eylem
                       ┌───────────────▼──────────────┐
                       │  Lab 8 tüm vücut ID QP       │   1 kHz
                       │  denge sert bir kısıttır     │
                       └───────────────┬──────────────┘
                                       │ τ (29)
                                  MuJoCo G1
```

---

## Politika neden eklem hedefi üretmiyor

`plan/LAB_09.md` eylem uzayını *"tüm tahrikli serbestlik dereceleri için eklem
konum hedefleri"* olarak belirtir. Sabit tabanlı bir kol için bu doğrudur ve
`humanoid_vla` bunu yapar. Yüzen tabanda ise Lab 7'nin zaten ölçtüğü şeye çarpar:
PD ile izlenen bir eklem konum referansı bu robotu **dengeleyemez**. Lab 7'nin ZMP
yürüyüşünün altı denemeden sonra başarısız olmasının ve Lab 8'in var olmasının
sebebi budur.

Bu yüzden birincil kafa, Lab 8'in QP'sinin tükettiğini üretir; brief'in birebir
kafası ise bir **ablasyon** olarak korunur. İkisi de eğitilir, ikisi de ölçülür.
Lab 7'den Lab 8'e ve buraya uzanan yay, ancak koşturularak kapanır — alıntılanarak
değil.

`task` kafası, 9 boyut, pelvisin yalnız-sapma (yaw-only) çerçevesinde:

| dilim | anlamı |
|---|---|
| `0:3` | sağ el hedefi |
| `3:6` | sol el hedefi |
| `6` | yürüyüş komutu — bir yürüyüş birimi at ya da dur |
| `7` | sağ kavramayı kapat |
| `8` | sol kavramayı kapat |

**Tam pelvis dönüşü değil, yalnızca sapma.** Pelvis yürürken sürekli yunuslama ve
yalpalama yapar; bunu bir el hedefine katmak, politikanın sabit tutması gereken
bir büyüklüğün içine yürüyüş salınımı enjekte ederdi. Hedefi pelvise göre ifade
etmek ise aynı uzanmayı, yürüyüşün neresinde olursa olsun aynı eylem yapan şeydir.

---

## Politikanın bilmesine izin verilmeyen şey

`state` 62 sayıdır: 29 eklem konumu, 29 eklem hızı, pelvis yüksekliği, pelvis
yalpalaması, pelvis yunuslaması ve kavrama biti. Pelvisin **dünya x, y ve
sapması** dışarıda bırakılmıştır ve bu dışarıda bırakma, bir şey ölçen bir
değerlendirme ile hiçbir şey ölçmeyen bir değerlendirme arasındaki farktır.

Kendi dünya koordinatları verilen bir politika buradaki her görevi ölü hesapla
çözebilir — `x > 0.25` olana kadar yürü, sabit bir ofsete uzan — bir görüntüye
hiç bakmadan ve talimatını hiç okumadan. Ne görüyü ne dili öğrenmiş olur, ama
mükemmel bir başarı oranı yazdırır. Politikanın nerede olduğuna dair bildiği her
şey piksellerden gelmek zorundadır.

Geriye kalan tam olarak gerçek bir robotun dış donanım olmadan gözlediği şeydir:
eklem enkoderleri ve yükseklik, yalpalama, yunuslama için bir IMU. Kısıt keyfî
değil fizikseldir ve `tests/test_scene_and_contract.py`, robotu dünyada
ötelemenin durumu değiştirmediğini doğrular.

---

## Sahne neden iki nesne taşıyor

Yalnızca görev etiketlerine koşullanmış bir politika, görevi robotun kendi
duruşundan çıkarabilir — yürümek ile uzanmak birbirine hiç benzemez — ve dili
tümüyle yok sayabilir. Böyle ölçülen her başarı oranı, sahne hakkında bir
ifadedir.

Bu yüzden kaide üzerinde bir **kırmızı bardak** ve bir **mavi kutu** durur,
hangisinin daha yakın olduğu tohum başına rastgeleleştirilir ve talimat hedefi
adlandırır. Aynı görüntü, farklı talimatlar altında farklı eylem gerektirir. Bu,
aksi hâlde mümkün olmayan üç şeyi mümkün kılar:

- **talimat takası testi** — aynı başlangıç durumu, öbür nesnenin cümlesi,
  davranış onu takip ediyor mu;
- gerçekten dile koşullanmış bir *yürüme* görevi, çünkü adlandırılan nesne ne
  kadar yürüneceğine karar verir (iki adım ya da dört);
- brief'in capstone cümlesi olan *"kırmızı bardağı al"*, oradaki tek nesnenin
  etiketi olarak değil, tam anlamıyla.

---

## Uzman ve görev kümesinin neden iki görev olduğu

Gösterimler Lab 8'in tüm vücut kontrolcüsünden, değiştirilmeden gelir.
`expert.VLAExpert`, Lab 8'in `Capstone` sınıfını türetir ve her faz metodunu
miras alır; yalnızca sahne, hedef seçimi ve bir gözlem yakalama kancası farklıdır.

Görev kümesi brief'in üç-beş görevi değil, **`walk` ve `pick`**'tir ve sebep
modelde değil uzmandadır. Lab 8'in capstone kapısı tek bir konfigürasyonda 4/4;
rastgeleleştirilmiş iki nesneli bir sahnede aynı dizi **1/8** aldı. Görev başına
ölçüm:

| görev | ölçülen | mekanizma |
|---|---|---|
| `walk` + `pick` | **40/40** | — |
| `carry` | 1/12 | `carry_targets` kavramayı yükün etrafında aynalar; bu kavrama ofsetinde iki bilek hedefi 22–35 mm arayla çıkar, yani iki bilek de neredeyse aynı noktaya isteniyor |
| `place` | 5/10 | el görevleri yalnız konumu denetler, bu yüzden nesne bileğin o an sahip olduğu eğimle (ölçülen 22°) bırakılır ve işaretten yuvarlanır |

Uzmanı bölümlerinin yarısında düşen bir gösterim kümesi, bir politikaya düşmeyi
öğretir; hiçbir model çalışması bunu telafi etmez. `place` görevini geri
getirmek Lab 8'in yığınında bir el **yönelim** görevi ister — yani Lab 8 işi.

### Ayakta durma kararlılık bütçesi

Lab 8'in `_freeze_balance` metodu, DCM hedefini fazın başladığı andaki değerine
sabitler. Bu kısa bir hareket için doğru, uzun bir hareket için yanlıştır: bir
kolu hareket ettirmek kütle merkezini kaydırır ve dondurulmuş hedef, robotu artık
bir dinlenme konfigürasyonunu tarif etmeyen bir anlık görüntüye doğru komutlar.

Lab 8 buna hiç çarpmadı, çünkü manipülasyon fazları arasında **yürüyordu** ve
yürüme DCM'i sıfırdan yeniden planlar. Lab 9'un taşıma yürüyüşü yok, dolayısıyla
tüm manipülasyonu tek bir sürekli ayakta durmada geçiyor:

| sürekli ayakta durma | tamamlanan bölüm |
|---|---|
| 11.5 s (Lab 8'in süreleri) | 0 / 4 |
| 6.9 s | 3 / 4 |
| **5.2 s** | **4 / 4** |

İmza tartışmasız: DCM hatası LIPM oranında üstel büyür, 4.5 mm'den başlayarak
~0.15 s'de ikiye katlanır; bu sırada el hâlâ 5 mm'ye izler ve tepe tork 21 N·m'de
durur. Bir doyma değil, bir kararsızlık. Lab 9'un faz süreleri 5.6 s'lik bir
bütçeye göre boyutlandırılmıştır ve bir test bunu doğrular.

---

## Modüller

| Dosya | Rolü |
|---|---|
| `lab9_common.py` | Yollar, görüntü/gözlem sabitleri, talimat sözlüğü, Lab 8 yeniden dışa aktarımları |
| `vla_scene.py` | İki nesneli rastgeleleştirilebilir sahne, kafa + bilek kameraları, dört weld, bırakma işareti |
| `observations.py` | Gözlem ve eylem sözleşmesi — düzenin tanımlandığı tek yer |
| `expert.py` | Lab 8'in capstone'u türetilmiş: tohumlanabilir, hedef seçilebilir, gözlem yakalayan |
| `collect_demos.py` | Çok süreçli koşumlar, faz bazlı dilimleme |
| `dataset.py` | Dolgu maskeli öbek pencereleri, tohum bazlı bölme, normalleştirme istatistikleri |
| `text_encoder.py` | Dondurulmuş CLIP metin kulesi ve talimat bankası |
| `act_policy.py` | ACT modeli: iki kamera, türetilmiş jeton sayısı, iki eylem kafası |
| `train.py` | Maskeli L1 eğitimi, taban-çizgisine göre doğrulama |
| `policy_runner.py` | Kapalı çevrim yürütme: 10 Hz politika, Lab 8'in QP'si üzerinde |
| `evaluate.py` | Görev başına başarı, geniş aralık, talimat takası, eklem ablasyonu |
| `capstone_demo.py` | Serbest metin girer, kayıtlı bölüm çıkar, çıkarım profillemesi |
| `mN_*.py` | Kilometre taşı başına bir kapı betiği, kanıtını `media/` altına yazar |

Sahneye dair hiçbir şey depoya işlenmez: Menagerie artı Lab 8'in spec kurucusu
üzerinden çalışma zamanında inşa edilir — tam olarak Lab 8'in yaptığı gibi.

---

## Veri akışı

### Bir gösterim toplamak

```
tohum ──► Randomisation ──► sahne (iki nesne, kameralar, weld'ler)
                             │
                    Lab 8 kontrolcüsü bölümü 1 kHz'de koşturur
                             │  her 100 tikte:
                             ▼
       iki 128px render + 62-d durum + uzmanın kendi komutu
                             │  faza göre dilimlenir
                             ▼
              walk segmenti          pick segmenti
        "kırmızı bardağa yürü"   "kırmızı bardağı al"
```

Saklanan eylem, ulaşılan durum değil **uzmanın komutudur**. Davranış klonlama
uzmanın *yaptığını* taklit eder; uyumlu ve bozucuya maruz bir sistemde ikisi
farklıdır ve sonucu eğitmek, politikaya kendi geçmişini kovalamayı öğretir.

Yalnızca başarılı bölümler yazılır. Başarısız olan, devrilen bir robotun kaydıdır
ve kareleri, devrilme anına kadar iyi olanlardan ayırt edilemez.

### Bir kapalı çevrim tiki

```
MuJoCo durumu ─► iki render + durum ─► ACT ─► öbek (20, 9)
                    │ ~194 ms                    │ ilk eylem
                    │ (yazılım render,           ▼
                    │  asıl hız sınırlayıcı)  dünya çerçevesine çöz
                                                 │
                                      yürüyüş? ─┴─ dur
                                            │         │
                                    Lab 8 yürüyüş     el görevleri +
                                    birimi            dondurulmuş DCM
                                            │         │
                                            └────┬────┘
                                       Lab 8 QP 1 kHz → τ
```

İki ayaklı bir robota adımın ortasında "dur" denemez, bu yüzden yürüyüş komutu
yalnızca birim sınırlarında işlenir ve bir birim, bir adım artı kapanış adımıdır
— Lab 8'in doğruladığı konfigürasyon (L-M5-e: adım ortasında biten bir yürüyüş,
bir sonrakine atlatamayacağı bir duruş devreder).

---

## Model

`ozkannceylan/humanoid_vla`'nın `ACTPolicy`'sinden uyarlanmıştır; o da Zhao ve
ark. (RSS 2023). Farklar ve sebepleri:

| yukarı akış | burada | neden |
|---|---|---|
| 49 uzamsal jeton, sabit kodlu | öznitelik haritasından türetilir (128 px'de 16) | 49, ResNet18'in 224 px girdi için 7×7 çıktısıdır ve başka her boyutta sessizce yanlıştır |
| tek kamera | kafa + bilek, her biri kendi kamera gömmesiyle | 128 px'de el nesneye yaklaştığında nesneler kafa görüntüsünde bir avuç pikseldir |
| eylem = eklem hedefleri | iki kafa, `task` birincil | Lab 7'nin bulgusu |
| isteğe bağlı zamansal topluluk | yok | adım başına fazladan bir ileri geçiş, ve bu kontrol döngüsü zaten render-bağımlı |
| ImageNet ResNet18, layer4 ince ayarlı | aynı | koru |
| norm istatistikleri modül tamponlarında | aynı | koru — kendi çıktısını denormalleştiremeyen bir checkpoint sorunsuz yüklenir ve bir ölçek çarpanı kadar yanlıştır |

~15.8 M parametre, ~13.0 M eğitilebilir.

### Talimat bankası

Metin kulesi *eğitim* zamanında sözlüğü gömmek için kullanılır ve gömmeler
checkpoint'in içine yazılır. Değerlendirme, kapalı çevrim koşucusu ve capstone
sonrasında ne `transformers` ne de ağ ister — talimatı bankadan arar. Gerçekten
yeni bir cümleyi çıkarım anında kodlamak hâlâ çalışır ve hâlâ kuleyi ister.

Bu sözlük üzerinde ölçüldü: aynı komutun başka sözcüklerle ifadeleri kosinüs
0.957'de, farklı anlamlı komutlar 0.846'da — 0.111'lik bir marj. Başka
sözcüklere dayanıklılık ile talimat ayırt edilebilirliği, koşullandırmanın
ihtiyaç duyduğu iki özelliktir ve zıt yönlere çekerler; bu yüzden ikisi de
eğitimden **önce** denetlenir, kötü bir başarı oranından sonra çıkarsanmaz.

---

## Eğitim

Öbek üzerinde maskeli L1. Yanlış yapılması kolay iki ayrıntı:

**Maske.** Bir segmentin sonuna yaklaşırken `chunk_size` kadar gerçek eylem
kalmaz. Maskesiz, dolgu kuyruğu politikaya görevin bitiminden iki saniye önce
durmayı öğretir.

**Bölme kare bazlı değil, sahne tohumu bazlıdır.** Aynı bölümde 100 ms arayla iki
kare neredeyse aynıdır; kare bazlı bir bölme, ezberi ölçen bir doğrulama kaybı
raporlar ve bu mükemmel görünür.

Doğrulama **ham birimlerde, ortalamayı-tahmin-et taban çizgisinin yanında**
raporlanır. Normalleştirilmiş 0.31'lik bir L1 hiçbir şey söylemez. Aynı sayının
el hedefi milimetresi cinsinden hâli, eğitim ortalamasını tahmin etmenin alacağı
puanın yanında, modelin hiçbir şey öğrenip öğrenmediğini söyler.

`layer4`, kafanın öğrenme oranının onda birini alır: birkaç bin örneğin
iyileştirebileceğinden daha hızlı bozabileceği ImageNet öznitelikleri taşır.

---

## Çerçeve ve birim kuralları

- El hedefleri **pelvise göre, yalnız-sapma**, metre cinsinden.
- Görüntüler 128×128 uint8 RGB; model içinde ImageNet istatistikleriyle
  normalleştirilir.
- Lab 8'in kuralları değişmeden geçer: `pin.LOCAL_WORLD_ALIGNED` Jacobian'lar,
  konfigürasyon güncellemeleri için `pin.integrate` (yüzen tabanda `nq ≠ nv`),
  Pinocchio'nun dünyasının MuJoCo'nunkinin `PELVIS_MJCF_Z` altında oturması.
- Yabancı laboratuvarlar `sys.path`'e **`append` ile eklenir, asla `insert(0)`
  ile değil** — laboratuvarlar modül adlarını paylaşır ve yabancı bir `src/`
  dizinini öne koymak yerel modülleri sessizce gölgeler.

---

## Hesabın nereye gittiği

| büyüklük | ölçülen |
|---|---|
| CPU / RAM | 4 çekirdek, 15 GB, **CUDA aygıtı yok** |
| MuJoCo ekran dışı render | **97 ms/kare**, çözünürlükten bağımsız (gölge, yansıma ve gökkutusu açıkken 380 ms) |
| ResNet18 ileri+geri, yığın 16 | 224 px'de 32 örnek/s · 128 px'de 117 · 96 px'de 193 |
| Bir uzman bölümü | ~11 s simülasyon için ~35 s duvar saati |

Bunların her biri plan yazılmadan **önce** ölçüldü ve görüntü boyutunu, gölge
ayarlarını, veri kümesi büyüklüğünü ve devir bütçesini bunlar belirledi.
Varsayılan donanıma göre yazılmış bir plan, sahip olmadığınız bir makine için
yazılmış bir plandır.
