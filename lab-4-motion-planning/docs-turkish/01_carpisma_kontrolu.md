# Carpisma Kontrolu

## Son Mimari

Lab 4 carpisma dogrusu artik dogrudan yurutulen MuJoCo geometrisinden gelir:

- Menagerie `universal_robots_ur5e`
- monte `robotiq_2f85`
- ayni sahnedeki masa ve engel geometrileri

Yani planlayici ile simulasyon ayni robotu gorur. Ayrik bir ozel carpisma
modeli artik birincil dogruluk kaynagi degildir.

## CollisionChecker Arayuzu

Sinif su fonksiyonlari korur:

- `is_collision_free(q)`
- `is_path_free(q1, q2, resolution=0.05)`
- `compute_min_distance(q)`

Ic tarafta:

1. kanonik MuJoCo sahnesi yuklenir
2. robot-engel ve anlamli kendi-carpisma ciftleri kaydedilir
3. her `q` icin MuJoCo sahnesi ileri sarilip carpisma sonucu okunur

## Kendi-Carpisma Politikasi

- kol-kol ve kol-tutucu ciftleri kontrol edilir
- robot-cevre ciftleri kontrol edilir
- ayni Robotiq mekanizmasi icindeki anlamsiz ic yakinliklar planlama
  carpisma sinyali olarak sayilmaz

## Son Durum

- Lab 4 carpisma testleri kanonik yigin uzerinde geciyor
- FK uyumu korunuyor
- planlama artik yurutulen gercek geometriye bakiyor
