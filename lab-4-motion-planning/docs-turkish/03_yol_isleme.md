# Yol Isleme

## Genel Bakis

Lab 4 yol isleme iki asamayi korur:

1. kisayol alma (`shortcut_path`)
2. zaman parametrizasyonu (`parameterize_topp_ra`)

## Kisayol Alma

Ham planlayici yolundan gereksiz ara noktalar cikarilir. Iki uzak waypoint
arasindaki dogrusal eklem-uzayi baglanti carpisma-ozgurse ara noktalar silinir.

Bu asamadaki carpisma sorgulari artik kanonik MuJoCo geometrisi uzerinden
yapilir.

## Zaman Parametrizasyonu

`parameterize_topp_ra(...)` fonksiyon adi korunmustur.

- TOPP-RA mevcutsa onu kullanir
- mevcut degilse hiz ve ivme sinirlarina uyan muhafazakar quintic bir
  zamanlandirma yedegi kullanir

Bu sayede Lab 4 boru hatti ayni kalir ve cevreye bagli kurulum sorunlari
tum laboratuvari durdurmaz.
