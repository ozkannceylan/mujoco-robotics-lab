# B3: Full Pipeline — Öğrenme Notu

## Zincir

Bu adımda parçalar birleştirildi:

```text
target -> IK -> trajectory -> PD -> plant -> metrics/log
```

## Demo'lar

`src/b3_full_pipeline.py` üç demo içerir:

1. Pick and place
   - `(0.20, 0.30) -> (0.40, 0.10) -> (0.20, 0.30)`
   - waypoint'ler analitik IK ile joint-space'e çevrilir
   - segmentler quintic trajectory ile bağlanır
2. Circle tracking
   - cartesian daire hedefi üretilir
   - her örnekte IK çözülür
   - PD controller ile takip edilir
3. Singularity edge
   - dış workspace sınırına yaklaşan hedeflerde `pinv` ve `dls` karşılaştırılır

## Çıktılar

- `docs/b3_pick_place_log.csv`
- `docs/b3_circle_log.csv`
- `docs/b3_singularity_log.csv`
- `docs/b3_metrics.csv`

`matplotlib` ve uygun writer varsa ayrıca:

- `docs/b3_pipeline_paths.png`
- `docs/b3_pick_place.gif`

## Beklenen Yorum

- pick-and-place kapalı döngüde hatasız tekrar edebilmelidir
- circle tracking için gerçek path istenen daireye yakın kalmalıdır
- singularity kenarında `dls`, `pinv`'e göre daha dayanıklı olmalıdır
