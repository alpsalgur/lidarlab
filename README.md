# LidarLab

LiDAR kayıtları (frame başına bir `.npz`) için görüntüleyici ve entegre Python ortamı.

## Kurulum

```powershell
python -m venv venv
venv\Scripts\python -m pip install -e ".[dev]"
venv\Scripts\python tools\smoke_test.py      # Qt + VTK + konsol birlikte çalışıyor mu
```

## Test verisi

```powershell
# sentetik kayıt: 200 frame, 3 eko, kaydın ortasından itibaren yağmur ve sprey
venv\Scripts\python -m lidarlab.datasets.synthetic data\synthetic\city_rain

# KITTI raw sürüşünü indir (~0,9 GB) ve npz'ye çevir
venv\Scripts\python -m lidarlab.datasets.kitti 2011_09_26_drive_0011 data\kitti
```

## Geliştirme

```powershell
venv\Scripts\python -m pytest
venv\Scripts\python -m ruff check .
```
