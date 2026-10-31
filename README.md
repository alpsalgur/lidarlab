# LidarLab

LiDAR test kayıtları (frame başına bir `.npz`) için görüntüleyici ve entegre Python ortamı.

## Kurulum

```powershell
python -m venv venv
venv\Scripts\python -m pip install -e ".[dev]"
venv\Scripts\python tools\smoke_test.py      # Qt + VTK + konsol birlikte çalışıyor mu
```

## Çalıştırma

```powershell
.\LidarLab.bat                                    # ya da Explorer'da çift tık
venv\Scripts\lidarlab data\synthetic\city_rain   # bir test run ile başlat
venv\Scripts\python -m lidarlab                  # konsol penceresiyle (hata ayıklarken)
```

Uygulama son açılan test run'ı ve pencere düzenini hatırlar; `--reset-layout` varsayılan düzene döner.

| Bölge | İçerik |
|---|---|
| Sol | **Test Runs**: `data/` altındaki kayıtlar (çift tık açar, `+` başka klasör ekler) · **Pipeline Browser**: kayıt, işleme adımları ve katmanlar; kutucuklar aç/kapa, sütunlar adım sonrası nokta sayısı ve süre |
| Orta | 3D görünüm ve oynatıcı çubuğu. Seek bar'daki açık gri bölüm bellekteki frame'leri gösterir |
| Sağ | **Properties**: renklendirme, nokta boyutu, eko görünürlüğü, sahne, kamera · **Information**: dosya, zaman, poz, eko başına nokta, sınırlar, diziler, süreler |
| Alt | **Python Shell** (IPython, görüntüleyiciyle aynı süreç) · **Script Editor** |

Klavye: Space oynat/durdur · ←/→ frame (Shift ile 10) · Home/End · 1–4 kamera (3D, BEV, sürücü, yan) · R sığdır · C sonraki renk alanı · +/− nokta boyutu · F1 betik referansı. Yazı yazılan alanlarda (konsol, editör) bu tuşlar normal çalışır.

## Python ile veri işleme

Konsol ve editör görüntüleyiciyle aynı isimleri paylaşır: `rec` (test run), `frame` (o anki frame, ham), `view` (görüntüleyici), `pipe` (işleme adımları), `lab`, `np`.

```python
frame.fields                       # intensity, echo, ring
frame[frame.echo != 0]             # alt küme: tüm alanlar ve kaynak indeksleri korunur
frame.range, frame.pulse           # türetilmiş: mesafe, aynı lazer atışının ekoları

@lab.step                          # oynatırken her frame'e uygulanır
def remove_cover_echo(f):
    return f.echo != 0             # True = noktayı tut

@view.highlight(color="red")       # silmeden boyar, her frame'de yeniden hesaplanır
def weak(f):
    return f.intensity < 10

view.output.save("cikti.npz")      # adımlardan sonraki frame, kaynak düzeninde
```

Hazır örnekler: Script Editor → Examples (eko keşfi, kapak ekosu, yağmur ekoları, parlak noktalar, ROR, dışa aktarma). Editörde F5 betiği, Ctrl+Enter seçimi ya da satırı çalıştırır. "Run on external save" açıkken editörde açık bir betiği VS Code'da kaydetmek onu yeniden yükleyip çalıştırır; adımlar aynı isimle yerini alır.

GUI'siz kullanım:

```python
import lidarlab as lab
rec = lab.open("data/synthetic/city_rain")
f = rec[150]
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
# arayüzü kullanıcısız sürer (yaklaşık 1 dakika pencere açılır), data\previews'a ekran görüntüsü yazar
venv\Scripts\python tools\ui_selftest.py data\synthetic\city_rain data\kitti\2011_09_26_drive_0011
```
