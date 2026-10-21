# leadopt_bench — Faz 1: Görev 1 (veri toplama altyapısı)

LLM'lerin lead optimization muhakemesini **yayınlanmış deneysel SAR verisine** karşı
retrospektif olarak ölçmek için veri katmanı. Bu fazda tahmin modeli oracle'ı,
docking ve afinite tahmincisi **yok** — doğru cevap zaten ChEMBL'de.

## Durum

| Görev | Durum |
|---|---|
| 1 — Veri toplama altyapısı | Bitti; 14 hedef tarandı, 716 seri |
| 2 — Seri bölme (erken/geç) | Bitti; 32 birincil + 33 duyarlılık bölmesi |
| 3 — LLM döngüsü | Altyapı hazır + mock ile doğrulandı; **API anahtarı bekliyor** |
| 4 — Metrikler | Kuruldu + mock tabanına karşı kalibre edildi; 3 tanım kusuru düzeltildi |
| 5 — Başarısızlık analizi | Kategoriler + kontrol merdiveni hazır; gerçek LLM verisi bekliyor |

**LLM kolu neden boş:** makinede Anthropic kimlik bilgisi yok. Kod hazır;
anahtar gelince tek komut:
`python3 -c "from leadopt_bench import llm; llm.run_batch(llm.get_provider('anthropic'), model='claude-opus-5-5')"`

## Panel (Görev 1-2 sonucu)

4 hedef, iki yıl katmanı, `assay` gruplaması:

| Hedef | Rol | eski (≤2015) | yeni (2024+) |
|---|---|---|---|
| BRD4 | maruziyet ekseni | 5 | 5 |
| HDAC1 | maruziyet ekseni | 5 | 4 |
| Trombin | maruziyet-pozitif kontrol | 5 | (2, iddiadan dışlandı) |
| EGFR | maruziyet-pozitif kontrol | 5 | (1, iddiadan dışlandı) |

**Birincil strateji `window`**: lead sabit pChEMBL [5.0, 7.0] penceresinden, kazanç ≥1.0 log.
`activity` (alt çeyrek) duyarlılık kontrolü olarak ayrıca diske yazılıyor.

### Maruziyet ekseni ne iddia ediyor, ne iddia etmiyor

2024+ serilerin eğitim verisinde **olmadığını** iddia etmiyoruz — bu modellerin bilgi
kesme tarihleri o yılları kapsıyor. Eksenin ölçtüğü şey **maruziyet tekrarı**: 2013
tarihli bir EGFR makalesi binlerce atıf, derleme ve ders kitabı üzerinden tekrar
tekrar görülmüş; 2024 tarihli bir BRD4 makalesi büyük olasılıkla bir kez.

Eksen ilk kurulduğunda zorlukla karışıktı: eski kampanyalar mikromolardan başlayıp
~3.3 log kazanıyor, modern olanlar zaten potent bir yerden ~2.4 kazanıyordu
(lead potensi p=0.047). Sabit potens penceresi bunu çözdü — maruziyet ekseni
hedeflerinde lead potensi p=0.51, kazanç p=0.18, hedef sayısı p=0.87.
Confound'un neredeyse tamamı Trombin ve EGFR'deydi; o yüzden onlar yalnız eski
serileriyle kontrol görevi görüyor.

### Lead seçimi

Pencere içinden **temsili** bileşik: seri bütününe Tanimoto benzerliği en yüksek olan.
Bir kimyacı serinin en tuhaf üyesinden optimizasyona başlamaz, ve bir serinin en zayıf
bileşiği genelde tam olarak odur — SAR'ı bozmuş bir deneme. Seçim ayrıca **kazanç payı
olan** üyelerle sınırlanıyor: merkezi bir lead serinin tepesine o kadar yakın olabilir ki
hiçbir bileşik min_gain'i geçmez ve seri boşa gider.

## Modüller

- `config.py` — yollar ve seri kabul ölçütleri (tek doğruluk kaynağı)
- `chembl.py` — ChEMBL REST istemcisi; her sayfa diske `.json.gz` olarak yazılır,
  tekrar çalıştırma ağa çıkmaz
- `series.py` — temizleme, standardizasyon, kongenerik seri kurma
- `survey.py` — hedef başına **kaç kullanılabilir seri** çıktığını ölçer
- `panel.py` — 4 hedefi iki yıl katmanına ayırıp paneli seçer
- `split.py` — erken/geç bölme: `window` (birincil), `activity`, `temporal`
- `llm.py` — sağlayıcı-agnostik değerlendirme döngüsü; şema-zorunlu JSON çıktı,
  koşul başına disk önbelleği, `MockProvider` ile ağsız kuru çalıştırma
- `metrics.py` — beş eksende puanlama + kategorilendirme
- `baselines.py` — null ve oracle kontrol kolları (merdiven aşağıda)

## Metrikler (Görev 4)

Her öneri tek bir kategoriye düşüyor: `invalid`, `unchanged`, `duplicate`,
`scaffold_violation`, `recovered_target`, `recovered_nonimprover`,
`novel_plausible`. Yanında ortogonal bayraklar: `generic_flag`,
`direction_precision`, `approach_gain`, `mw_delta`.

**Hiçbir potens/ADMET/sentezlenebilirlik tahmincisi yok.** Seride olmayan bir
öneri `novel_plausible` etiketlenir, iyi ya da kötü **puanlanmaz** — onun için
gerçek referansımız yok ve bir modelle referans uydurmak tam olarak bu projenin
tepki verdiği ödül sömürüsü olurdu.

İki eşik seriden kalibre ediliyor, elle seçilmiyor:
- **benzerlik tabanı** = gerçek kimyacıların bu lead'den yaptığı *en uzak* bileşik
- **"potens artıran değişiklik"** = aynı seride lead→hedef fonksiyonel grup deltaları

## İlk gerçek LLM kolu: Gemma 3 12B (2026-10-17)

Google DeepMind Gemma 3 12B (Q4_K_M), Ollama ile yerel, sıcaklık 0, şema-zorunlu
JSON. 32 bölme × 5 öneri = **160 öneri**, 61.7 dakika, sıfır maliyet.

| Kol | geçerli | iskelet | geri kaz. | **yön** | temelsiz | lead'in aynısı | geçersiz |
|---|---|---|---|---|---|---|---|
| **Gemma 3 12B** | 0.681 | 0.825 | **0.000** | **0.491** | **0.334** | **0.094** | **0.319** |
| mock | 0.594 | 0.594 | 0.006 | 0.526 | 0.006 | 0.000 | 0.406 |
| generic | 1.000 | 1.000 | 0.019 | 0.607 | 0.038 | 0.000 | 0.000 |
| random | 1.000 | 0.938 | 0.013 | 0.533 | 0.013 | 0.000 | 0.000 |
| neighbour | 1.000 | 0.019 | 0.000 | 0.507 | 0.000 | 0.000 | 0.000 |
| oracle | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |

### Neyin anlamlı olduğu, neyin olmadığı

Bu ayrım bulgunun kendisi kadar önemli:

**Anlamlı DEĞİL — null kollardan ayırt edilemiyor:**

| Karşılaştırma | Gemma | Null | p |
|---|---|---|---|
| yön isabeti vs `mock` | 0.491 | 0.526 | 0.693 |
| yön isabeti vs `generic` | 0.491 | 0.607 | 0.166 |
| yön isabeti vs `random` | 0.491 | 0.533 | 0.495 |
| geri kazanım vs `generic` | 0/160 | 3/160 | 0.248 |
| geri kazanım vs `random` | 0/160 | 2/160 | 0.498 |

**Anlamlı — ve kötü yönde:** geçerlilik 109/160 vs 160/160, **p=7.6×10⁻¹⁸**.

Yani doğru ifade "rastgeleden kötü" değil, **"rastgeleden ayırt edilemez"**:
160 önerinin hiçbiri gerçek bir iyileştirilmiş bileşiği tutmadı, ve seçtiği
değişiklik tipleri gerçek SAR'daki potens artıran tiplerle rastgele sübstitüent
seçmekten daha fazla örtüşmedi. İstatistiksel olarak ayrıldığı tek eksen
geçerlilik, orada da belirgin şekilde geride.

### Başarısızlık dağılımı (Görev 5)

| Kategori | n | % |
|---|---|---|
| geçersiz molekül | 51 | 31.9% |
| seride yok ama makul (`novel_plausible`) | 89 | 55.6% |
| lead'i olduğu gibi geri verdi | 15 | 9.4% |
| iskelet ihlali | 3 | 1.9% |
| gerçek seride var, iyileştirmeyen | 1 | 0.6% |
| tekrar | 1 | 0.6% |
| **gerçek iyileştirmeyi tuttu** | **0** | **0.0%** |

Önerilerin **%41.3'ü cevap bile değil** (%31.9 geçersiz + %9.4 lead'in aynısı).

Geçerli önerilerin **%33.4'ünde gerekçe, ne lead'de ne öneride bulunan bir yapıya
atıf yapıyor** — null kollarda bu oran %0–3.8. En sık uydurulan: nitro (24),
sülfonamid (6), piperazin (5), piperidin (4), hidroksamat (4).

**Test ettiğim ve çürüttüğüm hipotez:** nitro halüsinasyonunun SMILES'teki `no1`
(izoksazol halka kapanışı) parçasının nitro olarak okunmasından kaynaklandığını
sandım. Değil: 17 bölmede görülüyor, hiçbirinin lead'inde nitro yok ve yalnız
1'inde N-O halkası var. Yani temsil karışıklığı değil — model moleküle
bakmaksızın jenerik medisinal kimya sözcüğü üretiyor.

**Kapsam uyarısı:** bu 12B açık ağırlıklı bir model. Bulgu 12B ölçeği hakkında;
"LLM'ler lead optimization yapamıyor" demek için sınır modeli gerekiyor.

### Kontrol merdiveni (`baselines.py`)

Tek bir skor kendi başına yorumlanamaz. "Model gerçek iyileştirmelerin %12'sini
geri kazandı" ancak rastgele kimyanın, ezbere tavsiyenin ve kusursuz bilginin
**aynı 32 bölmede** ne aldığını bilince bulgu olur. Merdiven, zayıftan güçlüye:

| Kol | Ne yapıyor | geçerli | iskelet | tam geri kaz. | yakın g.k. | jenerik | **yön** |
|---|---|---|---|---|---|---|---|
| `mock` | SMILES metnine düz atom ekler | 0.594 | 0.594 | 0.000 | 0.006 | 0.444 | 0.526 |
| `generic` | her lead'e aynı 5 sübstitüent | 1.000 | 1.000 | 0.000 | 0.019 | 0.350 | 0.607 |
| `random` | rastgele geçerli analog | 1.000 | 0.938 | 0.000 | 0.013 | 0.025 | 0.533 |
| `neighbour` | aynı proteinin *başka* serilerinden potent bileşikler | 1.000 | **0.019** | 0.000 | **0.000** | 0.000 | 0.507 |
| `oracle` | gerçek iyileştirilmiş bileşikler (kasıtlı hile) | 1.000 | 1.000 | **1.000** | **1.000** | 0.079 | **1.000** |

Dördü de 5 öneri × 32 bölme = 160 öneri. `oracle` ve `neighbour`, hiçbir gerçek
modelin görmediği veriyi `bind_split` ile alıyor — kasıtlı hileler, öyle
etiketli, ve asla model olarak raporlanmayacak.

Bu merdiven dört şeyi kanıtlıyor:

1. **Metrikler tavana ulaşıyor.** Oracle her eksende tam 1.000. Yani ölçümü
   kıstıran bir puanlama hatası yok.
2. **Yön isabetinin tabanı 0.53–0.61.** En önemli kalibrasyon sonucu. 0.6
   alan bir model hiçbir şey göstermemiştir; yalnız 1.0'a yaklaşan değerler
   anlamlı. Bu yüzden `generic` kolu kalıcı.
3. **Geri kazanımın özgüllüğü yüksek.** Bütün null kollar ≤%1.9. Rastgele
   geçerli kimya gerçek bir iyileştirilmiş bileşiği neredeyse hiç tutmuyor —
   yani geri kazanım zor ve anlamlı bir metrik. LLM'in geçmesi gereken çıta 0.019.
4. **Benzerlik kısıtı gerçek iş yapıyor.** `neighbour` kolu doğru proteine karşı
   *gerçekten potent* bileşikler öneriyor ve %98.1'i iskelet ihlali olarak
   işaretleniyor. Geri kazanım, bilinen aktifleri dökerek oyunlanamıyor.

### Mock tabanı (metriklerin kalibrasyonu)

`MockProvider` lead'e tek atom ekleyip dolgu gerekçe yazıyor. Metrikler bunu iyi
kimya sanmamalı — 32 bölme, 160 öneri üzerinde:

| Ölçüm | Mock tabanı |
|---|---|
| tam geri kazanım | **0.000** |
| yakın geri kazanım | **0.006** |
| jenerik oran | 0.444 |
| **yön isabeti** | **0.526** |

Son satır kritik: yalnızca F/Cl/metil/OH ekleyen bir model yön isabetinde
**0.53** alıyor, çünkü halojen taraması gerçekten de potens artıran bir
değişiklik. **Yön isabeti asla mutlak okunmamalı**; mock kontrol kolu bu yüzden
kalıcı bir koşul olarak duruyor.

(Mock'un %40.6 `invalid` oranı bir mock artefaktı — SMILES metnine düz atom
ekliyor. Bulgu değil, ama geçerlilik ekseninin çalıştığını kanıtlıyor.)

### Düzeltilen üç tanım kusuru

**1. Benzerlik tabanı tek aykırı hedefle çöküyordu.** Ham minimum 4 seride
0.076'ya kadar iniyordu (BRD4 bromodomain serileri birkaç kemotipe yayılıyor),
bu da o serilerde iskelet-ihlali eksenini anlamsız kılıyordu. 0.25'e
sabitlendi — seri kabul ölçütü olan 0.40'ın kasten altında. Ölçüldü: gerçek
hedeflerin %1.1'ini dışlıyor, alakasız sonda molekülünü 32/32 seride yakalıyor
(ham minimumda 4 seri dejenereydi).

**2. Geri kazanım "lead'in yanında durmayı" ödüllendiriyordu.** Lead'e bir atom
eklemek, kimyacıların yaptığı her yakın analoga ~0.9 benzerlik veriyor — öneri
cevaba hiç *yaklaşmadığı* halde kredi alıyordu. Artık geri kazanım pozitif
`approach_gain` istiyor: öneri o bileşiğe lead'den daha yakın olmalı. Sahte
`recovered_nonimprover` sayısı 6'dan 1'e düştü.

**3. Yön isabetinin yüksek bir tabanı var** (yukarıdaki 0.526) — mutlak değer
olarak raporlanamaz.

### LLM döngüsü hakkında üç tasarım kararı

**Gerekçe molekülden ayrı bir alan.** Şemada `rationale` zorunlu. Bir öneri
kimyacıların gerçekten yaptığı değişiklik olabilirken gerekçesi uydurma olabilir;
Faz 1 tam olarak bunu ayırt etmek için var, o yüzden ikisi tek metne katlanmıyor.

**Çıktı şemayla zorlanıyor, düz metinden ayıklanmıyor.** Prose üzerinde regex,
model kalitesiyle *ilişkili* şekilde başarısız olur — ayrıştırma hatası sessizce
sonucun parçası olurdu. Ayrıştırılamayan yanıt hata olarak kaydediliyor, kurtarılmıyor.

**`temperature` yok.** Güncel Claude modelleri bu parametreyi reddediyor (400);
derinlik `output_config.effort` ile ayarlanıyor ve effort deneysel koşulun parçası
olarak her yanıtla kaydediliyor.

Prompt'un iki çeşidi var: `implicit` (benzerlik kısıtı söylenmez — "yeni molekül
icat etti" gerçek bir başarısızlık modu olarak ölçülebilsin) ve `explicit`
(iskelet korunması açıkça istenir). Birincil ölçüm `implicit`.

## Veri yolu

```
ChEMBL activity (binding, IC50/Ki, pChEMBL var)
  → confidence_score 9 assay'lere kısıtla
  → satır temizliği   (sansürlü/şüpheli/birimsiz ölç. at)
  → standardizasyon   (en büyük parça, nötrleştir, 150<MW<900, canonical)
  → molekül başına tek satır (tekrarlarda medyan pChEMBL)
  → gruplama          (assay | document | cluster)
  → kabul ölçütü      (n≥20, ortalama Tanimoto>0.40, log aralık≥2.0)
```

Her adım diske yazar, her adım yeniden çalıştırılabilir.

### Neden üç gruplama var

"Seri" ChEMBL'de tek bir şey değil:

| Mod | Tanım | Karşılaştırılabilirlik | Zamansal bölme |
|---|---|---|---|
| `assay` | tek `assay_chembl_id` | en yüksek: tek protokol | **mümkün değil** (tek yıl) |
| `document` | tek makalenin SAR tablosu | yüksek | **mümkün değil** (tek yıl) |
| `cluster` | Murcko iskeleti + Butina, makaleler arası | daha düşük: protokol karışır | mümkün |

Bu, Görev 2'yi doğrudan etkiliyor: tek bir assay'in tek bir yayın yılı vardır,
yani **zamansal bölme ancak `cluster` modunda anlamlıdır** — protokol karışımı
bedeliyle. `survey` çıktısı her seri için `temporally_splittable` alanını taşır.

`cluster` modu havuzun tamamını Butina ile kümeler; mesafe matrisi önceden
ayrılmış `float32` numpy dizisinde tutulur (8000 bileşik ≈ 128 MB). İskelete göre
kovalama **yapılmaz**, çünkü gerçek seriler bundan zarar görüyor: CDK2'nin
`CHEMBL5735220` serisinde 134 bileşik var ama **32 ayrı Murcko iskeleti** —
çekirdek sabit, değişen şey çözünürlük amininin halkası (piperidin/morfolin/
azetidin...). İskelete göre kovalamak bu seriyi tekil parçalara bölüyordu.
İskelet kovalama yalnızca 8000'den büyük havuzlarda yedek yol olarak kalıyor.

## Temizlikte neyi atıyoruz

| Filtre | Gerekçe |
|---|---|
| `standard_relation != "="` | `>10 µM` bir potens değil, bir alt sınır |
| `data_validity_comment` dolu | ChEMBL'in kendisi o ölçüme güvenmiyor |
| `potential_duplicate` | aynı ölçümün ikinci kaydı seriyi şişirir |
| `standard_units != "nM"` | µg/mL, MW bağlamı olmadan çevrilemez |
| RDKit ayrıştıramıyor | kullanılamaz yapı |
| MW < 150 veya > 900 | fragment ya da lead optimization konusu olmayan büyüklük |

## Kullanım

```bash
# hedef taraması (sonuç: data/reports/survey_phase1.{json,csv})
python3 -c "from leadopt_bench.survey import survey; survey(['CHEMBL301'])"

# testler
python3 -m pytest leadopt_bench/tests/ -q
```

## Testler

155 test; ağ erişimi yok (sahte oturum/sağlayıcı). Kapsam: tuz/yük
standardizasyonu, sansürlü ve şüpheli ölçümlerin atılması, kongenerik-vs-çeşitli
ayrımı, tekrar ölçümlerin medyanla birleşmesi, çevresel halkası değişen serinin
kümelemede bölünmemesi (yukarıdaki kusurun regresyon testi), sayfalama,
önbellek isabeti, hata sonrası önbelleğin kirlenmemesi, bölme stratejilerinin
sırası/eşikleri, **prompt'a hedef bileşik sızmaması**, koşul başına önbellek
ayrımı, sağlayıcı hatasının yükseltilmeden kaydedilmesi, geçersiz yapıların
gerekçeli reddi, aykırı hedefin benzerlik tabanını çökertmemesi (regresyon),
geri kazanımın yaklaşma şartı (regresyon), **mock çıktısının iyi kimya
sanılmaması**, seride olmayan yakın analogun cezalandırılmaması.

Ayrıca her kuru çalıştırmada 32 **gerçek** prompt üzerinde sızıntı denetimi
yapılıyor (şu an: 0 sızıntı).
