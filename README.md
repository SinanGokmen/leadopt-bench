# leadopt-bench

LLM'lerin **lead optimization muhakemesini**, yayınlanmış deneysel SAR verisine karşı
retrospektif olarak ölçen bir boru hattı.

Temel fikir: bilinen bir optimizasyon serisinin **erken** bileşiğini modele ver,
iyileştirmesini iste, önerilerini kimyacıların gerçekte vardığı **geç** bileşiklerle
karşılaştır. Cevap anahtarı bir tahmin modeli değil — ChEMBL'de ölçülmüş aktivite.

**Bu fazda kasıtlı olarak yok:** ADMET tahmincisi, docking, afinite oracle'ı.
Seride bulunmayan bir öneri etiketlenir ama **puanlanmaz** — onun için referans yok,
ve bir modelle referans uydurmak alandaki ödül sömürüsü problemini tekrarlamak olur.

---

## Bir molekülü hemen deneyin

Kendi fikrinizi (ya da bir med chem'cinin fikrini) tarihsel kayda karşı test edin:

```python
from leadopt_bench import metrics as M

# hangi serilere karşı test edilebilir
for s in M.available_series()[:5]:
    print(s["series_id"], s["target"], "lead p=", s["lead_pchembl"])

# bir molekülü değerlendir
sc = M.evaluate_smiles(
    "CC(C)(C)c1cc(NC(=O)Nc2ccc(Oc3ccnc(-c4cnn(CCCCCCC(=O)NO)c4)c3)cc2)no1",
    "CHEMBL6106211_IC50",
    modification="asil zincirini kısalt",
    rationale="hidroksamik asit çinkoya bağlanıyor, zincir uzunluğu cebe oturmalı",
)
print(M.explain(sc))
```

Yukarıdaki çağrının gerçek çıktısı — verdiğiniz molekülü kimyacılar gerçekten
yapmış, o yüzden ölçülmüş değerini söylüyor:

```
geçerli molekül, lead'e Tanimoto 1.000 (seri tabanı 0.663)
aynı Murcko iskeleti
KİMYACILAR BUNU YAPTI ve potensi arttı
eşleşen gerçek bileşik: CHEMBL6149363, ölçülmüş p-değeri 7.51
boyut değişimi -4 ağır atom, -56 Da
```

(Tanimoto 1.000 bir kısıt gösteriyor: Morgan parmak izi dairesel olduğu için
yarıçapının ötesindeki zincir uzunluğu değişimini ayırt etmiyor. Araç boyut
değişimini ayrıca raporladığı için kör değil, ama iskelet kontrolü zincir
uzunluğuna duyarsızdır.)

Hiç denenmemiş bir analog verdiğinizde:

```
geçerli molekül, aynı Murcko iskeleti
seride yok -- kimse yapmamış, referansımız yok, karar uzmanın
değişiklik tipi isabeti 0.00 (seride görülmeyenler: hydroxamic_acid-, carboxylic_acid+)
```

Son satır işe yarar bir uyarı: HDAC'ta hidroksamik asit çinko bağlayan gruptur, ve
o seri onu atma yönünü hiç denemedi.

---

## Ne buldu

**Gemma 3 12B** (yerel, ücretsiz), 32 bölme × 5 öneri = 160 öneri:

| Ölçüm | Gemma 3 12B | jenerik tavsiye | rastgele analog | oracle |
|---|---|---|---|---|
geçerli molekül | 0.681 | 1.000 | 1.000 | 1.000 |
geri kazanım | **0.000** | 0.019 | 0.013 | 1.000 |
yön isabeti | **0.491** | 0.607 | 0.533 | 1.000 |
temelsiz gerekçe | **0.334** | 0.038 | 0.013 | 0.000 |

**Doğru ifade "rastgeleden kötü" değil, "rastgeleden ayırt edilemez":**
yön isabeti bütün null kollardan istatistiksel olarak ayrılmıyor (p=0.17–0.78),
geri kazanım da (0/160 vs 3/160, p=0.25). Anlamlı şekilde ayrıldığı tek eksen
geçerlilik — orada da geride (p=7.6×10⁻¹⁸).

Başarısızlığın dağılımı: 51 geçersiz molekül, 15 lead'in aynısı, 3 iskelet ihlali,
35 temelsiz gerekçeli. **160 önerinin 106'sı (%66) otomatik elenebiliyor**, 54'ü
uzman kararına kalıyor.

Gerekçelerin %33'ü molekülde **bulunmayan** yapılara atıf yapıyor — var olmayan
piperidin, var olmayan kinolin, 17 bölmede uydurma nitro grubu.

> Kapsam uyarısı: bu 12B açık ağırlıklı bir model. Bulgu 12B ölçeği hakkındadır.
> Sınır modeli kolu kodda hazır ama bu depoda koşulmadı.

---

## Kontrol merdiveni

Tek bir skor yorumlanamaz. Beş kol aynı 32 soruyu çözüyor:

| Kol | Ne yapıyor |
|---|---|
`generic` | her lead'e aynı 5 sübstitüent — her moleküle söylenebilecek tavsiye |
`random` | rastgele ama kimyasal olarak geçerli analog |
`neighbour` | aynı proteinin *başka* serilerinden potent bileşikler |
`oracle` | gerçek cevaplar (kasıtlı hile, tavan kontrolü) |
`mock` | SMILES metnine düz atom ekler (ayrıştırma tabanı) |

İkisi kritik:

- **`oracle` tam 1.000 alıyor** → metriklerin tavanı var, ölçümü kısıtlayan hata yok.
- **`neighbour` %98 iskelet ihlali alıyor** → benzerlik kısıtı gerçekten iş yapıyor;
  geri kazanım, bilinen aktifleri dökerek oyunlanamıyor.

Ve bir kalibrasyon uyarısı: yalnız flor/klor/metil ekleyen `generic` kolu yön
isabetinde **0.607** alıyor, çünkü halojen taraması gerçekten potens artıran bir
değişiklik. **Bu metrik mutlak okunamaz** — null kola göre okunur.

---

## Boru hattı

```
ChEMBL activity (binding, IC50/Ki, pChEMBL var)
  → confidence_score 9 assay'lere kısıtla
  → satır temizliği   (sansürlü/şüpheli/birimsiz ölçümleri at)
  → standardizasyon   (en büyük parça, nötrleştir, 150<MW<900, kanonik)
  → molekül başına tek satır (tekrarlarda medyan pChEMBL)
  → kongenerik seri   (n≥20, ortalama Tanimoto>0.40, log aralık≥2.0)
  → erken/geç bölme   (lead sabit potens penceresinden, kazanç≥1 log)
  → LLM'e sor         (yalnız lead + protein + deney tipi + potens)
  → üç kontrol + kategorilendirme
```

14 hedef tarandı → **716 seri**. Panel: BRD4, HDAC1, trombin, EGFR → **32 bölme**.

Her ChEMBL sayfası diske gzip'li yazılır; tekrar çalıştırma ağa çıkmaz.

### Üç kontrol — yargı değil, olgu

Hiçbirinde model görüşü yok, üçü de doğrulanabilir:

1. **Geçerli molekül mü** — RDKit ayrıştırma + sanitizasyon, tek parça, kütle aralığı
2. **İskelet korundu mu** — Morgan Tanimoto; eşik **seriden** kalibre: kimyacıların o
   lead'den yaptığı en uzak bileşik (0.25'e sabitlenmiş)
3. **Gerekçe tutarlı mı** — 40 halka/grubun SMARTS sözlüğü; metinde adı geçen her
   yapı lead'de **veya** öneride aranır. İkisinde de yoksa, model kendisine
   verilmeyen bir bileşiği anlatıyor.

---

## Kurulum ve kullanım

```bash
pip install -r requirements.txt

# testler (ağ erişimi gerektirmez)
python3 -m pytest leadopt_bench/tests/ -q

# yerel bir modelle tam koşu (ollama)
python3 -c "from leadopt_bench import llm; \
  llm.run_batch(llm.OllamaProvider(), model='gemma3:12b')"

# puanlama
python3 -c "from leadopt_bench import metrics as M; \
  M.write_scores(M.score_all('gemma3:12b'), 'rapor')"
```

Sağlayıcılar aynı arayüzden çağrılır: `ollama` (yerel), `openrouter`,
`anthropic`, `mock`. Model değiştirmek tek parametre.

## Modüller

| Dosya | Ne yapıyor |
|---|---|
`chembl.py` | önbellekli ChEMBL REST istemcisi |
`series.py` | temizleme, standardizasyon, kongenerik seri kurma |
`survey.py` | hedef başına kaç kullanılabilir seri çıktığını ölçer |
`panel.py` | hedefleri yıl katmanlarına ayırıp paneli seçer |
`split.py` | erken/geç bölme (`window`, `activity`, `temporal`) |
`llm.py` | sağlayıcı arayüzü, prompt, şema-zorunlu çıktı, önbellek |
`metrics.py` | beş eksende puanlama, üç kontrol, `evaluate_smiles` |
`baselines.py` | null ve oracle kontrol kolları |

Ayrıntılı tasarım notları ve metodolojik kararlar: [leadopt_bench/README.md](leadopt_bench/README.md)

## Testler

180 test, ağ erişimi yok (sahte oturum/sağlayıcı). En kritik olanlar:

- **prompt'a hedef bileşik sızmaması** — her koşuda 32 gerçek prompt denetlenir
- **oracle'ın tam geri kazanım alması** — alamazsa puanlama bozuktur
- kongenerik serinin kümelemede bölünmemesi
- aykırı bir hedefin benzerlik tabanını çökertmemesi
- geri kazanımın "hedefe yaklaşma" şartı
- mock çıktısının iyi kimya sanılmaması

## Lisans

MIT
