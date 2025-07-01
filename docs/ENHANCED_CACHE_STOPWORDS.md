# Enhanced Cache dengan Sastrawi Library untuk Multi-Layer Caching

## Tujuan Enhancement

Meningkatkan cache hit rate dengan menggunakan **Sastrawi library** untuk stemming dan stopword removal yang lebih robust dalam fungsi canonicalize_query(), menggantikan manual stopwords list.

## Perubahan yang Dilakukan

### Sebelum Enhancement:

-   Manual stopwords list (~100+ kata yang dikategorikan)
-   Tidak ada stemming
-   Query "Apa mekanisme jaminan kesehatan pemerintah bagi orang miskin?" menghasilkan canonical berbeda

### Setelah Enhancement (Sastrawi):

-   **Sastrawi StopWordRemover**: Automatic stopword removal dengan kamus lengkap
-   **Sastrawi Stemmer**: Indonesian stemming untuk mengurangi infleksi kata
-   **Fallback mechanism**: Manual stopwords jika Sastrawi tidak tersedia

## Workflow Canonicalization dengan Sastrawi

```python
def canonicalize_query(query: str) -> str:
    # Step 1: Clean dan lowercase
    clean_query = re.sub(r'[^\w\s]', ' ', query.lower())

    # Step 2: Remove stopwords dengan Sastrawi StopWordRemover
    no_stopwords = stopword_remover.remove(clean_query)

    # Step 3: Stemming dengan Sastrawi Stemmer
    stemmed = stemmer.stem(no_stopwords)

    # Step 4: Tokenize, deduplicate, dan sort
    tokens = [token for token in stemmed.split() if len(token) > 1]
    return " ".join(sorted(set(tokens)))
```

## Hasil Testing dengan Sastrawi

### Demo Sastrawi Processing:

```
"Bagaimana mekanisme jaminan kesehatan bagi orang miskin?"
Canonical: 'bagaimana jamin mekanisme miskin orang sehat'

"Apa mekanisme jaminan kesehatan pemerintah bagi orang miskin?"
Canonical: 'apa jamin mekanisme miskin orang perintah sehat'

"Bagaimanakah mekanisme-mekanisme jaminan kesehatan pemerintahan untuk orang-orang miskin?"
Canonical: 'bagaimana jamin mekanisme miskin orang perintah sehat'
```

### Keunggulan Sastrawi Stemming:

-   **"jaminan" → "jamin"** (mengurangi infleksi)
-   **"kesehatan" → "sehat"** (ke root word)
-   **"pemerintahan" → "perintah"** (stemming kompleks)
-   **"mengetahui" → "tahu"** (handle prefiks me-)
-   **"dijelaskan" → "jelas"** (handle prefiks di- dan sufiks -kan)

### Performance Enhancement:

-   **3/5 queries** → LEVEL 2 SEMANTIC HIT (similarity > 0.80)
-   **Stemming** membuat variasi infleksi jadi lebih mirip
-   **Automatic stopword removal** lebih comprehensive dari manual list
-   **Better semantic clustering** karena words di-reduce ke root form

## Contoh Query yang Kena Cache:

```python
# LEVEL 2 SEMANTIC HIT (dengan Sastrawi):
"Apa mekanisme jaminan kesehatan pemerintah bagi orang miskin?" (0.8725) ✅
"Bagaimanakah mekanisme-mekanisme jaminan kesehatan pemerintahan untuk orang-orang miskin?" (0.9101) ✅
"Jelaskan mekanisme yang digunakan untuk menjamin kesehatan orang miskin" (0.9039) ✅
```

## Impact pada Cache Performance

1. **Better Stemming**: Handle infleksi Indonesia yang kompleks (ber-, me-, -kan, -an, dll.)
2. **Comprehensive Stopwords**: Sastrawi punya stopwords list yang lebih lengkap
3. **Robust Processing**: Library teruji untuk Indonesian NLP tasks
4. **Maintainable**: Tidak perlu maintain manual stopwords list
5. **Future-proof**: Bisa di-update via library version

## Dependencies Baru

-   `Sastrawi==1.0.1` ditambahkan ke requirements.txt
-   Automatic fallback ke manual approach jika Sastrawi tidak tersedia
-   Backward compatibility terjaga

## File yang Dimodifikasi

-   `src/cache/smart_cache.py` - Replace manual stopwords dengan Sastrawi
-   `requirements.txt` - Tambah Sastrawi dependency
-   Testing menunjukkan improvement dalam semantic clustering dari stemming

## Kesimpulan

Sastrawi memberikan improvement signifikan dalam:

-   **Stemming accuracy** untuk Bahasa Indonesia
-   **Automatic stopword handling**
-   **Better semantic similarity** karena words di-reduce ke root form
-   **Maintenance reduction** (tidak perlu manual update stopwords)

Query paraphrase sekarang lebih likely untuk kena cache karena stemming yang lebih sophisticated!
