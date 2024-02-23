# CockroachDB: Store/Retrieve Based on Text Embeddings

**DISCLAIMER**


## Set up

```
$ export DB_URL="postgres://dbuser:passwd@127.0.0.1:26257/defaultdb?sslmode=require&sslrootcert=/crdb-certs/ca.crt"
$ psql $DB_URL
defaultdb=> CREATE TABLE text_embed
defaultdb-> (
defaultdb(>   uri STRING NOT NULL
defaultdb(>   , chunk_num INT NOT NULL
defaultdb(>   , token STRING NOT NULL
defaultdb(>   , chunk STRING NOT NULL
defaultdb(>   , PRIMARY KEY (uri, chunk_num)
defaultdb(> );
CREATE TABLE
Time: 28.192 ms
defaultdb=> CREATE INDEX ON text_embed USING GIN (token gin_trgm_ops);
CREATE INDEX
Time: 873.797 ms
```

## Index some documents

```
$ time ./array_embeddings.py ./data/*.txt
Log level: WARN (export LOG_LEVEL=[DEBUG|INFO|WARN|ERROR] to change this)
Indexing file ./data/aus_nz.txt now ...
Indexing file ./data/bafta_awards.txt now ...
Indexing file ./data/cockroach_labs.txt now ...
Indexing file ./data/decware_amp.txt now ...
Indexing file ./data/f64_group.txt now ...
Indexing file ./data/monaco.txt now ...
Indexing file ./data/new_orleans.txt now ...
Indexing file ./data/nyt_sesame_chicken.txt now ...
Indexing file ./data/nyt_three_ingredients.txt now ...
Indexing file ./data/palm_springs.txt now ...
Indexing file ./data/porsche_cabriolet.txt now ...
Indexing file ./data/robert_altman.txt now ...
Indexing file ./data/spotted_dove.txt now ...
Indexing file ./data/terry_blacks.txt now ...
Indexing file ./data/top_jazz_albums.txt now ...

real	0m8.457s
user	0m23.194s
sys	0m21.955s
```

## Query the index

```
$ time ./ngram_embeddings.py -q triode vacuum tube amp
Log level: ERROR (export LOG_LEVEL=[DEBUG|INFO|WARN|ERROR] to change this)
Query string: 'triode vacuum tube amp'
Token: '8b fl ii 9m a1 hn 69 f4 1o dq ae bi 95 a5 ge 34 j6 2q 7b 2i kj ay 50 8j da 5h it 61 gr 99 81 dm cl k9 ig fr 7r kd in 8u b1 62 c4 84 06 cc aa 29 dp 42 4n h9 cz cj ac i3 as ic bt gv dk 1e 68 39 '

terms_regex: (triode|amp|vacuum|tube)

URI: data/decware_amp.txt
SCORE: 0.281
TOKEN: 69 1d in fs 3n hn f4 ez 7t jw a1 9f 95 ae k9 jo ke 9e 6p 0k fm 36 8d 8g 8u 5h bi 48 84 ii dk jv 6c 3o hk 38 8z j6 8i 4b 2x 35 bv fk k5 2f 8j 3j 1o fa 6v ji dw hj b2 81 a5 42 29 8b gd 62 9h 9o
CHUNK: The result of this series vacuum tube power supply is a complete blocking of power supply harmonics, noise, hash, grain, and spikes

URI: data/decware_amp.txt
SCORE: 0.270
TOKEN: al ez aa hn 1e f4 95 9h 8o c4 1w 7r 3u ij 7f d0 0x bz 1t 62 a5 9j 1o 84 4n ke 5r 5h fm 2k fa 15 0p j7 2i 2h c2 ii jw km 42 8b hc ef 3c 35 81 em 2j 3q eq in j8 es jh a2 j6 dz k9 b7 4r 4y 29 0d
CHUNK: The basic model, an XLR balanced model, and this model, the SE84UFO3 which is unique because it consists of two individual amplifiers, one for each speaker

URI: data/decware_amp.txt
SCORE: 0.264
TOKEN: hn aa ij 3o a5 9h 7t 69 0v de f4 ez 2s 7o 81 8b jh fa gf bz c4 1b kg 3d 94 9o as 5g 7w 0d f6 dd 35 6j ii 6d kt 15 7r 6p 46 kf 16 7y k9 1w 62 5o a1 j6 eq ae fl 36 84 1e in 2m 8d em ht h9 8g 2h
CHUNK: There are 3 models of our 2 watt series of Zen Triode amplifier

URI: data/decware_amp.txt
SCORE: 0.243
TOKEN: ez al 8g dx ke 7c aa aw 8b 2x 3o 8i bi dp hn a5 7d a1 ii 1e 8j in ae 2i eo fg 4p fs 3a 62 57 d0 14 9h 3n 3b ey 59 4y de 74 ir jp 9q 7y 1f gb fl 5h ig eb 3j 0x 1o 4b cv 6p 2o 2n 6i j8 95 6d 1i
CHUNK: In the Zen Triode, the clean power from the vacuum tube is only inches away from the input stage which uses it, and there are no connectors or plugs or circuit boards to complicate matters

SQL query time: 42.68813133239746 ms


real	0m2.232s
user	0m4.602s
sys	0m2.669s
```

## How

See https://www.cockroachlabs.com/blog/use-cases-trigram-indexes/

Embeddings have some large dimensionality, like 768 or more.  Goal here is to
select the N most important dimensions, the ones with the largest magnitudes,
and generate a base36 token for each of these.

These will be concatenated with a space or, maybe, some other special
character between them and used as a token which will be indexed using a GIN with
the `gin_trgm_ops` option.

Likely, a second round of filtering using a regex type approach may boost relevance:
```
... AND chunk ~* '(termA|termB|...)'
```

The trigram index looks like this:
```
CREATE INDEX ON sentences USING GIN (embed_token gin_trgm_ops);
```

Python has [a base36 module](https://pypi.org/project/base36/)

```
$ pip install base36
```

Generate a zero-padded base36 string for a dimension in the embedding array:
```
import base36
base36.dumps(1).zfill(2)
```

Need to sort the dictionary of base36 key to float value from embedding array, by value.
Keep only the top n values.  Something like this:
```
n = 3
x = {"01": 0.123, "02": 0.922, "03": 0.456, "04": -0.999}
x_sorted = dict(sorted(x.items(), key=lambda item: abs(item[1]), reverse=True)[:n])
```

## References

* https://stackoverflow.blog/2023/11/09/an-intuitive-introduction-to-text-embeddings/
* https://huggingface.co/blog/bert-101
* https://huggingface.co/distilbert/distilbert-base-uncased
* https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/

