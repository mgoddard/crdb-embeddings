# CockroachDB: Semantic Search

New in CockroachDB 24.2 is support for vectors.  Now, we can store vector
embeddings within CockroachDB with pgvector-compatible semantics to build
AI-driven applications. Numerous built-in functions have been added for running
similarity search across vectors.

The caveat with this is that vector indexing is not supported in this release.
Still, that won't stop us from experimenting with some semantic indexing and
search.  To make up for the lack of indexing on these vectors, we'll add a
K-Means clustering step to our app and create an index on the cluster ID values
which we'll map to the primary key if the table storing the data.  This gives
us a two phased approach to search: (1) use this index, (2) order the result set
according to each row's cosine similarity to the query string.

This demo can be run locally using the steps outlined below or in Kubernetes (K8s),
adjacent to a
[CockroachDB deployment](https://www.cockroachlabs.com/docs/stable/deploy-cockroachdb-with-kubernetes),
using the deployment defined [here](./k8s/crdb-embeddings.yaml).

## DB setup

* Install a CockroachDB instance, using v. 24.2.x
* Create a user account for the app
* GRANT this user/role access to the DB being used

## Configure environment variables

If deploying locally or on a VM, edit the `./env.sh` file; if deploying in K8s, these
variables are defined in the deployment YAML file (`./k8s/crdb-embeddings.yaml`).
Here is a detailed explanation of the various environment variables used by the app
and related scripts (search, index, ...):


Host name and port used by the app (a Python Flask REST service):
```
export FLASK_PORT=8080
export FLASK_HOST=some-host.domain
```

The DB connection string:
```
export DB_URL="postgres://some_role:password@127.0.0.1:26257/db_name?sslmode=require&sslrootcert=/certs/ca.crt"
```

Log level (INFO is fairly chatty):
```
export LOG_LEVEL=INFO
```

Number of threads to run in the Flask app:
```
export N_THREADS=10
```

The process of generating embeddings from text is expensive, so the embeddings
are cached for reuse (in case queries are repeated, etc.):
```
export CACHE_SIZE=1024
```

The minimum length of a sentence, in characters:
```
export MIN_SENTENCE_LEN=8
```

The number of clusters to create in the K-Means model.  The tradeoff here is
of query speed vs. recall; e.g. if the number is too low, more data must be
scanned during the cosine similarity phase.  If the number is too high, then
matching documents may be missed entirely as their cluster ID value will not
align with the cluster ID value in the query string.  For the small data size
used in my own experiments, 258209 rows, a value of 100 seemed like the best
fit:
```
export N_CLUSTERS=100
```

The fraction of rows to scan when building the K-Means model.  A value of 1.0
might make more sense for a small data set, but a smaller fraction would be
a better fit for a larger data set.  Again, there's a tradeoff here in terms
of fidelity, with a larger fraction (theoretically) leading to a better model:
```
export TRAIN_FRACTION=0.75
```

```
export MODEL_FILE=/tmp/model.pkl
```

```
export MODEL_FILE_URL="https://storage.googleapis.com/crl-goddard-text/model.pkl"
```

```
export BATCH_SIZE=512
```

```
export KMEANS_VERBOSE=1
```

```
export KMEANS_MAX_ITER=25
```

```
export SECRET="TextWithNoSpecialChars"
```

```
export BLOB_STORE_KEEP_N_ROWS=3
```

# For client
export MAX_RESULTS=5

## Start the Flask server process

```
$ ./ngram_embeddings.py -s
pg_trgm.similarity_threshold: 0.2 (set via 'export MIN_SIM=0.1')
n_threads: 1 (set via 'export N_THREADS=10')
Log level: WARN (export LOG_LEVEL=[DEBUG|INFO|WARN|ERROR] to change this)
```

## Index some documents

```
$ . ./env.sh
$ time ./index_doc.py ./data/*.txt
data/5am_club.txt: SUCCESS (t = 1010.720 ms)
data/aus_nz.txt: SUCCESS (t = 323.854 ms)
data/bafta_awards.txt: SUCCESS (t = 179.528 ms)
data/bourdain.txt: SUCCESS (t = 322.123 ms)
[...]
data/top_jazz_albums.txt: SUCCESS (t = 429.001 ms)
data/undoctored.txt: SUCCESS (t = 491.508 ms)
data/victoria.txt: SUCCESS (t = 539.466 ms)

real	0m33.947s
user	0m0.135s
sys	0m0.048s
```

## Query the index

```
$ q="vacuum tube triode amplifier sound"
$ time ./search_client.sh $q
[
  {
    "uri": "data/decware_amp.txt",
    "sim": 0.318,
    "token": "69 1d in fs 3n hn f4 ez 7t jw a1 9f 95 ae k9 jo ke 9e 6p 0k fm 36 8d 8g 8u 5h bi 48 84 ii dk jv 6c 3o hk 38 8z j6 8i 4b 2x 35 bv fk k5 2f 8j 3j 1o fa 6v ji dw hj b2 81 a5 42 29 8b gd 62 9h 9o ",
    "chunk": "The result of this series vacuum tube power supply is a complete blocking of power supply harmonics, noise, hash, grain, and spikes"
  }
]

real	0m0.480s
user	0m0.010s
sys	0m0.016s
```

## References

* https://stackoverflow.blog/2023/11/09/an-intuitive-introduction-to-text-embeddings/
* https://huggingface.co/blog/bert-101
* https://huggingface.co/distilbert/distilbert-base-uncased
* https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
* https://www.postgresql.org/docs/9.1/functions-array.html

