# CockroachDB: Semantic Search

New in CockroachDB 24.2 is support for vectors.  Now, we can store vector
embeddings within CockroachDB with pgvector-compatible semantics to build
AI-driven applications. Numerous built-in functions have been added for running
similarity search across vectors.

The caveat with this is that vector indexing is not supported in this release.
Still, that won't stop us from experimenting with some semantic indexing and
search.  To make up for the lack of indexing on these vectors, we'll add a
K-Means clustering step to our app and create an index on the cluster ID values
which we'll map to the primary key of the table storing the data.  This gives
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

The number of clusters to create in the K-Means model.  The tradeoff here is of
query speed vs. recall; e.g. if the number is too low, more data must be
scanned during the cosine similarity phase.  If the number is too high, then
matching documents may be missed entirely as their cluster ID value will not
align with the cluster ID value that gets mapped to the query string.  For the
small data size used in my own experiments, 250k rows, a value of 100 seemed
like the best fit:
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

A filesystem location where the model can be read and written.  This is used
during the "bootstrap" phase when the model file is downloaded from the URL
provided below.  After that, the model is stored in the DB itself:
```
export MODEL_FILE=/tmp/model.pkl
```

See above.  This model was built according to the discussion above, but it
should be suitable for getting started:
```
export MODEL_FILE_URL="https://storage.googleapis.com/crl-goddard-text/model.pkl"
```

This applies during the process of assigning a cluster ID value to each of the rows.
The value of 512 shown here yielded the best data insert rate for me.
```
export BATCH_SIZE=512
```

When building the K-Means model, the verbosity level can be adjusted.  The value
of "1" here yields enough output so you can be convinced something is happening.
It can be disabled by setting this to "0":
```
export KMEANS_VERBOSE=1
```

Building a K-Means model is iterative.  This value of "25" reduced the model
build times yet resulted in a useful model.  Probably with experimenting with:
```
export KMEANS_MAX_ITER=25
```

This is just a string that's required as a URL parameter to the app's `/cluster_assign`
and `/build_model` endpoints since, if the app is exposed on a network, we'd rather
not have random clients hitting these.  Set it to any string containing no spaces or special
charaters:
```
export SECRET="TextWithNoSpecialChars"
```

The `blob_store` table gets a new row containing a model each time that `build_model`
process runs.  This table is pruned to ensure it has at most this number of rows:
```
export BLOB_STORE_KEEP_N_ROWS=3
```

The number of results retrieved and displayed as JSON by the client:
```
export MAX_RESULTS=5
```

## Start the Flask server process

```
[16:33:10 crdb-embeddings]$ . ./env.sh
[16:34:01 crdb-embeddings]$ ./pgvector_embeddings.py
kmeans_max_iter: 25 (set via 'export KMEANS_MAX_ITER=25')
kmeans_verbose: 1 (set via 'export KMEANS_VERBOSE=1')
batch_size: 512 (set via 'export BATCH_SIZE=512')
n_clusters : 100 (set via 'export N_CLUSTERS=50')
train_fraction: 0.75 (set via 'export TRAIN_FRACTION=0.10')
model_file: /tmp/model.pkl (set via 'export MODEL_FILE=./model.pkl')
model_url: https://storage.googleapis.com/crl-goddard-text/model.pkl (set via 'export MODEL_FILE_URL=https://somewhere.com/path/model.pkl')
min_sentence_len: 8 (set via 'export MIN_SENTENCE_LEN=12')
cache_size: 1024 (set via 'export CACHE_SIZE=1024')
n_threads: 10 (set via 'export N_THREADS=10')
max_retries: 3 (set via 'export MAX_RETRIES=3')
shared secret: TextWithNoSpecialChars
blob_store_keep_n_rows: 3
Log level: INFO (export LOG_LEVEL=[DEBUG|INFO|WARN|ERROR] to change this)
[08/22/2024 04:34:06 PM MainThread] BertTokenizer: 0.7987060546875 s
Model will run on cpu
[08/22/2024 04:34:09 PM MainThread] BertModel + eval: 2.566020965576172 s
[08/22/2024 04:34:09 PM MainThread] Checking whether text_embed table exists
[08/22/2024 04:34:09 PM MainThread] text_embed table already exists
/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/sqlalchemy_cockroachdb/base.py:226: SAWarning: Did not recognize type 'vector' of column 'embedding'
  warn(f"Did not recognize type '{type_name}' of column '{name}'")
/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/sqlalchemy_cockroachdb/base.py:226: SAWarning: Did not recognize type 'ARRAY' of column 'top_n'
  warn(f"Did not recognize type '{type_name}' of column '{name}'")
[08/22/2024 04:34:09 PM MainThread] Fetching model from the DB ...
[08/22/2024 04:34:09 PM MainThread] OK
[08/22/2024 04:34:09 PM MainThread] K-means model loaded
[08/22/2024 04:34:09 PM MainThread] You may need to update K-means cluster assignments by making a GET request to the /cluster_assign/TextWithNoSpecialChars endpoint.
[08/22/2024 04:34:09 PM MainThread] Serving on http://0.0.0.0:1972
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
[16:35:11 crdb-embeddings]$ q="which amplifiers have vacuum tube triode sound"
[16:38:08 crdb-embeddings]$ time ./search_client.sh $q
[
  {
    "uri": "data/decware_amp.txt",
    "sim": 0.8592629666081919,
    "chunk": "There are 3 models of our 2 watt series of Zen Triode amplifier"
  },
  {
    "uri": "data/muse_electronics.txt",
    "sim": 0.8521749669893637,
    "chunk": "This balanced circuitry makes converting the amplifier to balanced inputs simply a matter of changing the RCA input connector to an XLR"
  },
  {
    "uri": "tmp/wiki_pages/klmán_kandó.txt",
    "sim": 0.8520580730724183,
    "chunk": "Series motors, and why 1 phase on the supply? The series motor needs 16 1/3 Hz network"
  },
  {
    "uri": "tmp/wiki_pages/liquid_scintillation_counting.txt",
    "sim": 0.8492422839608895,
    "chunk": "Many counters have two photo multiplier tubes connected in a coincidence circuit"
  },
  {
    "uri": "tmp/wiki_pages/amphenol.txt",
    "sim": 0.8482197822951335,
    "chunk": "Schmitt, whose first product was a tube socket for Vacuum tube|radio tubes (valveholder bases)"
  }
]

real	0m0.190s
user	0m0.012s
sys	0m0.014s
```

## Rebuild the K-Means model

This takes a while (it is influenced by the `TRAIN_FRACTION` value).

```
[16:55:59 crdb-embeddings]$ . ./env.sh
[16:56:02 crdb-embeddings]$ ./build_model.sh $SECRET
```

## Refresh the cluster ID to row assignments

This also takes a while, but is necessary after the model is rebuilt.

```
[16:57:43 crdb-embeddings]$ . ./env.sh
[16:57:50 crdb-embeddings]$ ./cluster_assign.sh $SECRET
```

## References

* https://www.cockroachlabs.com/docs/releases/v24.2#v24-2-0-sql
* https://stackoverflow.blog/2023/11/09/an-intuitive-introduction-to-text-embeddings/
* https://huggingface.co/blog/bert-101
* https://huggingface.co/distilbert/distilbert-base-uncased
* https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/

