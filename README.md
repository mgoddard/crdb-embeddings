# CockroachDB: Semantic Search

New in CockroachDB 24.2 is support for vectors.  Now we can store vector
embeddings within CockroachDB with pgvector-compatible semantics to build
AI-driven applications. Numerous built-in functions have been added for running
similarity search across vectors.  Here, we use the `<=>` (cosine similarity)
operator.

The caveat with this is that vector indexing is not supported in this release.
Still, that won't stop us from experimenting with some semantic indexing and
search.  To make up for the lack of indexing on these vectors, we'll add a
K-Means clustering step to our app and create an index on the cluster ID values
which we'll map to the primary key of the table storing the data.  This gives
us a two phased approach to search: (1) use this index, (2) order the resulting
rows according to each row's cosine similarity to the query string.

This demo can be run locally using the steps outlined below or in Kubernetes (K8s),
adjacent to a
[CockroachDB deployment](https://www.cockroachlabs.com/docs/stable/deploy-cockroachdb-with-kubernetes),
using the deployment defined [here](./k8s/crdb-embeddings.yaml).

## DB setup

* Install a CockroachDB instance, using version 24.2+
* Apply a CockroachDB Enterprise license
* Create a user account for the app
* GRANT this user/role access to the DB being used
* The app creates all the required tables and indexes

## Configure environment variables

The **simplest way to get started** is to run the Docker image. To do that, edit the
`./docker_run_image.sh` file, adjusting the lines near the top as necessary:
```
# EDIT THESE TO SUIT YOUR DEPLOYMENT
crdb_certs="$HOME/certs"
export DB_URL="postgres://test_role:123abc@host.docker.internal:26257/defaultdb?sslmode=require&sslrootcert=$crdb_certs/ca.crt"
port=1999 # You point your client at this port on localhost, so 'export FLASK_PORT=1999' in env.sh
```
You can then skip down to [Start the Flask server Docker image](#Start-the-Flask-server-Docker-image).

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

The minimum length of a sentence, in characters:
```
export MIN_SENTENCE_LEN=8
```

The number of clusters to create in the K-Means model.  The tradeoff here is
query speed vs. recall; e.g. if the number is too low, more data must be
scanned during the cosine similarity phase.  If the number is too high, then
matching documents may be missed entirely as their cluster ID value will not
align with the cluster ID value that gets mapped to the query string.  For the
small data size used in these experiments, 50k rows, a value of 500 seemed
like the best fit:
```
export N_CLUSTERS=500
```

The fraction of rows to scan when building the K-Means model.  A value of 1.0
might make more sense for a small data set, but a smaller fraction would be
better suited to a larger data set.  Again, there's a tradeoff here in terms
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

See above.  This model was built according to the discussion above, and it
should be suitable for getting started:
```
export MODEL_FILE_URL="https://storage.googleapis.com/crl-goddard-text/model_Fastembed_500.pkl"
```

This applies during the process of assigning a cluster ID value to each of the rows.
The value of 512 shown here yielded the best data insert rate:
```
export BATCH_SIZE=512
```

When building the K-Means model, the verbosity level can be adjusted.  The value
of "1" here yields enough output so you can be convinced something is happening.
It can be disabled by setting this to "0":
```
export KMEANS_VERBOSE=1
```

Building a K-Means model is iterative.  This value of "100" seems to work well enough:
```
export KMEANS_MAX_ITER=100
```

When starting out without a K-Means model (if not using the `MODEL_FILE_URL`, for
example), this would be set to `True` and then the documents could be added using
`index_doc.py` and then the model would be built as shown below.  Then, the app
would be restarted with this value reset to `False` and the cluster assignment step
would need to be done prior to running a search.
```
export SKIP_KMEANS=False
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

## Start the Flask server Docker image

```
[13:22:56 crdb-embeddings]$ ./docker_run_image.sh
1.26: Pulling from mgoddard/crdb-embeddings-arm
Digest: sha256:a83a9d8cd33342eba584ffa08a72660ad4595171ee8e4dfa78bf9a899de2315e
Status: Image is up to date for mgoddard/crdb-embeddings-arm:1.26
docker.io/mgoddard/crdb-embeddings-arm:1.26
Collecting psycopg2-binary
  Downloading psycopg2_binary-2.9.9-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl (2.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.9/2.9 MB 16.6 MB/s eta 0:00:00
[... Many more such lines removed here for brevity ...]
Building wheels for collected packages: PyStemmer
  Building wheel for PyStemmer (setup.py): started
  Building wheel for PyStemmer (setup.py): finished with status 'done'
  Created wheel for PyStemmer: filename=PyStemmer-2.2.0.1-cp38-cp38-linux_aarch64.whl size=612716 sha256=9b9eb0b5d11052fff130237e588c074cef726edcd7dcc246301a43d28a6478fe
  Stored in directory: /tmp/pip-ephem-wheel-cache-aj__ssz4/wheels/78/04/32/a81f10f01775fcadba622dbcf8305f8053ab1db21b20a25fc4
Successfully built PyStemmer
Installing collected packages: snowballstemmer, PyStemmer, mpmath, mmh3, flatbuffers, zipp, waitress, urllib3, typing-extensions, tqdm, threadpoolctl, sympy, pyyaml, psycopg2-binary, protobuf, pillow, packaging, numpy, MarkupSafe, loguru, joblib, itsdangerous, idna, humanfriendly, greenlet, fsspec, filelock, click, charset-normalizer, certifi, Werkzeug, SQLAlchemy, scipy, requests, psycopg2-pool, pgvector, onnx, Jinja2, importlib-metadata, coloredlogs, sqlalchemy-cockroachdb, scikit-learn, onnxruntime, huggingface-hub, Flask, tokenizers, fastembed
Successfully installed Flask-2.1.0 Jinja2-3.1.4 MarkupSafe-2.1.5 PyStemmer-2.2.0.1 SQLAlchemy-2.0.25 Werkzeug-2.2.2 certifi-2024.7.4 charset-normalizer-3.3.2 click-8.1.7 coloredlogs-15.0.1 fastembed-0.3.6 filelock-3.15.4 flatbuffers-24.3.25 fsspec-2024.6.1 greenlet-3.0.3 huggingface-hub-0.24.6 humanfriendly-10.0 idna-3.8 importlib-metadata-8.4.0 itsdangerous-2.2.0 joblib-1.4.2 loguru-0.7.2 mmh3-4.1.0 mpmath-1.3.0 numpy-1.24.4 onnx-1.16.2 onnxruntime-1.19.0 packaging-24.1 pgvector-0.3.2 pillow-10.4.0 protobuf-5.27.4 psycopg2-binary-2.9.9 psycopg2-pool-1.2 pyyaml-6.0.2 requests-2.32.3 scikit-learn-1.3.2 scipy-1.10.1 snowballstemmer-2.2.0 sqlalchemy-cockroachdb-2.0.1 sympy-1.13.2 threadpoolctl-3.5.0 tokenizers-0.20.0 tqdm-4.66.5 typing-extensions-4.12.2 urllib3-2.2.2 waitress-3.0.0 zipp-3.20.1
kmeans_max_iter: 250 (set via 'export KMEANS_MAX_ITER=25')
kmeans_verbose: 1 (set via 'export KMEANS_VERBOSE=1')
skip_kmeans: False (set via 'export SKIP_KMEANS=False')
batch_size: 768 (set via 'export BATCH_SIZE=512')
n_clusters : 1000 (set via 'export N_CLUSTERS=50')
train_fraction: 0.5 (set via 'export TRAIN_FRACTION=0.10')
model_file: /tmp/model.pkl (set via 'export MODEL_FILE=./model.pkl')
model_url: https://storage.googleapis.com/crl-goddard-text/model_Fastembed_1k.pkl (set via 'export MODEL_FILE_URL=https://somewhere.com/path/model.pkl')
min_sentence_len: 8 (set via 'export MIN_SENTENCE_LEN=12')
n_threads: 10 (set via 'export N_THREADS=10')
max_retries: 3 (set via 'export MAX_RETRIES=3')
shared secret: TextWithNoSpecialChars
blob_store_keep_n_rows: 3
Log level: INFO (export LOG_LEVEL=[DEBUG|INFO|WARN|ERROR] to change this)
Fetching 5 files: 100%|██████████| 5/5 [00:03<00:00,  1.48it/s]
[08/28/2024 05:27:56 PM MainThread] TextEmbedding model ready: 4.718861103057861 s
[08/28/2024 05:27:56 PM MainThread] Checking whether text_embed table exists
[08/28/2024 05:27:57 PM MainThread] text_embed table already exists
/usr/local/lib/python3.8/site-packages/sqlalchemy_cockroachdb/base.py:226: SAWarning: Did not recognize type 'vector' of column 'embedding'
  warn(f"Did not recognize type '{type_name}' of column '{name}'")
[08/28/2024 05:27:57 PM MainThread] Fetching model from the DB ...
/usr/local/lib/python3.8/site-packages/sklearn/base.py:348: InconsistentVersionWarning: Trying to unpickle estimator KMeans from version 1.5.1 when using version 1.3.2. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
  warnings.warn(
[08/28/2024 05:27:58 PM MainThread] OK
[08/28/2024 05:27:58 PM MainThread] K-means model loaded
[08/28/2024 05:27:58 PM MainThread] You may need to update K-means cluster assignments by making a GET request to the /cluster_assign/TextWithNoSpecialChars endpoint.
[08/28/2024 05:27:58 PM MainThread] Serving on http://0.0.0.0:18080

huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
```

## Index some documents

**Note:** This process is time consuming and would benefit from being parallelized.
The app itself is multi-threaded, and CockroachDB scales very nicely, so altering
`./index_doc.py` so it can submit multiple indexing requests in parallel is all that
has to be done.

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

**Caveat:** If your data set is small, the default model will perform better.
For now, if you perform this step and want to roll back to the default model,
perform the actions listed under "Rollback to bootstrap model", below.

This takes a while (the `TRAIN_FRACTION` value affects the time).

```
[16:55:59 crdb-embeddings]$ . ./env.sh
[16:56:02 crdb-embeddings]$ ./build_model.sh $SECRET
```

## Refresh the cluster ID to row assignments

This also takes a while, but is **necessary after the model is rebuilt**.
If the search results don't make sense, it's likely you need to run this.

```
[16:57:43 crdb-embeddings]$ . ./env.sh
[16:57:50 crdb-embeddings]$ ./cluster_assign.sh $SECRET
```

## Rollback to bootstrap model

In the database, truncate the `blob_store` table:
```
defaultdb=> TRUNCATE TABLE blob_store;
```

Remove the existing model on the filesystem:
```
[18:49:47 crdb-embeddings]$ . ./env.sh
[18:49:53 crdb-embeddings]$ rm -f $MODEL_FILE
```

Restart the app

## References

* https://www.cockroachlabs.com/docs/releases/v24.2#v24-2-0-sql
* https://stackoverflow.blog/2023/11/09/an-intuitive-introduction-to-text-embeddings/
* https://huggingface.co/blog/bert-101
* https://huggingface.co/distilbert/distilbert-base-uncased
* https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
* https://github.com/qdrant/fastembed
* [Models supported by Fastembed](https://qdrant.github.io/fastembed/examples/Supported_Models/#supported-text-embedding-models)
* https://github.com/pgvector/pgvector-python?tab=readme-ov-file#psycopg-2
* [Example CockroachDB query plan](./test/plan.txt) for these semantic searches
* [JSON Wikipedia dump from 2020](https://www.kaggle.com/datasets/ltcmdrdata/plain-text-wikipedia-202011)
* https://docs.sqlalchemy.org/en/20/core/connections.html#streaming-with-a-dynamically-growing-buffer-using-stream-results

