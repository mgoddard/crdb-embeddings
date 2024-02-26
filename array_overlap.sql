/* Container to use when returning results of JSON_EACH_TEXT(svec) */
CREATE TYPE IF NOT EXISTS kv AS (k STRING, v FLOAT);

/* This is the syntax I was looking for (Shaun provided) */
WITH a AS
(
  SELECT json_each_text(svec) jet
  FROM text_embed
  WHERE
    uri = 'data/muse_electronics.txt'
    AND chunk_num = 0
)
SELECT (jet).@1 k, (jet).@2 v
FROM a
LIMIT 10;

/* Test the idea */
WITH q AS
(
  SELECT uri, (json_each_text(svec)::kv).k k, (json_each_text(svec)::kv).v v
  FROM text_embed
  WHERE
    uri = 'data/muse_electronics.txt'
    AND chunk_num = 0
  LIMIT 10
),
r AS
(
  SELECT (json_each_text(svec)::kv).k k, (json_each_text(svec)::kv).v v
  FROM text_embed
)
SELECT uri, q.k key, q.v*r.v score
FROM q, r
WHERE q.k = r.k
ORDER BY score DESC
LIMIT 20;

/*
  Scoring function based on values in JSONB svec column
  ERROR:  unimplemented: CTE usage inside a function definition
 */
CREATE OR REPLACE FUNCTION score_row (q JSONB, r JSONB)
RETURNS FLOAT
LANGUAGE SQL
AS $$
  WITH qq AS
  (
    SELECT (json_each_text(q)::kv).k k, (json_each_text(q)::kv).v v
  ),
  rr AS
  (
    SELECT (json_each_text(r)::kv).k k, (json_each_text(r)::kv).v v
  )
  SELECT sum(qq.v * rr.v) score
  FROM qq, rr
  WHERE qq.k = rr.k;
$$;

/* This will produce a count of overlapping keys in two JSON values */
CREATE OR REPLACE FUNCTION key_overlap (a JSONB, b JSONB)
RETURNS INT
LANGUAGE SQL
AS $$
  SELECT COUNT(*)
  FROM (
    SELECT JSONB_OBJECT_KEYS(a) INTERSECT SELECT JSONB_OBJECT_KEYS(b)
  );
$$;

/* This produces the count of overlapping elements of the given string arrays */
CREATE OR REPLACE FUNCTION overlap(a STRING[], b STRING[])
RETURNS INT
LANGUAGE SQL
AS $$
  SELECT COUNT(*)
  FROM (
    SELECT UNNEST(a) INTERSECT SELECT UNNEST(b)
  );
$$;

/*

defaultdb=> select overlap('{9s, al, 69, cd, in, gs, 9h, ij}'::string[], '{in, 9h, cd, gr, 0b, gb, 8h, b8, 7y, 4y, gs}'::string[]);
 overlap
---------
       4
(1 row)

Time: 3.493 ms

 */

