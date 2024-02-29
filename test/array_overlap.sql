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

/* Test the scoring approach */
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

/* Integrate this JSONB field scoring */
WITH qt AS
(
  SELECT uri, SIMILARITY(%s, token)::NUMERIC(4, 3) sim, token, chunk, svec
  FROM text_embed@text_embed_token_idx
  WHERE %s %% token
  ORDER BY sim DESC
  LIMIT %s
), q1 AS
(
  /* Both args here are the JSONB representing the search query */
  SELECT (json_each_text(%s)).@1 k, (json_each_text(%s)).@2 v
), q2 AS
(
  SELECT (json_each_text(svec)).@1 k, (json_each_text(svec)).@2 v
  FROM qt /* Do we need a WHERE clause here? */
), q3 AS
(
  SELECT q1.v * q2.v score
  WHERE q1.k = q2.k
)
SELECT qt.uri, qt.sim, qt.token, qt.chunk, q3.score
FROM qt, q3
ORDER BY q3.score DESC;

/* Try to rewrite using subselects? */
CREATE OR REPLACE FUNCTION score_row (q JSONB, r JSONB)
RETURNS FLOAT
LANGUAGE SQL
AS $$
  SELECT COALESCE(SUM(qv * rv), 0.0) score
  FROM (
    SELECT
      (json_each_text(q)).@1 qk
      , ((json_each_text(q)).@2)::float qv
      , (json_each_text(r)).@1 rk
      , ((json_each_text(r)).@2)::float rv
  )
  WHERE qk = rk;
$$;

SELECT film_id, title, length, rating
FROM film f
WHERE length > (
    SELECT AVG(length)
    FROM film
    WHERE rating = f.rating
);

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

