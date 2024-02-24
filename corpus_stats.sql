/* Generate what could be used as a stop word list */
WITH a AS
(
  SELECT REGEXP_SPLIT_TO_TABLE(LOWER(chunk), E'\\W+') term
  FROM text_embed
)
SELECT term, COUNT(*) n
FROM a
WHERE
  LENGTH(term) > 1
  AND term !~ E'^\\d+$'
GROUP BY term
ORDER BY n DESC, term ASC
LIMIT 20;

