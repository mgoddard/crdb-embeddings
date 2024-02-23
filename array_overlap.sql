
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

