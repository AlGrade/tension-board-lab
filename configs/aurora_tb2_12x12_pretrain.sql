-- Pretraining rows with at least one community grade.
SELECT
    c.uuid AS climb_id,
    c.layout_id AS layout_id,
    CASE c.layout_id
        WHEN 10 THEN 'mirror'
        WHEN 11 THEN 'spray'
    END AS source_layout,
    cs.angle AS angle,
    CASE
        WHEN INSTR(dg.boulder_name, '/') > 0
            THEN SUBSTR(dg.boulder_name, INSTR(dg.boulder_name, '/') + 1)
        ELSE dg.boulder_name
    END AS grade,
    cs.difficulty_average AS difficulty_average,
    c.frames AS frames,
    cs.ascensionist_count AS ascents
FROM climbs AS c
INNER JOIN climb_stats AS cs
    ON cs.climb_uuid = c.uuid
INNER JOIN difficulty_grades AS dg
    ON dg.difficulty = CAST(ROUND(cs.display_difficulty) AS INTEGER)
WHERE c.layout_id IN (10, 11)
  AND c.frames_count = 1
  AND c.is_listed = 1
  AND c.is_draft = 0
  AND c.edge_left >= -68
  AND c.edge_right <= 68
  AND c.edge_bottom >= 0
  AND c.edge_top <= 144
  AND cs.angle IN (35, 40, 45, 50, 55)
  AND cs.ascensionist_count >= 1
ORDER BY c.layout_id, c.uuid, cs.angle;

-- placements
SELECT
    p.layout_id AS layout_id,
    p.id AS placement_id,
    p.set_id AS set_id,
    h.x AS raw_x,
    h.y AS raw_y,
    (h.x + 64.0) / 128.0 AS x,
    (h.y - 4.0) / 136.0 AS y
FROM placements AS p
INNER JOIN holes AS h
    ON h.id = p.hole_id
WHERE p.layout_id IN (10, 11)
  AND p.set_id IN (12, 13);
