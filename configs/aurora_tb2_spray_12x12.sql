-- One row per climb and angle. Aurora's display_difficulty is already the
-- benchmark override (when present) or the rounded community average.
SELECT
    c.uuid AS climb_id,
    cs.angle AS angle,
    CASE
        WHEN INSTR(dg.boulder_name, '/') > 0
            THEN SUBSTR(dg.boulder_name, 1, INSTR(dg.boulder_name, '/') - 1)
        ELSE dg.boulder_name
    END AS grade,
    c.frames AS frames,
    cs.ascensionist_count AS ascents,
    0 AS votes,
    c.uuid AS group_id
FROM climbs AS c
INNER JOIN climb_stats AS cs
    ON cs.climb_uuid = c.uuid
INNER JOIN difficulty_grades AS dg
    ON dg.difficulty = CAST(ROUND(cs.display_difficulty) AS INTEGER)
WHERE c.layout_id = 11
  AND c.frames_count = 1
  AND c.is_listed = 1
  AND c.is_draft = 0
  AND cs.angle IN (35, 40, 45, 50, 55)
  AND cs.ascensionist_count >= 3
ORDER BY c.uuid, cs.angle;

-- placements
-- Normalize the full 12x12 Spray coordinate extent to [0, 1]. Sets 12 and 13
-- are respectively the wood and plastic sets for product size 6.
SELECT
    p.id AS placement_id,
    (h.x + 64.0) / 128.0 AS x,
    (h.y - 4.0) / 136.0 AS y
FROM placements AS p
INNER JOIN holes AS h
    ON h.id = p.hole_id
WHERE p.layout_id = 11
  AND p.set_id IN (12, 13);
