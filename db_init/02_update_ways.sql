-- Add column blocked for ways table
ALTER TABLE ways ADD COLUMN blocked BOOLEAN DEFAULT FALSE;
UPDATE ways 
SET blocked = TRUE 
FROM configuration c
WHERE ways.tag_id = c.tag_id
    AND c.tag_key IN ('access', 'vehicle', 'motor_vehicle')
    AND c.tag_value IN ('no', 'private');