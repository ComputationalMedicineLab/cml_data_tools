-- SQL for creating the example tables used in the ICA example script

DROP TABLE cml_test_measurement IF EXISTS;
DROP TABLE cml_test_measurement_meta IF EXISTS;

DROP TABLE cml_test_condition IF EXISTS;
DROP TABLE cml_test_condition_meta IF EXISTS;

DROP TABLE cml_test_medication IF EXISTS;
DROP TABLE cml_test_medication_meta IF EXISTS;

DROP TABLE cml_test_sex IF EXISTS;
DROP TABLE cml_test_race IF EXISTS;
DROP TABLE cml_test_age IF EXISTS;
DROP TABLE cml_test_bmi IF EXISTS;
DROP TABLE cml_test_ana IF EXISTS;


-- Measurement
CREATE TABLE cml_test_measurement AS
SELECT DISTINCT
    A.grid AS id,
    to_char(B.measurement_date, 'YYYY-MM-DD') AS date,
    C.concept_code AS channel,
    last_value(B.value_as_number) OVER (
        PARTITION BY id, channel, date
        ORDER BY B.measurement_datetime ROWS
        BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS value
FROM SLE_IDS A
JOIN V_MEASUREMENT B ON (A.person_id = B.person_id)
JOIN MEASUREMENT_STATS C ON (B.measurement_concept_id = C.measurement_concept_id)
WHERE (
    B.value_as_number IS NOT NULL
    AND date >= '2000-01-01'
    AND C.count >= 1000
    AND (C.p01 <= 0 OR B.value_as_number > 0)
    AND (B.value_as_number < C.high_threshold)
    AND channel != '29463-7'
    AND channel != '8302-2'
    AND channel != '39156-5'
)
ORDER BY id, date;

-- Drop measurements that appear in fewer than 10 patients records
DELETE FROM CML_TEST_MEASUREMENT A
WHERE A.channel IN (
    SELECT channel
    FROM CML_TEST_MEASUREMENT
    GROUP BY channel
    HAVING count(distinct id) < 10
);

CREATE TABLE cml_test_measurement_meta AS
SELECT
    concept_code AS channel,
    concept_name AS description,
    p50 AS fill
FROM MEASUREMENT_STATS
WHERE (
    count >= 1000
    AND channel != '29463-7'
    AND channel != '8302-2'
    AND channel != '39156-5'
)
ORDER BY channel;


-- Condition
CREATE TABLE cml_test_condition AS
SELECT DISTINCT
    A.grid AS id,
    to_char(B.condition_start_date, 'YYYY-MM-DD') AS date,
    C.concept_code AS channel,
    '' AS value
FROM SLE_IDS A
JOIN V_CONDITION_OCCURRENCE B ON (A.person_id = B.person_id)
JOIN COUNTS C ON (B.condition_concept_id = C.concept_id)
WHERE (
    C.count >= 1000
    AND C.domain_id = 'Condition'
    AND date >= '2000-01-01'
)
ORDER BY id, date;

-- Fill value is 1 / (20 * 365.25)
CREATE TABLE cml_test_condition_meta AS
SELECT
    concept_code as channel,
    concept_name as description,
    0.00013689253935660506 AS fill
FROM COUNTS
WHERE domain_id = 'Condition' AND count >= 1000
ORDER BY channel;


-- Medications
CREATE TABLE cml_test_medication AS
SELECT DISTINCT
    A.grid AS id,
    to_char(B.drug_exposure_start_date, 'YYYY-MM-DD') AS date,
    C.concept_code AS channel,
    '' AS value
FROM SLE_IDS A
JOIN V_DRUG_EXPOSURE B ON (B.person_id = A.person_id)
JOIN V_X_DRUG_EXPOSURE X ON (B.drug_exposure_id = X.drug_exposure_id)
JOIN V_CONCEPT_ANCESTOR CA ON (CA.descendant_concept_id = B.drug_concept_id)
JOIN V_CONCEPT C on (CA.ancestor_concept_id = C.concept_id)
JOIN INGREDIENT_STATS I on (CA.ancestor_concept_id = I.ingredient_concept_id)
WHERE (
    C.concept_class_id = 'Ingredient'
    AND date >= '2000-01-01'
    AND I.count >= 1000
    AND (nullif(X.x_strength, '') IS NOT NULL OR nullif(X.x_dose, '') IS NOT NULL)
    AND nullif(X.x_frequency, '') IS NOT NULL
    AND B.route_source_value IS NOT NULL
    AND X.x_doc_stype = 'Problem list'
)
ORDER BY id, date, channel;

CREATE TABLE cml_test_medication_meta AS
SELECT DISTINCT
    ingredient_code AS channel,
    ingredient_name AS description,
    0.0 AS fill
FROM INGREDIENT_STATS
WHERE count >= 1000
ORDER BY channel;


-- Demographics
-- Sex
CREATE TABLE cml_test_sex AS
SELECT
    A.grid AS id,
    NULL AS date,
    C.concept_code AS channel,
    NULL AS value
FROM SLE_IDS A, V_PERSON P, V_CONCEPT C
WHERE P.person_id = A.person_id AND P.gender_concept_id = C.concept_id
ORDER BY id;

-- Race
CREATE TABLE cml_test_race AS
SELECT
    A.grid AS id,
    NULL AS date,
    CASE
        WHEN P.race_source_value LIKE '%,%' THEN 'M'
        WHEN P.race_source_value = ' ' THEN 'U'
        ELSE coalesce(P.race_source_value, 'U')
    END AS channel,
    NULL AS value
FROM V_PERSON P
RIGHT JOIN SLE_IDS A ON (P.person_id = A.person_id)
ORDER BY id;

-- Age
CREATE TABLE cml_test_age AS
SELECT
    A.grid AS id,
    NULL AS date,
    'age' AS channel,
    DATE(P.birth_datetime) AS value
FROM V_PERSON P
RIGHT JOIN SLE_IDS A ON (P.person_id = A.person_id)
ORDER BY id;

-- BMI
CREATE TABLE cml_test_bmi AS
SELECT DISTINCT
    A.grid AS id,
    to_char(B.measurement_date, 'YYYY-MM-DD') AS date,
    'BMI' AS channel,
    last_value(B.value_as_number) over (
        PARTITION BY A.grid, B.measurement_date
        ORDER BY B.measurement_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS value
FROM SLE_IDS A
JOIN V_MEASUREMENT B ON (A.person_id = B.person_id)
JOIN V_X_VS_BMI_CLEAN c ON (B.measurement_id = C.measurement_id)
WHERE (
    B.value_as_number IS NOT NULL
    AND C.x_is_cleaned='Y'
    AND date >= '2000-01-01'
)
ORDER BY id, date;

-- ANA
CREATE TABLE cml_test_ana AS
SELECT DISTINCT
    P.grid AS id,
    to_char(P.entry_date, 'YYYY-MM-DD') AS date,
    'ANA titer' AS channel,
    last_value(P.TITER) OVER (
        PARTITION BY p.grid, date
        ORDER BY P.TITER
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS value
FROM ANA P
RIGHT JOIN SLE_IDS G ON (P.person_id = G.person_id)
WHERE date >= '2000-01-01'
ORDER BY id, date;
