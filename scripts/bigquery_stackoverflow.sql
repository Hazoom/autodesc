pq.title as title,
pq.body as question_body,
pq.creation_date as creation_date,
pq.owner_user_id as owner_id,
pa.body as answer_body,
pq.tags as tags
FROM `bigquery-public-data.stackoverflow.posts_questions` as pq
join `bigquery-public-data.stackoverflow.posts_answers` as pa on pa.id = pq.accepted_answer_id
where pq.creation_date >= '2016-01-01'
and REGEXP_CONTAINS(pq.tags, 'pandas|tensorflow|keras|pytorch|torch|machine-learning|deep-learning|list|dictionary')
and REGEXP_CONTAINS(pq.tags, 'python')
and REGEXP_CONTAINS(pa.body, '<pre><code>')