query = """
select u.user_id, t.track_id 
from user_track ut 
join users u on ut.user_id = u.user_id
join track t on t.track_id = ut.track_id
order by u.user_id;
"""