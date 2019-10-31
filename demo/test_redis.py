import redis 
import settings 


db = redis.StrictRedis(host=settings.REDIS_HOST,
    port=settings.REDIS_PORT, db=settings.REDIS_DB)
db.flushdb()

print('pok')
