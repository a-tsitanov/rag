from src.services.milvus import milvus_service
from src.services.neo4j import neo4j_service
from src.services.redis import redis_service


def get_redis():
    return redis_service


def get_milvus():
    return milvus_service


def get_neo4j():
    return neo4j_service
