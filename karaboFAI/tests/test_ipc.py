import unittest
import time

from redis.client import PubSub, Redis

from karaboFAI.logger import logger
from karaboFAI.services import start_redis_server
from karaboFAI.ipc import (
    redis_connection, RedisConnection, RedisSubscriber, RedisPSubscriber,
    _global_connections, reset_redis_connections, MetaRedisConnection
)
from karaboFAI.pipeline.worker import ProcessWorker
from karaboFAI.processes import wait_until_redis_shutdown

logger.setLevel("CRITICAL")


class TestRedisConnection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        start_redis_server()

    @classmethod
    def tearDownClass(cls):
        wait_until_redis_shutdown()

    def setUp(self):
        db = redis_connection()
        for c in db.client_list():
            db.client_kill_filter(c["id"])

        reset_redis_connections()

    def testCreateConnection(self):
        client_list = redis_connection().client_list()
        n_clients = len(client_list)
        # why n_clients == 2 here?
        self.assertEqual(2, n_clients)

        db1 = redis_connection()
        self.assertTrue(db1.ping())
        db2 = redis_connection()
        self.assertTrue(db2.ping())
        self.assertIs(db1, db2)
        # expect new client being created
        n_clients += 1
        self.assertEqual(n_clients, len(db1.client_list()))

        db1_bytes = redis_connection(decode_responses=False)
        self.assertTrue(db1_bytes.ping())
        db2_bytes = redis_connection(decode_responses=False)
        self.assertTrue(db2_bytes.ping())
        self.assertIs(db1_bytes, db2_bytes)
        # expect to have one more client without decode response
        n_clients += 1
        self.assertEqual(n_clients, len(db1_bytes.client_list()))

    def testCreateConnectionLazily(self):
        class Host:
            db = RedisConnection()

        n_clients = len(redis_connection().client_list())
        host = Host()
        self.assertEqual(n_clients, len(redis_connection().client_list()))
        host.db.ping()
        # connection was already created when the server was started
        self.assertEqual(n_clients, len(redis_connection().client_list()))

    def testCreateConnectionInProcess(self):
        class DumpyProcess(ProcessWorker):
            def run(self):
                """Override."""
                self._db.ping()
                time.sleep(0.1)

        n_clients = len(redis_connection().client_list())

        proc = DumpyProcess('dummy')
        proc.start()
        # child process share the same connection as that in the main process
        self.assertEqual(n_clients, len(redis_connection().client_list()))

        proc.join()

    def testCreatePubSub(self):
        class SubHost:
            sub = RedisSubscriber('abc')

        class PSubHost:
            sub = RedisPSubscriber('abc')

        n_clients = len(redis_connection().client_list())
        sub_host = SubHost()
        self.assertEqual(n_clients, len(redis_connection().client_list()))
        # add a connection when first called
        sub_host.sub.get_message()
        n_clients += 1
        self.assertEqual(n_clients, len(redis_connection().client_list()))

        psub_host = PSubHost()
        self.assertEqual(n_clients, len(redis_connection().client_list()))
        # add a connection when first called
        psub_host.sub.get_message()
        n_clients += 1
        self.assertEqual(n_clients, len(redis_connection().client_list()))

    def testTrackingConnections(self):
        self.assertTrue(bool(_global_connections))
        # clear the current registrations
        _global_connections.clear()
        self.assertFalse(bool(_global_connections))

        class Host:
            db = RedisConnection()

        c1 = RedisConnection()
        self.assertListEqual(['RedisConnection'], list(_global_connections.keys()))
        self.assertIsInstance(_global_connections['RedisConnection'][0](), RedisConnection)

        class SubHost:
            sub = RedisSubscriber('abc')

        self.assertListEqual(['RedisConnection', 'RedisSubscriber'],
                             list(_global_connections.keys()))
        self.assertIsInstance(_global_connections['RedisConnection'][0](), RedisConnection)
        self.assertIsInstance(_global_connections['RedisSubscriber'][0](), RedisSubscriber)

        class PSubHost1:
            sub = RedisPSubscriber('abc')

        class PSubHost2:
            sub = RedisPSubscriber('abc')

        self.assertListEqual(['RedisConnection', 'RedisSubscriber', 'RedisPSubscriber'],
                             list(_global_connections.keys()))
        self.assertIsInstance(_global_connections['RedisConnection'][0](), RedisConnection)
        self.assertIsInstance(_global_connections['RedisSubscriber'][0](), RedisSubscriber)
        self.assertIsInstance(_global_connections['RedisPSubscriber'][0](), RedisPSubscriber)
        self.assertIsInstance(_global_connections['RedisPSubscriber'][1](), RedisPSubscriber)
        self.assertIsNot(_global_connections['RedisPSubscriber'][0](),
                         _global_connections['RedisPSubscriber'][1]())

        # test reset_redis_connections

        Host().db
        self.assertIsInstance(_global_connections['RedisConnection'][0]()._db, Redis)
        SubHost().sub
        self.assertIsInstance(_global_connections['RedisSubscriber'][0]()._sub, PubSub)
        PSubHost1().sub
        self.assertIsInstance(_global_connections['RedisPSubscriber'][0]()._sub, PubSub)
        PSubHost2().sub
        self.assertIsInstance(_global_connections['RedisPSubscriber'][1]()._sub, PubSub)
        reset_redis_connections()
        self.assertIsNone(_global_connections['RedisConnection'][0]()._db)
        self.assertIsNone(_global_connections['RedisSubscriber'][0]()._sub)
        self.assertIsNone(_global_connections['RedisPSubscriber'][0]()._sub)
        self.assertIsNone(_global_connections['RedisPSubscriber'][0]()._sub)
