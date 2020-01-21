import unittest
import time

from redis.client import PubSub, Redis

from extra_foam.config import config
from extra_foam.logger import logger
from extra_foam.services import start_redis_server
from extra_foam.ipc import (
    redis_connection, RedisConnection, RedisSubscriber, RedisPSubscriber,
    _global_connections, reset_redis_connections
)
from extra_foam.pipeline.worker import ProcessWorker
from extra_foam.processes import wait_until_redis_shutdown

logger.setLevel("CRITICAL")


class TestRedisConnection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        start_redis_server()

    @classmethod
    def tearDownClass(cls):
        wait_until_redis_shutdown()

    def setUp(self):
        self._db = redis_connection()

        for c in self._db.client_list():
            self._db.execute_command("CLIENT KILL", "ID", c["id"])

        reset_redis_connections()

    def testCreateConnection(self):
        client_list = self._db.client_list()
        n_clients = len(client_list)
        self.assertEqual(1, n_clients)

        db = redis_connection()
        self.assertTrue(db.ping())
        # _GLOBAL_REDIS_CONNECTION was set to None in "setUp", so new pool is created.
        self.assertIsNot(db, self._db)
        n_clients += 1
        self.assertEqual(n_clients, len(self._db.client_list()))

        # _GLOBAL_REDIS_CONNECTION_BYTES was set to None in "setUp", so new pool is created
        db1_bytes = redis_connection(decode_responses=False)
        self.assertTrue(db1_bytes.ping())
        db2_bytes = redis_connection(decode_responses=False)
        self.assertTrue(db2_bytes.ping())
        self.assertIs(db1_bytes, db2_bytes)
        n_clients += 1
        self.assertEqual(n_clients, len(db1_bytes.client_list()))

    def testCreateConnectionLazily(self):
        class Host:
            db = RedisConnection()

        n_clients = len(self._db.client_list())

        host = Host()  # lazily created client
        self.assertEqual(n_clients, len(self._db.client_list()))

        host.db.ping()  # new pool is created
        n_clients += 1
        # connection was already created when the server was started
        self.assertEqual(n_clients, len(self._db.client_list()))

    def testCreateConnectionInProcess(self):
        class DumpyProcess(ProcessWorker):
            def run(self):
                """Override."""
                self._db.ping()
                time.sleep(0.1)

        n_clients = len(self._db.client_list())

        proc = DumpyProcess('dummy')
        proc.start()
        # child process share the same connection as that in the main process
        self.assertEqual(n_clients, len(self._db.client_list()))

        proc.join()

    def testCreatePubSub(self):
        class SubHost:
            sub = RedisSubscriber('abc')

        class PSubHost:
            sub = RedisPSubscriber('abc')

        n_clients = len(self._db.client_list())
        sub_host = SubHost()
        self.assertEqual(n_clients, len(self._db.client_list()))
        # add a connection when first called
        sub_host.sub.get_message()
        n_clients += 1
        self.assertEqual(n_clients, len(self._db.client_list()))

        psub_host = PSubHost()
        self.assertEqual(n_clients, len(self._db.client_list()))
        # add a connection when first called
        psub_host.sub.get_message()
        n_clients += 1
        self.assertEqual(n_clients, len(self._db.client_list()))

    def testTrackingConnections(self):
        self.assertTrue(bool(_global_connections))
        # clear the current registrations
        _global_connections.clear()
        self.assertFalse(bool(_global_connections))

        class Host:
            db = RedisConnection()

        RedisConnection()
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
