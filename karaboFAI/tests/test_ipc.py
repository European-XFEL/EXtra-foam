import unittest
import time

from karaboFAI.logger import logger
from karaboFAI.services import start_redis_server
from karaboFAI.ipc import (
    redis_connection, RedisConnection, RedisSubscriber, RedisPSubscriber
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

    def testCreateConnection(self):
        # TODO: clean the clients from other tests
        n_clients = len(redis_connection().client_list())

        db1 = redis_connection()
        self.assertTrue(db1.ping())
        db2 = redis_connection()
        self.assertTrue(db2.ping())
        self.assertIs(db1, db2)
        # no new client should be created
        self.assertEqual(n_clients, len(db1.client_list()))

        db1_bytes = redis_connection(decode_responses=False)
        self.assertTrue(db1_bytes.ping())
        db2_bytes = redis_connection(decode_responses=False)
        self.assertTrue(db2_bytes.ping())
        self.assertIs(db1_bytes, db2_bytes)
        # expect to have one more client without decode response
        n_clients += 1
        self.assertEqual(n_clients, len(db1_bytes.client_list()))

        self.assertIsNot(db1, db1_bytes)

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

        n_clients = len(redis_connection().client_list())
        sub_host = SubHost()
        self.assertEqual(n_clients, len(redis_connection().client_list()))
        # add a connection when first called
        sub_host.sub.get_message()
        n_clients += 1
        self.assertEqual(n_clients, len(redis_connection().client_list()))

        class PSubHost:
            sub = RedisPSubscriber('abc')

        psub_host = PSubHost()
        self.assertEqual(n_clients, len(redis_connection().client_list()))
        # add a connection when first called
        psub_host.sub.get_message()
        n_clients += 1
        self.assertEqual(n_clients, len(redis_connection().client_list()))
