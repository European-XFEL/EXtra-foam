import unittest
import time

from karaboFAI.services import start_redis_server
from karaboFAI.ipc import (
    redis_connection, redis_subscribe, redis_psubscribe
)
from karaboFAI.pipeline.worker import ProcessWorker, ProcessManager


class TestRedisConnection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        start_redis_server()

    @classmethod
    def tearDownClass(cls):
        ProcessManager.shutdown_redis()

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
        n_clients = len(redis_connection().client_list())
        redis_subscribe('abc')
        n_clients += 1
        self.assertEqual(n_clients, len(redis_connection().client_list()))

        redis_psubscribe('abc?')
        n_clients += 1
        self.assertEqual(n_clients, len(redis_connection().client_list()))
