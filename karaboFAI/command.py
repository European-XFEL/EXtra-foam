"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Command Proxy.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .ipc import RedisConnection, RedisSubscriber
from .pipeline.serialization import deserialize_image, serialize_image


class CommandProxy:

    __db = RedisConnection()
    __ref_sub = RedisSubscriber("command:reference_image",
                                decode_responses=False)

    def set_ref_image(self, image):
        """Publish the reference image in Redis."""
        if image is None:
            return

        self.__db.publish("command:reference_image", 'next')
        self.__db.publish("command:reference_image", serialize_image(image))

    def remove_ref_image(self):
        """Notify to remove the current reference image."""
        self.__db.publish("command:reference_image", 'remove')

    def get_ref_image(self):
        """Try to get the reference image.

        :return: None for no update; numpy.ndarray for receiving a new
            reference image; -1 for removing the current reference image.
        """
        # process all messages related to reference
        sub = self.__ref_sub
        if sub is None:
            return

        ref = None
        while True:
            msg = sub.get_message(ignore_subscribe_messages=True)

            if msg is None:
                # the channel is empty
                break

            data = msg['data']
            if data.decode("utf-8") == 'next':
                msg = sub.get_message()
                ref = deserialize_image(msg['data'])
            else:
                # remove reference
                ref = -1
        return ref
