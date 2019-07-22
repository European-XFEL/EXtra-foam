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
from .serialization import deserialize_image, serialize_image


class CommandProxy:

    __db = RedisConnection()
    __ref_sub = RedisSubscriber("command:reference_image",
                                decode_responses=False)
    __mask_sub = RedisSubscriber("command:image_mask",
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
        sub = self.__ref_sub
        if sub is None:
            return

        ref = None
        # process all messages related to reference
        while True:
            msg = sub.get_message(ignore_subscribe_messages=True)

            if msg is None:
                # the channel is empty
                break

            action = msg['data']
            if action == b'next':
                ref = deserialize_image(sub.get_message()['data'])
            else:
                # remove reference
                ref = -1

        return ref

    def add_mask(self, mask_region):
        self.__db.publish("command:image_mask", 'add')
        self.__db.publish("command:image_mask", str(mask_region))

    def remove_mask(self, mask_region):
        self.__db.publish("command:image_mask", 'remove')
        self.__db.publish("command:image_mask", str(mask_region))

    def set_mask(self, mask):
        """Publish the image mask in Redis."""
        self.__db.publish("command:image_mask", 'set')
        self.__db.publish("command:image_mask",
                          serialize_image(mask, is_mask=True))

    def clear_mask(self):
        """Notify to completely clear all the image mask."""
        self.__db.publish("command:image_mask", 'clear')

    def update_mask(self, mask):
        """Parse all masking operations.

        :return: a list of masking operations.
        """
        sub = self.__mask_sub
        if sub is None:
            return

        # process all messages related to mask
        while True:
            msg = sub.get_message(ignore_subscribe_messages=True)
            if msg is None:
                break

            action = msg['data']
            if action == b'set':
                mask = deserialize_image(sub.get_message()['data'], is_mask=True)
            elif action in [b'add', b'remove']:
                data = sub.get_message()['data'].decode("utf-8")
                x, y, w, h = [int(v) for v in data[1:-1].split(',')]
                if action == b'add':
                    mask[y:y+h, x:x+w] = True
                else:
                    mask[y:y+h, x:x+w] = False
            else:  # data == 'clear'
                mask = None

        return mask
