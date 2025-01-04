# Communicator Object

import hashlib
import io
import json
import logging
import pickle
import time
from collections import defaultdict

import pika
import torch
from colorama import Fore

from app.config import config
from app.config.logger import fed_logger

logging.getLogger("pika").setLevel(logging.FATAL)


class Communicator(object):
    def __init__(self):
        self.connection = None
        self.channel = None
        self.url = None
        self.should_close = False
        self.send_bug = False

    def close_connection(self, ch, c):
        self.should_close = True
        if ch.is_open:
            ch.close()
        if c.is_open:
            c.close()
        while not (c.is_closed and ch.close):
            pass
        self.should_close = False
        # self.connection = None
        # self.channel = None

    def open_connection(self, url=None):
        fed_logger.debug("connecting")
        if url is None:
            url = config.mq_host
        else:
            url = config.EDGE_MQ_MAP[url]
        # url = config.mq_host
        self.url = url

        # url = config.mq_host

        # else:
        #     url = config.mq_host + url + ':5672/%2F'
        # url = config.mq_url
        # fed_logger.info(Fore.RED + f"{url}")
        connection = None
        while connection is None:
            connection = self.connect(url)

        channel = None
        while channel is None:
            try:
                channel = connection.channel()
                channel.basic_recover(requeue=True)
                channel.confirm_delivery()
            except Exception:
                continue
        # self.connection = self.connect(url)
        # self.connection.ioloop.start()
        fed_logger.debug("connection established")
        return channel, connection

    def connect(self, url):
        try:
            return pika.BlockingConnection(pika.ConnectionParameters(host=url, port=config.mq_port,
                                                                     credentials=pika.PlainCredentials(
                                                                         username=config.mq_user,
                                                                         password=config.mq_pass)))
        except Exception as e:
            pass

    def on_connection_open(self, _unused_connection):
        fed_logger.info("opened")
        self.channel = self.connection.channel()
        fed_logger.info("connected")

    def reconnect(self, _unused_connection: pika.BlockingConnection, channel):
        if _unused_connection is not None and not self.should_close and (_unused_connection.is_closed):
            self.connection = None
            self.channel = None
            return self.open_connection(self.url)
        elif _unused_connection is not None and not (_unused_connection.is_closed) and channel.close:
            channel = _unused_connection.channel()
            channel.confirm_delivery()
            return channel, _unused_connection
        else:
            return self.open_connection(self.url)

    @staticmethod
    def declare_queue_if_not_exist(exchange, msg, channel):
        queue = None
        while queue is None:
            try:
                channel.exchange_declare(exchange=config.cluster + "." + exchange, durable=True,
                                         exchange_type='topic')
                queue = channel.queue_declare(queue=config.cluster + "." + msg[0] + "." + exchange)
                channel.queue_bind(exchange=config.cluster + "." + exchange,
                                   queue=config.cluster + "." + msg[0] + "." + exchange,
                                   routing_key=config.cluster + "." + msg[0] + "." + exchange)
            except Exception:
                continue
        return queue

    def send_msg(self, exchange, msg, is_weight=False, url=None):
        bb = self.serialize_message(msg, is_weight)
        if not isinstance(bb, bytes):
            bb = bb.encode('utf-8')
        message_id = hashlib.sha256(bb).hexdigest()
        chunks = chunk_message(bb)
        total_chunks = len(chunks)

        channel, connection = self.open_connection(url)
        queue = self.declare_queue_if_not_exist(exchange, msg, channel)

        for idx, chunk in enumerate(chunks):
            published = False
            self.send_bug = False
            while not published:
                try:
                    self.reconnect(connection, channel)
                    fed_logger.debug(Fore.GREEN + f"publishing {config.cluster}.{msg[0]}.{exchange}")
                    properties = pika.BasicProperties(
                        delivery_mode=pika.DeliveryMode.Transient,
                        headers={
                            "message_id": message_id,
                            "chunk_index": idx,
                            "total_chunks": total_chunks
                        }
                    )
                    channel.basic_publish(
                        exchange=config.cluster + "." + exchange,
                        routing_key=config.cluster + "." + msg[0] + "." + exchange,
                        body=chunk,
                        mandatory=True,
                        properties=properties
                    )
                    # if self.send_bug:
                    fed_logger.debug(Fore.RED + f"published {config.cluster}.{msg[0]}.{exchange}")
                    published = True

                except Exception as e:
                    if published:
                        continue
                    self.send_bug = True
                    fed_logger.error(Fore.RED + f"Failed to send chunk {idx}: {e}")
                    continue
                time.sleep(1)


        self.close_connection(channel, connection)
        fed_logger.info(Fore.GREEN + f"Published message in {total_chunks} chunks.")

    def recv_msg(self, exchange, expect_msg_type: str = None, is_weight=False, url=None):
        channel, connection = self.open_connection(url)
        received_chunks = defaultdict(dict)  # Store chunks by message_id
        fed_logger.debug(Fore.YELLOW + f"Receiving {config.cluster}.{expect_msg_type}.{exchange}")

        try:
            self.reconnect(connection, channel)
            queue = channel.queue_declare(queue=config.cluster + "." + expect_msg_type + "." + exchange)
            channel.exchange_declare(exchange=config.cluster + "." + exchange, durable=True, exchange_type='topic')
            channel.queue_bind(
                exchange=config.cluster + "." + exchange,
                queue=config.cluster + "." + expect_msg_type + "." + exchange,
                routing_key=config.cluster + "." + expect_msg_type + "." + exchange
            )

            for method_frame, properties, body in channel.consume(
                    queue=config.cluster + "." + expect_msg_type + "." + exchange):
                message_id = properties.headers["message_id"]
                chunk_index = properties.headers["chunk_index"]
                total_chunks = properties.headers["total_chunks"]

                # Save the chunk
                received_chunks[message_id][chunk_index] = body

                # Check if all chunks are received
                if len(received_chunks[message_id]) == total_chunks:
                    all_chunks = [received_chunks[message_id][i] for i in range(total_chunks)]
                    full_message = b"".join(all_chunks)
                    del received_chunks[message_id]  # Cleanup

                    res = self.deserialize_message(full_message, is_weight)
                    channel.basic_ack(method_frame.delivery_tag)
                    channel.stop_consuming()
                    self.close_connection(channel, connection)

                    msg = [expect_msg_type]
                    msg.extend(res)
                    fed_logger.debug(Fore.CYAN + f"received {msg[0]},{type(msg[1])},{is_weight}")
                    return msg

        except Exception as e:
            fed_logger.exception(Fore.RED + f"Error receiving message: {e}")
            self.close_connection(channel, connection)
            raise

    @staticmethod
    def serialize_message(msg, is_weight=False):
        if is_weight:
            ll = []
            for o in msg[1:]:
                to_send = io.BytesIO()
                torch.save(o, to_send, _use_new_zipfile_serialization=False)
                to_send.seek(0)
                ll.append(bytes(to_send.read()))
            return pickle.dumps(ll)
        else:
            return json.dumps(msg[1:])

    @staticmethod
    def deserialize_message(msg, is_weight=False):
        if is_weight:
            fl = []
            ll = pickle.loads(msg)
            for o in ll:
                fl.append(torch.load(io.BytesIO(o)))
            return fl
        else:
            return json.loads(msg)


def chunk_message(msg, chunk_size=500 * 1024 * 1024):
    """Splits the message into chunks."""
    chunks = []
    # fed_logger.info(len(msg))
    for i in range(0, len(msg), chunk_size):
        if len(msg[i:]) > chunk_size:
            chunks.append(msg[i:i + chunk_size])
        else:
            chunks.append(msg[i:])
    return chunks
