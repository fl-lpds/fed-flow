import logging

# logging.basicConfig(filename="/fed-flow/app/logs/log1.log", filemode='w', level=logging.INFO,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("requests").setLevel(logging.WARNING)
fed_logger = logging.getLogger(__name__)
