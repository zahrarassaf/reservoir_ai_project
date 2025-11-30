import logging
import sys

# Fix Unicode logging for Windows
class UnicodeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except UnicodeEncodeError:
            # Replace Unicode characters with ASCII equivalents
            record.msg = record.msg.encode('ascii', 'ignore').decode('ascii')
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        UnicodeStreamHandler()
    ]
)
logger = logging.getLogger(__name__)
