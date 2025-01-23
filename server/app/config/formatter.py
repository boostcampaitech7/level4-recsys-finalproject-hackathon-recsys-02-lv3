import logging
from datetime import datetime
import pytz

class TimezoneFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, timezone='Asia/Seoul'):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.timezone = pytz.timezone(timezone)

    def formatTime(self, record, datefmt=None):
        # 로그의 timestamp를 Asia/Seoul 시간대로 변환
        dt = datetime.fromtimestamp(record.created, self.timezone)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()
