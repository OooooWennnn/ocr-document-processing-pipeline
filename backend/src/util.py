from datetime import datetime
from zoneinfo import ZoneInfo

def now_toronto() -> datetime:
    return datetime.now(ZoneInfo("America/Toronto"))