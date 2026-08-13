import sys
sys.path.append('.')
from app.graph import select_tools_for_message
class T:
    def __init__(self, n):
        self.name = n
tools = [T('get_recent_emails'), T('create_calendar_event'), T('create_google_meet'), T('read_file'), T('get_weather')]
print([t.name for t in select_tools_for_message('check my latest emails and calendar events', tools)])
print([t.name for t in select_tools_for_message('create a new Google Meet link', tools)])
print([t.name for t in select_tools_for_message('summarize this PDF', tools)])
