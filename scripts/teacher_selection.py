"""Explicit source-bucket filtering; programming cues are not a semantic oracle."""
import re
PROGRAMMING_PATTERN = r'```|\b(python|javascript|typescript|java|rust|golang|sql|html|css|bash|programming|function|algorithm|debug|code|coding|script|implementation)\b|c\+\+|c#|编程|代码|函数|算法|程序|调试'

def eligible_teacher_prompt(row):
    return row['domain']!='code' or bool(re.search(PROGRAMMING_PATTERN,row['messages'][0]['content'],re.I))
