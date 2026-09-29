"""Stable request identity excludes framework-generated message IDs."""
from .prepare import digest


def request_seed(messages,master_seed):
    semantic=[]
    for message in messages:
        role=message['role'] if isinstance(message,dict) else message.role
        content=message['content'] if isinstance(message,dict) else message.content
        if not isinstance(content,str):raise ValueError('This protocol supports text messages only')
        semantic.append({'role':role,'content':content})
    return int(digest([master_seed,semantic])[:8],16)
