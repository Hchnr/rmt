"""Extend request waiting without changing frozen model/generation code."""
import argparse
import math
import queue
import sys
from rmt.inference import server


def configure_timeout(seconds):
    if not math.isfinite(seconds) or seconds<=0:raise ValueError('Positive finite response timeout required')
    def submit(self,prompt,cfg):
        response=queue.Queue(maxsize=1)
        self.queue.put((prompt,cfg,response),timeout=5)
        result=response.get(timeout=seconds)
        if isinstance(result,Exception):raise result
        return result
    server.Batcher.submit=submit


def main():
    p=argparse.ArgumentParser();p.add_argument('--response-timeout',type=float,required=True)
    args,remaining=p.parse_known_args()
    configure_timeout(args.response_timeout)
    sys.argv=['rmt.inference.server',*remaining]
    server.main()

if __name__=='__main__':main()
