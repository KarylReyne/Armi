from processing.thread_with_exec import ThreadWithExc, ThreadKillerException


class ThreadManager(object):
    def __init__(self):
        self.active_threads = {}

    def is_busy(self):
        bool = False
        for id in self.active_threads:
            bool = bool and self.active_threads[id].is_alive()
        return bool

    def execute_as_new_process(self, id=None, target=None, args=None, join=True):
        t = ThreadWithExc(target=target, args=args)
        self.active_threads[id] = t
        t.start()
        if join:
            t.join()

    def kill_separate_process(self, id):
        try:
            self.active_threads[id].raiseExc(ThreadKillerException)
        except Exception as e:
            print("kill_separate_process: {}".format(e))
