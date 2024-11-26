import fcntl


class FileLock:
    """
    文件锁类：用于处理文件的读写锁定机制
    """
    def __init__(self, filename):
        """
        初始化文件锁
        Args:
            filename: 需要锁定的文件名
        """
        self.filename = filename
        self.handle = None

    def acquire_read_lock(self):
        """
        获取共享读锁
        - 使用 LOCK_SH 实现共享锁，允许多个进程同时读取
        - LOCK_NB 设置为非阻塞模式，如果无法获取锁则立即返回
        """
        self.handle = open(self.filename + '.lock', 'r')
        fcntl.flock(self.handle, fcntl.LOCK_SH | fcntl.LOCK_NB)

    def acquire_write_lock(self):
        """
        获取独占写锁
        - 使用 LOCK_EX 实现排他锁，确保同一时间只有一个进程可以写入
        - LOCK_NB 设置为非阻塞模式，如果无法获取锁则立即返回
        """
        self.handle = open(self.filename + '.lock', 'w')
        fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def release_lock(self):
        """
        释放锁
        - 使用 LOCK_UN 解除锁定
        - 关闭文件句柄并清理资源
        """
        if self.handle is not None:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None
