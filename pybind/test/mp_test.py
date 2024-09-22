import torch
import torch.multiprocessing as mp
import torch.nn as nn
from pyslamcpts import Frame, SparseMap, Calibration

class KeyFrame(nn.Module, Frame):
    def __init__(self, id: int, cam_num: int) -> None:
        nn.Module.__init__(self)
        Frame.__init__(self, id, cam_num)
        # 使用 torch.Tensor 使其支持共享内存
        self.other = torch.zeros(1, dtype=torch.float32).share_memory_()

def worker(shared_tensor):
    # 在进程中访问共享张量
    while True:
        print(f"Worker process received tensor value: {shared_tensor.other.item()}")

if __name__ == "__main__":
    mp.set_start_method('spawn')

    idx = 0
    tensor = KeyFrame(idx, 0)

    # 创建子进程
    processes = []
    p1 = mp.Process(target=worker, args=(tensor,))
    p1.start()

    # 主进程中更新共享张量
    while True:
        tensor.other[0] = idx  # 更新张量值
        idx += 1

    p1.join()
