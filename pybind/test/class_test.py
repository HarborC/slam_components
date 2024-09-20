from pyslamcpts import Frame, SparseMap, Calibration

class KeyFrame(Frame):
    def __init__(self, id: int, cam_num: int) -> None:
        super().__init__(id, cam_num)
        self.other = None
        
class Map(SparseMap):
    def __init__(self, calib: Calibration) -> None:
        super().__init__(calib)
        self.dense_map = None
        
kf = KeyFrame(1, 0)
print(kf.id())
print(kf.get_body_pose())

calib = Calibration()
map = Map(calib)
map.add_keyframe(kf)

print(map.get_frame_ids())
