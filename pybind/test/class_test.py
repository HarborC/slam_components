from pyslamcpts import Frame, SparseMap, Calibration

class KeyFrame(Frame):
    def __init__(self, id: int, cam_num: int) -> None:
        super().__init__(id, cam_num)
        self.other = 1
        
class Map(SparseMap):
    def __init__(self, calib: Calibration):
        SparseMap.__init__(self, calib)
        
kf = KeyFrame(1, 0)
print(kf.id())
print(kf.get_body_pose())

f = Frame(2, 0)

calib = Calibration()
map = Map(calib)
map.add_keyframe(kf)
map.add_keyframe(f)

print(map.get_frame_ids())

kf.other = 2
kf2 = map.get_frame(1)
f2 = map.get_frame(2)
print(type(kf2))
print(type(f2))
print(kf2.other)
print(f2.id())