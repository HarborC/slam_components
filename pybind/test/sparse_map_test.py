from pyslamcpts import Frame, SparseMap, Calibration

calib = Calibration()

frame1 = Frame()
frame1.set_id(10)
print(frame1.id())
frame2 = Frame()
frame2.set_id(11)
print(frame2.id())

sparse_map = SparseMap(calib)
sparse_map.add_keyframe(frame1)
print(sparse_map.get_frame_ids())
sparse_map.add_keyframe(frame2)
print(sparse_map.get_frame_ids())

frame1.set_id(12)
frame1_ = sparse_map.get_frame(10)
print(frame1_.id() == frame1.id())
