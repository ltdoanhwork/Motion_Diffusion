# Auto-extracted skeleton geometry from BVH

bone_pairs = [
    (0, 1),  # Hips -> Spine
    (1, 2),  # Spine -> Spine1
    (2, 3),  # Spine1 -> Spine2
    (3, 4),  # Spine2 -> Spine3
    (4, 5),  # Spine3 -> Neck
    (5, 6),  # Neck -> Neck1
    (6, 7),  # Neck1 -> Head
    (7, 8),  # Head -> HeadEnd
    (8, 9),  # HeadEnd -> HeadEnd_Nub
    (4, 10),  # Spine3 -> RightShoulder
    (10, 11),  # RightShoulder -> RightArm
    (11, 12),  # RightArm -> RightForeArm
    (12, 13),  # RightForeArm -> RightHand
    (13, 14),  # RightHand -> RightHandMiddle1
    (14, 15),  # RightHandMiddle1 -> RightHandMiddle2
    (15, 16),  # RightHandMiddle2 -> RightHandMiddle3
    (16, 17),  # RightHandMiddle3 -> RightHandMiddle4
    (17, 18),  # RightHandMiddle4 -> RightHandMiddle4_Nub
    (13, 19),  # RightHand -> RightHandRing
    (19, 20),  # RightHandRing -> RightHandRing1
    (20, 21),  # RightHandRing1 -> RightHandRing2
    (21, 22),  # RightHandRing2 -> RightHandRing3
    (22, 23),  # RightHandRing3 -> RightHandRing4
    (23, 24),  # RightHandRing4 -> RightHandRing4_Nub
    (19, 25),  # RightHandRing -> RightHandPinky
    (25, 26),  # RightHandPinky -> RightHandPinky1
    (26, 27),  # RightHandPinky1 -> RightHandPinky2
    (27, 28),  # RightHandPinky2 -> RightHandPinky3
    (28, 29),  # RightHandPinky3 -> RightHandPinky4
    (29, 30),  # RightHandPinky4 -> RightHandPinky4_Nub
    (13, 31),  # RightHand -> RightHandIndex
    (31, 32),  # RightHandIndex -> RightHandIndex1
    (32, 33),  # RightHandIndex1 -> RightHandIndex2
    (33, 34),  # RightHandIndex2 -> RightHandIndex3
    (34, 35),  # RightHandIndex3 -> RightHandIndex4
    (35, 36),  # RightHandIndex4 -> RightHandIndex4_Nub
    (31, 37),  # RightHandIndex -> RightHandThumb1
    (37, 38),  # RightHandThumb1 -> RightHandThumb2
    (38, 39),  # RightHandThumb2 -> RightHandThumb3
    (39, 40),  # RightHandThumb3 -> RightHandThumb4
    (40, 41),  # RightHandThumb4 -> RightHandThumb4_Nub
    (4, 42),  # Spine3 -> LeftShoulder
    (42, 43),  # LeftShoulder -> LeftArm
    (43, 44),  # LeftArm -> LeftForeArm
    (44, 45),  # LeftForeArm -> LeftHand
    (45, 46),  # LeftHand -> LeftHandMiddle1
    (46, 47),  # LeftHandMiddle1 -> LeftHandMiddle2
    (47, 48),  # LeftHandMiddle2 -> LeftHandMiddle3
    (48, 49),  # LeftHandMiddle3 -> LeftHandMiddle4
    (49, 50),  # LeftHandMiddle4 -> LeftHandMiddle4_Nub
    (45, 51),  # LeftHand -> LeftHandRing
    (51, 52),  # LeftHandRing -> LeftHandRing1
    (52, 53),  # LeftHandRing1 -> LeftHandRing2
    (53, 54),  # LeftHandRing2 -> LeftHandRing3
    (54, 55),  # LeftHandRing3 -> LeftHandRing4
    (55, 56),  # LeftHandRing4 -> LeftHandRing4_Nub
    (51, 57),  # LeftHandRing -> LeftHandPinky
    (57, 58),  # LeftHandPinky -> LeftHandPinky1
    (58, 59),  # LeftHandPinky1 -> LeftHandPinky2
    (59, 60),  # LeftHandPinky2 -> LeftHandPinky3
    (60, 61),  # LeftHandPinky3 -> LeftHandPinky4
    (61, 62),  # LeftHandPinky4 -> LeftHandPinky4_Nub
    (45, 63),  # LeftHand -> LeftHandIndex
    (63, 64),  # LeftHandIndex -> LeftHandIndex1
    (64, 65),  # LeftHandIndex1 -> LeftHandIndex2
    (65, 66),  # LeftHandIndex2 -> LeftHandIndex3
    (66, 67),  # LeftHandIndex3 -> LeftHandIndex4
    (67, 68),  # LeftHandIndex4 -> LeftHandIndex4_Nub
    (63, 69),  # LeftHandIndex -> LeftHandThumb1
    (69, 70),  # LeftHandThumb1 -> LeftHandThumb2
    (70, 71),  # LeftHandThumb2 -> LeftHandThumb3
    (71, 72),  # LeftHandThumb3 -> LeftHandThumb4
    (72, 73),  # LeftHandThumb4 -> LeftHandThumb4_Nub
    (0, 74),  # Hips -> RightUpLeg
    (74, 75),  # RightUpLeg -> RightLeg
    (75, 76),  # RightLeg -> RightFoot
    (76, 77),  # RightFoot -> RightForeFoot
    (77, 78),  # RightForeFoot -> RightToeBase
    (78, 79),  # RightToeBase -> RightToeBaseEnd
    (79, 80),  # RightToeBaseEnd -> RightToeBaseEnd_Nub
    (0, 81),  # Hips -> LeftUpLeg
    (81, 82),  # LeftUpLeg -> LeftLeg
    (82, 83),  # LeftLeg -> LeftFoot
    (83, 84),  # LeftFoot -> LeftForeFoot
    (84, 85),  # LeftForeFoot -> LeftToeBase
    (85, 86),  # LeftToeBase -> LeftToeBaseEnd
    (86, 87),  # LeftToeBaseEnd -> LeftToeBaseEnd_Nub
]

skeleton_tree = [
    (0, 1, [0.0, 6.269896, -2.264934]),  # Hips -> Spine
    (1, 2, [0.0, 12.478628, -2.20032]),  # Spine -> Spine1
    (2, 3, [0.0, 12.622911, -1.104362]),  # Spine1 -> Spine2
    (3, 4, [0.0, 12.671129, 0.0]),  # Spine2 -> Spine3
    (4, 5, [0.0, 16.291454, 1.629145]),  # Spine3 -> Neck
    (5, 6, [0.0, 3.456791, 0.30243]),  # Neck -> Neck1
    (6, 7, [0.0, 3.417274, 0.602559]),  # Neck1 -> Head
    (7, 8, [0.0, 9.72773, -0.0]),  # Head -> HeadEnd
    (8, 9, [0.0, 9.727722, -0.0]),  # HeadEnd -> HeadEnd_Nub
    (4, 10, [0.0, 11.636753, 5.87917]),  # Spine3 -> RightShoulder
    (10, 11, [-19.553394, 8e-06, 0.0]),  # RightShoulder -> RightArm
    (11, 12, [-30.623638, 1.1e-05, 0.0]),  # RightArm -> RightForeArm
    (12, 13, [-25.458359, 8e-06, 0.0]),  # RightForeArm -> RightHand
    (13, 14, [-9.328308, 4e-06, 0.0]),  # RightHand -> RightHandMiddle1
    (14, 15, [-4.931488, 0.0, 0.0]),  # RightHandMiddle1 -> RightHandMiddle2
    (15, 16, [-3.177132, 0.0, 0.0]),  # RightHandMiddle2 -> RightHandMiddle3
    (16, 17, [-1.92765, 0.0, 0.0]),  # RightHandMiddle3 -> RightHandMiddle4
    (17, 18, [-1.927643, 0.0, 0.0]),  # RightHandMiddle4 -> RightHandMiddle4_Nub
    (13, 19, [-0.25, -0.25, -0.855603]),  # RightHand -> RightHandRing
    (19, 20, [-8.228668, 4e-06, -0.742586]),  # RightHandRing -> RightHandRing1
    (20, 21, [-4.579102, 0.0, -0.413236]),  # RightHandRing1 -> RightHandRing2
    (21, 22, [-3.089668, 0.0, -0.278823]),  # RightHandRing2 -> RightHandRing3
    (22, 23, [-1.908813, 0.0, -0.172259]),  # RightHandRing3 -> RightHandRing4
    (23, 24, [-1.908813, 0.0, -0.172258]),  # RightHandRing4 -> RightHandRing4_Nub
    (19, 25, [-0.172089, -0.25, -0.87461]),  # RightHandRing -> RightHandPinky
    (25, 26, [-6.812508, 4e-06, -1.523409]),  # RightHandPinky -> RightHandPinky1
    (26, 27, [-3.617294, 0.0, -0.808899]),  # RightHandPinky1 -> RightHandPinky2
    (27, 28, [-2.311783, 0.0, -0.516959]),  # RightHandPinky2 -> RightHandPinky3
    (28, 29, [-1.725502, 0.0, -0.385855]),  # RightHandPinky3 -> RightHandPinky4
    (29, 30, [-1.725502, 0.0, -0.385856]),  # RightHandPinky4 -> RightHandPinky4_Nub
    (13, 31, [-0.25, -0.25, 0.855603]),  # RightHand -> RightHandIndex
    (31, 32, [-9.013367, 4e-06, 0.8134]),  # RightHandIndex -> RightHandIndex1
    (32, 33, [-4.737785, 0.0, 0.427556]),  # RightHandIndex1 -> RightHandIndex2
    (33, 34, [-2.835075, 0.0, 0.255847]),  # RightHandIndex2 -> RightHandIndex3
    (34, 35, [-1.745514, 0.0, 0.157522]),  # RightHandIndex3 -> RightHandIndex4
    (35, 36, [-1.745514, 0.0, 0.157522]),  # RightHandIndex4 -> RightHandIndex4_Nub
    (31, 37, [-0.172089, -0.75, 0.87461]),  # RightHandIndex -> RightHandThumb1
    (37, 38, [-5.4757, 0.845421, 2.271264]),  # RightHandThumb1 -> RightHandThumb2
    (38, 39, [-3.582893, 0.553185, 1.486152]),  # RightHandThumb2 -> RightHandThumb3
    (39, 40, [-2.19529, 0.338943, 0.910584]),  # RightHandThumb3 -> RightHandThumb4
    (40, 41, [-2.19529, 0.338943, 0.910584]),  # RightHandThumb4 -> RightHandThumb4_Nub
    (4, 42, [0.0, 11.636753, 5.87917]),  # Spine3 -> LeftShoulder
    (42, 43, [19.553394, 8e-06, 0.0]),  # LeftShoulder -> LeftArm
    (43, 44, [30.623638, 1.1e-05, 0.0]),  # LeftArm -> LeftForeArm
    (44, 45, [25.458359, 8e-06, 0.0]),  # LeftForeArm -> LeftHand
    (45, 46, [9.327454, 4e-06, 0.0]),  # LeftHand -> LeftHandMiddle1
    (46, 47, [4.935944, 0.0, 0.0]),  # LeftHandMiddle1 -> LeftHandMiddle2
    (47, 48, [3.187286, 0.0, 0.0]),  # LeftHandMiddle2 -> LeftHandMiddle3
    (48, 49, [1.919037, 0.0, 0.0]),  # LeftHandMiddle3 -> LeftHandMiddle4
    (49, 50, [1.919052, 0.0, 0.0]),  # LeftHandMiddle4 -> LeftHandMiddle4_Nub
    (45, 51, [0.25, -0.25, -0.911864]),  # LeftHand -> LeftHandRing
    (51, 52, [8.228249, 4e-06, -0.742548]),  # LeftHandRing -> LeftHandRing1
    (52, 53, [4.570602, 0.0, -0.412469]),  # LeftHandRing1 -> LeftHandRing2
    (53, 54, [3.097679, 0.0, -0.279546]),  # LeftHandRing2 -> LeftHandRing3
    (54, 55, [1.900299, 0.0, -0.17149]),  # LeftHandRing3 -> LeftHandRing4
    (55, 56, [1.900299, 0.0, -0.17149]),  # LeftHandRing4 -> LeftHandRing4_Nub
    (51, 57, [0.16703, -0.25, -0.930643]),  # LeftHandRing -> LeftHandPinky
    (57, 58, [6.794594, 4e-06, -1.519404]),  # LeftHandPinky -> LeftHandPinky1
    (58, 59, [3.623344, 0.0, -0.810251]),  # LeftHandPinky1 -> LeftHandPinky2
    (59, 60, [2.307434, 0.0, -0.515988]),  # LeftHandPinky2 -> LeftHandPinky3
    (60, 61, [1.717804, 0.0, -0.384134]),  # LeftHandPinky3 -> LeftHandPinky4
    (61, 62, [1.717804, 0.0, -0.384135]),  # LeftHandPinky4 -> LeftHandPinky4_Nub
    (45, 63, [0.25, -0.25, 0.911864]),  # LeftHand -> LeftHandIndex
    (63, 64, [8.99826, 4e-06, 0.812038]),  # LeftHandIndex -> LeftHandIndex1
    (64, 65, [4.745354, 0.0, 0.428239]),  # LeftHandIndex1 -> LeftHandIndex2
    (65, 66, [2.836342, 0.0, 0.255961]),  # LeftHandIndex2 -> LeftHandIndex3
    (66, 67, [1.737732, 0.0, 0.15682]),  # LeftHandIndex3 -> LeftHandIndex4
    (67, 68, [1.737732, 0.0, 0.156819]),  # LeftHandIndex4 -> LeftHandIndex4_Nub
    (63, 69, [0.16703, -0.75, 0.930643]),  # LeftHandIndex -> LeftHandThumb1
    (69, 70, [5.434509, 0.839062, 2.254179]),  # LeftHandThumb1 -> LeftHandThumb2
    (70, 71, [3.593353, 0.554794, 1.490485]),  # LeftHandThumb2 -> LeftHandThumb3
    (71, 72, [2.185493, 0.337429, 0.906521]),  # LeftHandThumb3 -> LeftHandThumb4
    (72, 73, [2.185501, 0.337429, 0.906523]),  # LeftHandThumb4 -> LeftHandThumb4_Nub
    (0, 74, [-8.246678, 0.0, 0.0]),  # Hips -> RightUpLeg
    (74, 75, [0.0, -42.827576, 0.0]),  # RightUpLeg -> RightLeg
    (75, 76, [0.0, -43.165855, 0.0]),  # RightLeg -> RightFoot
    (76, 77, [0.0, -2.559708, 0.0]),  # RightFoot -> RightForeFoot
    (77, 78, [0.0, 0.0, 10.024612]),  # RightForeFoot -> RightToeBase
    (78, 79, [0.0, 0.0, 14.750254]),  # RightToeBase -> RightToeBaseEnd
    (79, 80, [0.0, 0.0, 14.75025]),  # RightToeBaseEnd -> RightToeBaseEnd_Nub
    (0, 81, [8.246678, 0.0, 0.0]),  # Hips -> LeftUpLeg
    (81, 82, [0.0, -42.827576, 0.0]),  # LeftUpLeg -> LeftLeg
    (82, 83, [0.0, -43.165855, 0.0]),  # LeftLeg -> LeftFoot
    (83, 84, [0.0, -2.559708, 0.0]),  # LeftFoot -> LeftForeFoot
    (84, 85, [0.0, 0.0, 10.024612]),  # LeftForeFoot -> LeftToeBase
    (85, 86, [0.0, 0.0, 14.330561]),  # LeftToeBase -> LeftToeBaseEnd
    (86, 87, [0.0, 0.0, 14.330564]),  # LeftToeBaseEnd -> LeftToeBaseEnd_Nub
]

joint_names = [
    'Hips',  # 0
    'Spine',  # 1
    'Spine1',  # 2
    'Spine2',  # 3
    'Spine3',  # 4
    'Neck',  # 5
    'Neck1',  # 6
    'Head',  # 7
    'HeadEnd',  # 8
    'HeadEnd_Nub',  # 9
    'RightShoulder',  # 10
    'RightArm',  # 11
    'RightForeArm',  # 12
    'RightHand',  # 13
    'RightHandMiddle1',  # 14
    'RightHandMiddle2',  # 15
    'RightHandMiddle3',  # 16
    'RightHandMiddle4',  # 17
    'RightHandMiddle4_Nub',  # 18
    'RightHandRing',  # 19
    'RightHandRing1',  # 20
    'RightHandRing2',  # 21
    'RightHandRing3',  # 22
    'RightHandRing4',  # 23
    'RightHandRing4_Nub',  # 24
    'RightHandPinky',  # 25
    'RightHandPinky1',  # 26
    'RightHandPinky2',  # 27
    'RightHandPinky3',  # 28
    'RightHandPinky4',  # 29
    'RightHandPinky4_Nub',  # 30
    'RightHandIndex',  # 31
    'RightHandIndex1',  # 32
    'RightHandIndex2',  # 33
    'RightHandIndex3',  # 34
    'RightHandIndex4',  # 35
    'RightHandIndex4_Nub',  # 36
    'RightHandThumb1',  # 37
    'RightHandThumb2',  # 38
    'RightHandThumb3',  # 39
    'RightHandThumb4',  # 40
    'RightHandThumb4_Nub',  # 41
    'LeftShoulder',  # 42
    'LeftArm',  # 43
    'LeftForeArm',  # 44
    'LeftHand',  # 45
    'LeftHandMiddle1',  # 46
    'LeftHandMiddle2',  # 47
    'LeftHandMiddle3',  # 48
    'LeftHandMiddle4',  # 49
    'LeftHandMiddle4_Nub',  # 50
    'LeftHandRing',  # 51
    'LeftHandRing1',  # 52
    'LeftHandRing2',  # 53
    'LeftHandRing3',  # 54
    'LeftHandRing4',  # 55
    'LeftHandRing4_Nub',  # 56
    'LeftHandPinky',  # 57
    'LeftHandPinky1',  # 58
    'LeftHandPinky2',  # 59
    'LeftHandPinky3',  # 60
    'LeftHandPinky4',  # 61
    'LeftHandPinky4_Nub',  # 62
    'LeftHandIndex',  # 63
    'LeftHandIndex1',  # 64
    'LeftHandIndex2',  # 65
    'LeftHandIndex3',  # 66
    'LeftHandIndex4',  # 67
    'LeftHandIndex4_Nub',  # 68
    'LeftHandThumb1',  # 69
    'LeftHandThumb2',  # 70
    'LeftHandThumb3',  # 71
    'LeftHandThumb4',  # 72
    'LeftHandThumb4_Nub',  # 73
    'RightUpLeg',  # 74
    'RightLeg',  # 75
    'RightFoot',  # 76
    'RightForeFoot',  # 77
    'RightToeBase',  # 78
    'RightToeBaseEnd',  # 79
    'RightToeBaseEnd_Nub',  # 80
    'LeftUpLeg',  # 81
    'LeftLeg',  # 82
    'LeftFoot',  # 83
    'LeftForeFoot',  # 84
    'LeftToeBase',  # 85
    'LeftToeBaseEnd',  # 86
    'LeftToeBaseEnd_Nub',  # 87
]
